"""
platform/main.py — VS AgentCore Multi-Agent Platform
==================================================================
Local mode:  talks to Supervisor on localhost:8000 via HTTP
Production:  invokes AgentCore Runtime via AWS SDK

SSE FLOW:
  UI → POST /api/v1/clinical-trial/chat (X-API-Key)
  ← StreamingResponse (text/event-stream)
    data: {"type": "token", "content": "..."}\n\n
    data: {"type": "tool_start", "name": "search_tool"}\n\n
    data: {"type": "interrupt", "question": "...", "options": [...]}\n\n
    data: {"type": "done", "latency_ms": 12345}\n\n

AGENTCORE INVOKE:
  invoke_agent_runtime(
    agentRuntimeArn=...,
    runtimeSessionId=thread_id,
    payload=json.dumps(payload),
  )
"""

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from decimal import Decimal

import boto3
import httpx

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from gateway.auth import verify_api_key, _get_api_key
from learning_pipeline import ContinuousLearningPipeline
from gateway.rate_limiter import RateLimiter
from gateway.schemas import ChatRequest, ResumeRequest
from gateway.logging_mw import LoggingMiddleware
from gateway.input_guardrail import check_input_guardrail

_executor = ThreadPoolExecutor(max_workers=20)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "VS AgentCore Multi-Agent Platform",
    description = "Clinical Trial Research Assistant — Gateway",
    version     = "1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["POST", "GET", "OPTIONS"],
    allow_headers  = ["*"],
)
app.add_middleware(LoggingMiddleware)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ── Config ────────────────────────────────────────────────────────────────
REGION         = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX     = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
LOCAL_MODE     = os.environ.get("LOCAL_MODE", "false").lower() == "true"
SUPERVISOR_URL = os.environ.get("SUPERVISOR_URL", "http://localhost:8000")
AGENT          = "clinical-trial"
rate_limiter   = RateLimiter(requests_per_minute=60)


def _get_agent_runtime_arn() -> str:
    """Fetch supervisor runtime ARN from SSM on every call — no cache."""
    ssm = boto3.client("ssm", region_name=REGION)
    for key in [
        f"{SSM_PREFIX}/agents/supervisor/runtime_arn",
        f"{SSM_PREFIX}/agent_runtime_arn",
        "/vs-agentcore/prod/agent_runtime_arn",
    ]:
        try:
            val = ssm.get_parameter(Name=key)["Parameter"]["Value"]
            log.info(f"[PLATFORM] Supervisor runtime ARN: {val}")
            return val
        except Exception:
            continue
    raise RuntimeError("Supervisor runtime ARN not found in SSM")


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Local streaming ───────────────────────────────────────────────────────
async def _stream_local(payload: dict, request_id: str):
    t0  = time.perf_counter()
    url = f"{SUPERVISOR_URL}/invocations"
    log.info(f"[PLATFORM] → Supervisor (local)  request_id={request_id}")
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", url, json=payload,
                                     headers={"Content-Type": "application/json"}) as response:
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        yield line + "\n\n"
                    else:
                        try:
                            event = json.loads(line)
                            yield _sse(event)
                        except json.JSONDecodeError:
                            continue
    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})


# ── Production streaming ──────────────────────────────────────────────────
async def _stream_agentcore(payload: dict, request_id: str):
    t0    = time.perf_counter()
    queue = asyncio.Queue()
    loop  = asyncio.get_running_loop()

    def _stream_to_queue():
        try:
            agent_arn = _get_agent_runtime_arn()
            from botocore.config import Config
            client = boto3.client(
                "bedrock-agentcore", region_name=REGION,
                config=Config(read_timeout=300, connect_timeout=10,
                              retries={"max_attempts": 0}),
            )
            log.info(f"[PLATFORM] → AgentCore  request_id={request_id}  thread={payload.get('thread_id')}")
            response = client.invoke_agent_runtime(
                agentRuntimeArn  = agent_arn,
                runtimeSessionId = payload["thread_id"],
                payload          = json.dumps(payload).encode("utf-8"),
            )
            streaming_body = response.get("response")
            if streaming_body:
                _partial = b""
                for chunk in streaming_body.iter_chunks(chunk_size=65536):
                    if chunk:
                        _partial += chunk
                        if b"\n" in _partial:
                            last_nl  = _partial.rfind(b"\n")
                            complete = _partial[:last_nl + 1]
                            _partial = _partial[last_nl + 1:]
                            loop.call_soon_threadsafe(queue.put_nowait, complete)
                if _partial:
                    loop.call_soon_threadsafe(queue.put_nowait, _partial)
            else:
                raw = response.get("body", b"")
                if raw:
                    loop.call_soon_threadsafe(queue.put_nowait, raw)
        except Exception as exc:
            log.error(f"[PLATFORM] AgentCore error: {exc}")
            err = _sse({"type": "error", "message": str(exc)}).encode()
            loop.call_soon_threadsafe(queue.put_nowait, err)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(_executor, _stream_to_queue)

    try:
        while True:
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if chunk is None:
                break
            text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    yield line + "\n\n"
                else:
                    try:
                        event = json.loads(line)
                        yield _sse(event)
                    except json.JSONDecodeError:
                        continue
    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})


def _stream(payload: dict, request_id: str):
    if LOCAL_MODE:
        return _stream_local(payload, request_id)
    return _stream_agentcore(payload, request_id)


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "mode": "local" if LOCAL_MODE else "aws"}


@app.get("/")
async def root():
    ui_path = os.path.join(static_dir, "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {"message": "VS AgentCore Platform"}


@app.post(f"/api/v1/{AGENT}/chat")
async def chat(body: ChatRequest, request: Request, _: str = Depends(verify_api_key)):
    request_id = str(uuid.uuid4())[:8]
    user_id    = getattr(request.state, "user_id", "anonymous")
    if not rate_limiter.allow(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    blocked, block_reason = check_input_guardrail(body.message)
    if blocked:
        log.warning(f"[PLATFORM] Guardrail blocked  request_id={request_id}")
        raise HTTPException(status_code=400, detail=block_reason)
    payload = {"message": body.message, "thread_id": body.thread_id,
                "domain": body.domain, "resume": False}
    log.info(f"[PLATFORM] chat  request_id={request_id}  thread={body.thread_id}")
    return StreamingResponse(_stream(payload, request_id), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive", "X-Request-Id": request_id})


@app.post(f"/api/v1/{AGENT}/resume")
async def resume(body: ResumeRequest, request: Request, _: str = Depends(verify_api_key)):
    request_id = str(uuid.uuid4())[:8]
    user_id    = getattr(request.state, "user_id", "anonymous")
    if not rate_limiter.allow(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    payload = {"message": "", "thread_id": body.thread_id, "domain": body.domain,
                "resume": True, "user_answer": body.user_answer}
    return StreamingResponse(_stream(payload, request_id), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive", "X-Request-Id": request_id})


# ── Feedback ──────────────────────────────────────────────────────────────

@app.post(f"/api/v1/{AGENT}/feedback")
async def record_feedback(request: Request, _: str = Depends(verify_api_key)):
    """
    Record thumbs up/down feedback on a supervisor trace item in DynamoDB.

    The run_id is a unique UUID generated per handler() invocation in the
    supervisor, emitted in the done SSE event, and captured by the UI.
    Feedback is stored as additional attributes on the existing trace item.

    Payload:
        run_id  : str — unique request ID (DynamoDB partition key)
        rating  : str — "positive" or "negative"
        reason  : str — reason chip selected (negative only, optional)
        comment : str — free text (negative only, optional)
    """
    body    = await request.json()
    run_id  = body.get("run_id",  "")
    rating  = body.get("rating",  "")
    reason  = body.get("reason",  "")
    comment = body.get("comment", "")

    if not run_id or rating not in ("positive", "negative"):
        raise HTTPException(status_code=400, detail="run_id and valid rating required")

    try:
        dynamo = boto3.resource("dynamodb", region_name=REGION)
        table  = dynamo.Table(_get_trace_table_name())

        update_expr = "SET feedback_rating = :r, feedback_ts = :ts"
        expr_vals   = {":r": rating, ":ts": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
        if reason:
            update_expr += ", feedback_reason = :reason"
            expr_vals[":reason"] = reason
        if comment:
            update_expr += ", feedback_comment = :comment"
            expr_vals[":comment"] = comment

        table.update_item(
            Key                       = {"run_id": run_id},
            UpdateExpression          = update_expr,
            ExpressionAttributeValues = expr_vals,
        )
        log.info(f"[PLATFORM] Feedback  run_id={run_id[:8]}  rating={rating}  reason={reason}")
        return {"ok": True}

    except Exception as exc:
        log.error(f"[PLATFORM] Feedback failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Observability ─────────────────────────────────────────────────────────

def _flatten_decimals(obj):
    if isinstance(obj, Decimal):
        return float(obj) if "." in str(obj) else int(obj)
    if isinstance(obj, list):  return [_flatten_decimals(i) for i in obj]
    if isinstance(obj, dict):  return {k: _flatten_decimals(v) for k, v in obj.items()}
    return obj


@lru_cache(maxsize=1)
def _get_trace_table_name() -> str:
    return boto3.client("ssm", region_name=REGION).get_parameter(
        Name=f"{SSM_PREFIX}/dynamodb/trace_table_name"
    )["Parameter"]["Value"]


@app.get("/observability", response_class=HTMLResponse)
async def observability_dashboard():
    html_path = os.path.join(static_dir, "traces_dashboard.html")
    try:
        with open(html_path) as f:
            html = f.read()
        api_key = _get_api_key()
        html = html.replace("</head>",
            f'<script>window.PLATFORM_API_KEY = "{api_key}";</script></head>')
        return HTMLResponse(content=html)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dashboard HTML not found")


@app.get(f"/api/v1/{AGENT}/traces")
async def list_traces(limit: int = 200, _: str = Depends(verify_api_key)):
    try:
        dynamo   = boto3.resource("dynamodb", region_name=REGION)
        table    = dynamo.Table(_get_trace_table_name())
        items    = []
        response = table.scan()
        items.extend([_flatten_decimals(i) for i in response.get("Items", [])])
        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend([_flatten_decimals(i) for i in response.get("Items", [])])
        items.sort(key=lambda x: float(x.get("ts", 0)), reverse=True)
        return items[:limit]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(f"/api/v1/{AGENT}/traces/{{thread_id}}")
async def get_trace(thread_id: str, _: str = Depends(verify_api_key)):
    try:
        from boto3.dynamodb.conditions import Attr
        dynamo = boto3.resource("dynamodb", region_name=REGION)
        table  = dynamo.Table(_get_trace_table_name())
        resp   = table.scan(FilterExpression=Attr("thread_id").eq(thread_id))
        return {"thread_id": thread_id, "traces": resp.get("Items", []),
                "count": len(resp.get("Items", []))}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Continuous Learning ──────────────────────────────────────────────────

_last_learning_report: dict = {}

@app.post(f"/api/v1/{AGENT}/learning/run")
async def run_learning_pipeline(_: str = Depends(verify_api_key)):
    """
    Trigger the continuous learning pipeline on demand.

    Steps:
      1. Fetch all traces + feedback from DynamoDB
      2. Diagnose failure patterns from negative feedback
      3. GPT-4o rewrites supervisor prompt
      4. Self-test: verify improved prompt classifies 3 probe queries correctly
      5. Auto-deploy if self-test passes (Bedrock + SSM)
      6. Detect RAG gaps from missing-info traces
      7. Generate fine-tuning JSONL from positive traces
    """
    global _last_learning_report
    try:
        pipeline = ContinuousLearningPipeline()
        report   = await pipeline.run()
        _last_learning_report = report.to_dict()
        log.info(f"[PLATFORM] Learning pipeline complete  run_id={report.run_id[:8]}")
        return _last_learning_report
    except Exception as exc:
        log.error(f"[PLATFORM] Learning pipeline error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(f"/api/v1/{AGENT}/learning/report")
async def get_learning_report(_: str = Depends(verify_api_key)):
    """Return the most recent learning pipeline report."""
    if not _last_learning_report:
        return {"status": "no_report", "message": "Run the pipeline first."}
    return _last_learning_report


@app.get(f"/api/v1/{AGENT}/prompt")
async def get_prompt_info(_: str = Depends(verify_api_key)):
    try:
        ssm   = boto3.client("ssm", region_name=REGION)
        g_id  = ssm.get_parameter(Name=f"{SSM_PREFIX}/bedrock/guardrail_id")["Parameter"]["Value"]
        g_ver = ssm.get_parameter(Name=f"{SSM_PREFIX}/bedrock/guardrail_version")["Parameter"]["Value"]
        return {"guardrail_id": g_id, "guardrail_version": g_ver, "ssm_prefix": SSM_PREFIX}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))