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


# ── Local streaming (HTTP to localhost supervisor) ────────────────────────
async def _stream_local(payload: dict, request_id: str):
    t0  = time.perf_counter()
    url = f"{SUPERVISOR_URL}/invocations"
    log.info(f"[PLATFORM] → Supervisor (local)  request_id={request_id}  thread={payload.get('thread_id')}")
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST", url,
                json    = payload,
                headers = {"Content-Type": "application/json"},
            ) as response:
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
        elapsed = round((time.perf_counter() - t0) * 1_000, 2)
        log.error(f"[PLATFORM] Stream error  request_id={request_id}  elapsed_ms={elapsed}  error={exc}")
        yield _sse({"type": "error", "message": str(exc)})


# ── Production streaming (AgentCore runtime via AWS SDK) ──────────────────
async def _stream_agentcore(payload: dict, request_id: str):
    """
    Invoke AgentCore Runtime via boto3 in a thread pool.
    Uses asyncio.Queue to pipe chunks from thread to async generator
    so the event loop is never blocked and the client sees tokens in real time.
    """
    t0    = time.perf_counter()
    queue = asyncio.Queue()
    loop  = asyncio.get_running_loop()  # get_event_loop() is deprecated in 3.10+

    def _stream_to_queue():
        try:
            agent_arn = _get_agent_runtime_arn()
            # Set read_timeout=300 — default 60s causes ReadTimeoutError
            # for complex queries that take 90-120s to complete
            from botocore.config import Config
            client    = boto3.client(
                "bedrock-agentcore",
                region_name = REGION,
                config      = Config(
                    read_timeout    = 300,   # 5 min — matches ALB idle timeout
                    connect_timeout = 10,
                    retries         = {"max_attempts": 0},  # no retries on streaming
                ),
            )

            log.info(
                f"[PLATFORM] → Supervisor (AgentCore)"
                f"  request_id={request_id}"
                f"  thread={payload.get('thread_id')}"
                f"  resume={payload.get('resume', False)}"
            )

            response = client.invoke_agent_runtime(
                agentRuntimeArn  = agent_arn,
                runtimeSessionId = payload["thread_id"],
                payload          = json.dumps(payload).encode("utf-8"),
            )

            streaming_body = response.get("response")
            if streaming_body:
                # Use 64KB chunks — chart config JSON can exceed 1KB causing
                # UTF-8 decode errors when a multi-byte char splits across chunks.
                # Also accumulate partial lines across chunks.
                _partial = b""
                for chunk in streaming_body.iter_chunks(chunk_size=65536):
                    if chunk:
                        _partial += chunk
                        # Only send complete lines (ending with \n)
                        if b"\n" in _partial:
                            last_nl = _partial.rfind(b"\n")
                            complete = _partial[:last_nl + 1]
                            _partial = _partial[last_nl + 1:]
                            loop.call_soon_threadsafe(queue.put_nowait, complete)
                # Flush any remaining partial data
                if _partial:
                    loop.call_soon_threadsafe(queue.put_nowait, _partial)
            else:
                raw = response.get("body", b"")
                if raw:
                    loop.call_soon_threadsafe(queue.put_nowait, raw)

        except Exception as exc:
            log.error(f"[PLATFORM] AgentCore invoke error: {exc}")
            err = _sse({"type": "error", "message": str(exc)}).encode()
            loop.call_soon_threadsafe(queue.put_nowait, err)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    loop.run_in_executor(_executor, _stream_to_queue)

    try:
        while True:
            try:
                # Wait max 10s for next chunk — send keepalive comment if nothing arrives
                # This prevents ALB from closing idle SSE connections during long tool calls
                chunk = await asyncio.wait_for(queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"  # SSE comment — ignored by client, resets ALB idle timer
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
        elapsed = round((time.perf_counter() - t0) * 1_000, 2)
        log.error(f"[PLATFORM] Stream error  request_id={request_id}  elapsed_ms={elapsed}  error={exc}")
        yield _sse({"type": "error", "message": str(exc)})


def _stream(payload: dict, request_id: str):
    """Route to local HTTP or AgentCore based on LOCAL_MODE."""
    if LOCAL_MODE:
        return _stream_local(payload, request_id)
    return _stream_agentcore(payload, request_id)


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":     "ok",
        "service":    "vs-agentcore-platform",
        "mode":       "local" if LOCAL_MODE else "aws",
        "supervisor": SUPERVISOR_URL if LOCAL_MODE else "AgentCore",
    }

@app.get("/")
async def root():
    ui_path = os.path.join(static_dir, "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {"message": "VS AgentCore Platform — UI not found in static/"}

@app.post(f"/api/v1/{AGENT}/chat")
async def chat(
    body:    ChatRequest,
    request: Request,
    _:       str = Depends(verify_api_key),
):
    request_id = str(uuid.uuid4())[:8]
    user_id    = getattr(request.state, "user_id", "anonymous")

    if not rate_limiter.allow(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    blocked, block_reason = check_input_guardrail(body.message)
    if blocked:
        log.warning(f"[PLATFORM] Input guardrail blocked  request_id={request_id}  reason={block_reason[:80]}")
        raise HTTPException(status_code=400, detail=block_reason)

    payload = {
        "message":   body.message,
        "thread_id": body.thread_id,
        "domain":    body.domain,
        "resume":    False,
    }

    log.info(f"[PLATFORM] → Supervisor  request_id={request_id}  thread={body.thread_id}")

    return StreamingResponse(
        _stream(payload, request_id),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
            "X-Request-Id":      request_id,
        },
    )

@app.post(f"/api/v1/{AGENT}/resume")
async def resume(
    body:    ResumeRequest,
    request: Request,
    _:       str = Depends(verify_api_key),
):
    request_id = str(uuid.uuid4())[:8]
    user_id    = getattr(request.state, "user_id", "anonymous")

    if not rate_limiter.allow(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    payload = {
        "message":     "",
        "thread_id":   body.thread_id,
        "domain":      body.domain,
        "resume":      True,
        "user_answer": body.user_answer,
    }

    return StreamingResponse(
        _stream(payload, request_id),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
            "X-Request-Id":      request_id,
        },
    )

# ── Observability ─────────────────────────────────────────────────────────

def _flatten_decimals(obj):
    if isinstance(obj, Decimal):
        return float(obj) if "." in str(obj) else int(obj)
    if isinstance(obj, list):
        return [_flatten_decimals(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _flatten_decimals(v) for k, v in obj.items()}
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
        html = html.replace(
            "</head>",
            f'<script>window.PLATFORM_API_KEY = "{api_key}";</script></head>'
        )
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
        log.error(f"[PLATFORM] list_traces error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

@app.get(f"/api/v1/{AGENT}/traces/{{thread_id}}")
async def get_trace(thread_id: str, _: str = Depends(verify_api_key)):
    try:
        from boto3.dynamodb.conditions import Attr
        dynamo = boto3.resource("dynamodb", region_name=REGION)
        table  = dynamo.Table(_get_trace_table_name())
        resp   = table.scan(FilterExpression=Attr("thread_id").eq(thread_id))
        return {"thread_id": thread_id, "traces": resp.get("Items", []), "count": len(resp.get("Items", []))}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get(f"/api/v1/{AGENT}/prompt")
async def get_prompt_info(_: str = Depends(verify_api_key)):
    try:
        ssm   = boto3.client("ssm", region_name=REGION)
        g_id  = ssm.get_parameter(Name=f"{SSM_PREFIX}/bedrock/guardrail_id")["Parameter"]["Value"]
        g_ver = ssm.get_parameter(Name=f"{SSM_PREFIX}/bedrock/guardrail_version")["Parameter"]["Value"]
        return {"guardrail_id": g_id, "guardrail_version": g_ver, "ssm_prefix": SSM_PREFIX}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))