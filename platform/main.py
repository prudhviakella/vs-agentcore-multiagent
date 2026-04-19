"""
platform/main.py — VS AgentCore Multi-Agent Platform (FastAPI)
===============================================================
COPY of vs-agentcore-platform-aws/platform/main.py with TWO changes:

  CHANGE 1 — SSM_PREFIX:
    /vs-agentcore/prod  →  /vs-agentcore-multiagent/prod

  CHANGE 2 — Runtime ARN SSM path (Supervisor instead of single agent):
    /vs-agentcore/prod/agent_runtime_arn
    →  /vs-agentcore-multiagent/prod/agents/supervisor/runtime_arn

Everything else is identical — same SSE streaming, same HITL resume,
same observability endpoints, same rate limiter, same auth.

The platform calls the SUPERVISOR AgentCore Runtime.
The Supervisor then routes to sub-agents via A2A.
From the platform's perspective this is transparent — same invoke_agent_runtime() call.

SSE FLOW:
  UI → POST /api/v1/clinical-trial/chat
     ← StreamingResponse (text/event-stream)
         token, tool_start, tool_end, interrupt, chart, done, error
  (chart is a new event type from Chart Agent — UI renders Chart.js inline)
"""

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from functools import lru_cache

import boto3
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse

from gateway.auth import verify_api_key, _get_api_key
from gateway.rate_limiter import RateLimiter
from gateway.schemas import ChatRequest, ResumeRequest
from gateway.logging_mw import LoggingMiddleware

_executor = ThreadPoolExecutor(max_workers=20)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title="VS AgentCore Multi-Agent Platform",
    description="Clinical Trial Research Assistant — Multi-Agent Gateway",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

# ── Config ────────────────────────────────────────────────────────────────
REGION       = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX   = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")  # ← CHANGE 1
AGENT        = "clinical-trial"
rate_limiter = RateLimiter(requests_per_minute=30)


@lru_cache(maxsize=1)
def _get_agent_runtime_arn() -> str:
    """
    Fetch Supervisor AgentCore Runtime ARN from SSM.
    CHANGE 2: reads Supervisor ARN, not single-agent ARN.
    """
    ssm = boto3.client("ssm", region_name=REGION)
    return ssm.get_parameter(
        Name=f"{SSM_PREFIX}/agents/supervisor/runtime_arn"  # ← CHANGE 2
    )["Parameter"]["Value"]


# ── SSE streaming ─────────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_from_agentcore(payload: dict, request_id: str):
    """
    Invoke Supervisor AgentCore Runtime and forward SSE events to UI.
    Identical to single agent — the multi-agent routing is transparent.
    """
    t0    = time.perf_counter()
    queue = asyncio.Queue()
    loop  = asyncio.get_event_loop()

    def _stream_to_queue():
        try:
            agent_arn = _get_agent_runtime_arn()
            client    = boto3.client("bedrock-agentcore", region_name=REGION)

            log.info(
                f"[PLATFORM] Invoking Supervisor"
                f"  request_id={request_id}"
                f"  thread_id={payload.get('thread_id')}"
                f"  resume={payload.get('resume', False)}"
            )

            response = client.invoke_agent_runtime(
                agentRuntimeArn  = agent_arn,
                runtimeSessionId = payload["thread_id"],
                payload          = json.dumps(payload).encode("utf-8"),
            )

            streaming_body = response.get("response")
            if streaming_body:
                for chunk in streaming_body.iter_chunks(chunk_size=1024):
                    if chunk:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
            else:
                raw = response.get("body", b"")
                if raw:
                    loop.call_soon_threadsafe(queue.put_nowait, raw)

        except Exception as exc:
            log.error(f"[PLATFORM] Stream error: {exc}")
            err_bytes = _sse({"type": "error", "message": str(exc)}).encode()
            loop.call_soon_threadsafe(queue.put_nowait, err_bytes)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(_executor, _stream_to_queue)

    try:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
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


# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "vs-agentcore-multiagent-platform", "version": "2.0.0"}


@app.post(f"/api/v1/{AGENT}/chat")
async def chat(body: ChatRequest, request: Request, _: str = Depends(verify_api_key)):
    request_id = str(uuid.uuid4())[:8]
    user_id    = getattr(request.state, "user_id", "anonymous")
    if not rate_limiter.allow(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    payload = {
        "message":   body.message,
        "thread_id": body.thread_id,
        "domain":    body.domain,
        "resume":    False,
    }
    return StreamingResponse(
        _stream_from_agentcore(payload, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
            "X-Request-Id":      request_id,
        },
    )


@app.post(f"/api/v1/{AGENT}/resume")
async def resume(body: ResumeRequest, request: Request, _: str = Depends(verify_api_key)):
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
        _stream_from_agentcore(payload, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
            "X-Request-Id":      request_id,
        },
    )


# ── Observability ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_trace_table_name() -> str:
    return boto3.client("ssm", region_name=REGION).get_parameter(
        Name=f"{SSM_PREFIX}/dynamodb/trace_table_name"
    )["Parameter"]["Value"]


def _flatten_decimals(obj):
    if isinstance(obj, Decimal):
        return float(obj) if "." in str(obj) else int(obj)
    if isinstance(obj, list):
        return [_flatten_decimals(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _flatten_decimals(v) for k, v in obj.items()}
    return obj


@app.get("/observability", response_class=HTMLResponse)
async def observability_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "static", "traces_dashboard.html")
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
        raise HTTPException(status_code=404, detail="Dashboard not found.")


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
    """Returns current prompt IDs for all 6 agents from SSM."""
    try:
        ssm    = boto3.client("ssm", region_name=REGION)
        agents = ["supervisor", "research", "knowledge", "hitl", "safety", "chart"]
        result = {}
        for agent in agents:
            try:
                p_id  = ssm.get_parameter(Name=f"/{agent}-agent/prod/bedrock/prompt_id")["Parameter"]["Value"]
                p_ver = ssm.get_parameter(Name=f"/{agent}-agent/prod/bedrock/prompt_version")["Parameter"]["Value"]
                result[agent] = {"prompt_id": p_id, "prompt_version": p_ver}
            except Exception:
                result[agent] = {"prompt_id": "not set", "prompt_version": "not set"}
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get(f"/api/v1/{AGENT}/agents")
async def list_agents(_: str = Depends(verify_api_key)):
    """Returns all sub-agent runtime ARNs from SSM — confirms deployment health."""
    try:
        ssm    = boto3.client("ssm", region_name=REGION)
        agents = ["supervisor", "research", "knowledge", "hitl", "safety", "chart"]
        result = {}
        for agent in agents:
            try:
                arn = ssm.get_parameter(
                    Name=f"{SSM_PREFIX}/agents/{agent}/runtime_arn"
                )["Parameter"]["Value"]
                result[agent] = arn
            except Exception:
                result[agent] = "not deployed"
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))