"""
platform/main.py — VS AgentCore Multi-Agent Platform (Local Mode)
==================================================================
For local testing: talks to Supervisor on localhost:8000 via HTTP
instead of invoke_agent_runtime().

Set LOCAL_MODE=true to use this mode.

Run:
  cd platform
  LOCAL_MODE=true PLATFORM_API_KEY=vs-test uvicorn main:app --port 8080 --reload
"""

import asyncio
import json
import logging
import os
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from gateway.auth import verify_api_key, _get_api_key
from gateway.rate_limiter import RateLimiter
from gateway.schemas import ChatRequest, ResumeRequest
from gateway.logging_mw import LoggingMiddleware

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title       = "VS AgentCore Multi-Agent Platform",
    description = "Clinical Trial Research Assistant — Local Gateway",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["POST", "GET", "OPTIONS"],
    allow_headers  = ["*"],
)
app.add_middleware(LoggingMiddleware)

# Mount static files (UI)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

REGION       = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX   = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
LOCAL_MODE   = os.environ.get("LOCAL_MODE", "false").lower() == "true"
SUPERVISOR_URL = os.environ.get("SUPERVISOR_URL", "http://localhost:8000")
AGENT        = "clinical-trial"
rate_limiter = RateLimiter(requests_per_minute=60)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_local(payload: dict, request_id: str):
    """Stream from local Supervisor agent via HTTP."""
    t0  = time.perf_counter()
    url = f"{SUPERVISOR_URL}/invocations"

    log.info(f"[PLATFORM] → Supervisor  request_id={request_id}  thread={payload.get('thread_id')}")

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
    """Serve the main UI."""
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
    """Send a new message — returns SSE stream."""
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
        _stream_local(payload, request_id),
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
    """Resume after HITL — returns SSE stream."""
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
        _stream_local(payload, request_id),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
            "X-Request-Id":      request_id,
        },
    )


@app.get("/observability", response_class=HTMLResponse)
async def observability_dashboard():
    """Serve the AgentCore Observability Dashboard."""
    html_path = os.path.join(static_dir, "traces_dashboard.html")
    try:
        with open(html_path) as f:
            html = f.read()
        api_key = _get_api_key()
        html    = html.replace(
            "</head>",
            f'<script>window.PLATFORM_API_KEY = "{api_key}";</script></head>'
        )
        return HTMLResponse(content=html)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dashboard HTML not found")


@app.get(f"/api/v1/{AGENT}/traces")
async def list_traces(
    limit: int = 200,
    agent: str = "supervisor-agent",
    _:     str = Depends(verify_api_key),
):
    """Return traces from DynamoDB."""
    try:
        import boto3
        from decimal import Decimal
        from boto3.dynamodb.conditions import Attr

        def fix(obj):
            if isinstance(obj, Decimal): return float(obj)
            if isinstance(obj, list):    return [fix(i) for i in obj]
            if isinstance(obj, dict):    return {k: fix(v) for k, v in obj.items()}
            return obj

        ddb       = boto3.resource("dynamodb", region_name=REGION)
        ssm       = boto3.client("ssm", region_name=REGION)
        tbl_name  = ssm.get_parameter(
            Name=f"{SSM_PREFIX}/dynamodb/trace_table_name"
        )["Parameter"]["Value"]
        table     = ddb.Table(tbl_name)

        scan_kwargs = {}
        if agent:
            scan_kwargs["FilterExpression"] = Attr("agent_name").eq(agent)

        items, resp = [], table.scan(**scan_kwargs)
        items.extend([fix(i) for i in resp.get("Items", [])])
        while "LastEvaluatedKey" in resp:
            resp = table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"], **scan_kwargs)
            items.extend([fix(i) for i in resp.get("Items", [])])

        items.sort(key=lambda x: float(x.get("ts", 0)), reverse=True)
        return items[:limit]

    except Exception as exc:
        log.error(f"[PLATFORM] list_traces error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)