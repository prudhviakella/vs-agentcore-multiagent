"""
a2a_tools/invoke.py
====================
Async sub-agent invocation via AWS Bedrock AgentCore using httpx.
Adds a module-level span buffer so TracerMiddleware can read
observability metadata (rag_metrics) after each sub-agent call.
"""
import asyncio
import json
import logging
import os
import threading
from urllib.parse import quote

import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import get_session

from agents.supervisor.a2a_tools.parser   import parse_sse_stream
from agents.supervisor.a2a_tools.registry import get_runtime_arns

log    = logging.getLogger(__name__)
REGION = os.environ.get("AWS_REGION", "us-east-1")
_BASE_URL = f"https://bedrock-agentcore.{REGION}.amazonaws.com"

# ── Span buffer ────────────────────────────────────────────────────────────
# Keyed by BASE session_id (thread_id, without __agent suffix).
# TracerMiddleware reads via pop_span_buffer() in after_agent.
_span_buffer:      dict[str, list] = {}
_span_buffer_lock: threading.Lock  = threading.Lock()


def _append_span(session_id: str, span: dict) -> None:
    with _span_buffer_lock:
        _span_buffer.setdefault(session_id, []).append(span)


def pop_span_buffer(session_id: str) -> list:
    with _span_buffer_lock:
        return _span_buffer.pop(session_id, [])


# ── SigV4 signing ──────────────────────────────────────────────────────────

def _build_signed_headers(url: str, headers: dict, body: bytes) -> dict:
    credentials = get_session().get_credentials().get_frozen_credentials()
    aws_request = AWSRequest(method="POST", url=url, data=body, headers=headers)
    SigV4Auth(credentials, "bedrock-agentcore", REGION).add_auth(aws_request)
    return dict(aws_request.headers)


# ── Invocation ─────────────────────────────────────────────────────────────

async def invoke_sub_agent(
    agent_name:  str,
    payload:     dict,
    token_queue: asyncio.Queue,
) -> str:
    """
    Invoke a sub-agent and stream its response.
    Span data (including rag_metrics from research agent) is buffered
    in _span_buffer keyed by the BASE session_id so TracerMiddleware
    can retrieve it in after_agent via pop_span_buffer().
    """
    arns = get_runtime_arns()
    if agent_name not in arns:
        raise RuntimeError(
            f"No runtime ARN for '{agent_name}'. "
            f"Available: {list(arns.keys())}."
        )

    runtime_arn = arns[agent_name]
    # agent_session_id is namespaced: "thread_id__agent_name"
    # Base session_id is just "thread_id" — what the tracer uses as run_id
    agent_session_id = payload.get("session_id", f"{agent_name}-session")
    base_session_id  = agent_session_id.split("__")[0] if "__" in agent_session_id else agent_session_id

    body = json.dumps(payload).encode("utf-8")
    url  = f"{_BASE_URL}/runtimes/{quote(runtime_arn, safe='')}/invocations"
    headers = {
        "Content-Type": "application/json",
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": agent_session_id,
    }
    signed_headers = _build_signed_headers(url, headers, body)

    log.info(f"[A2A] Invoking {agent_name}  session={agent_session_id[-8:]}")

    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST", url,
            headers = signed_headers,
            content = body,
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="replace")[:300]
                if response.status_code == 403:
                    raise RuntimeError(
                        f"AgentCore 403 Forbidden for '{agent_name}'. "
                        f"Check IAM role has bedrock-agentcore:InvokeAgentRuntime permission. "
                        f"Detail: {error_text}"
                    )
                raise RuntimeError(
                    f"AgentCore {response.status_code} for '{agent_name}': {error_text}"
                )

            answer, span_data = await parse_sse_stream(
                agent_name,
                response.aiter_lines(),
                token_queue,
            )

    log.info(f"[A2A] {agent_name} complete  answer_len={len(answer)}  span_keys={list(span_data.keys())}")

    # Buffer span so TracerMiddleware can read it in after_agent
    span_data.setdefault("agent",  agent_name)
    span_data.setdefault("status", "ok")
    _append_span(base_session_id, span_data)
    log.info(f"[A2A] Span buffered  agent={agent_name}  base_session={base_session_id[:8]}")

    return answer