"""
agents/supervisor/a2a_tools.py
================================
A2A tool wrappers — dynamically built from agent registry in SSM.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, AsyncIterator

import boto3
import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

REGION     = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
LOCAL_MODE = os.environ.get("LOCAL_MODE", "false").lower() == "true"

_executor = ThreadPoolExecutor(max_workers=10)

_span_buffer: dict[str, list] = {}
_span_buffer_lock = __import__("threading").Lock()


def _append_span(thread_id: str, span: dict) -> None:
    with _span_buffer_lock:
        if thread_id not in _span_buffer:
            _span_buffer[thread_id] = []
        _span_buffer[thread_id].append(span)


def pop_span_buffer(thread_id: str) -> list:
    with _span_buffer_lock:
        return _span_buffer.pop(thread_id, [])


class HITLInterrupt(Exception):
    def __init__(self, question: str, options: list[str], allow_freetext: bool = True):
        self.question       = question
        self.options        = options
        self.allow_freetext = allow_freetext
        super().__init__(f"HITL: {question}")


class AskUserInput(BaseModel):
    question:      str       = Field(description="The clarifying question to ask the user")
    options:       list[str] = Field(description="List of specific options for the user to choose from (2-6 items).")
    allow_freetext: bool     = Field(default=True, description="Whether to allow free-text response in addition to options")


def build_ask_user_tool() -> Any:
    async def ask_user_func(question: str, options: list[str], allow_freetext: bool = True) -> str:
        log.info(f"[Supervisor] HITL interrupt: question='{question}'  options={options}")
        raise HITLInterrupt(
            question       = question,
            options        = options,
            allow_freetext = allow_freetext,
        )

    return StructuredTool.from_function(
        coroutine   = ask_user_func,
        name        = "ask_user",
        description = (
            "Ask the user a clarifying question when the query is too vague. "
            "IMPORTANT: First search for real candidates using research_agent or knowledge_agent. "
            "Then call this tool with REAL options from the database — never made-up names. "
            "This PAUSES execution until the user responds."
        ),
        args_schema = AskUserInput,
    )


@lru_cache(maxsize=1)
def _get_agent_registry() -> list[dict]:
    ssm = boto3.client("ssm", region_name=REGION)
    try:
        value    = ssm.get_parameter(Name=f"{SSM_PREFIX}/agents/registry")["Parameter"]["Value"]
        registry = json.loads(value)
        log.info(f"[A2A] Registry loaded from SSM ({len(registry)} agents): {[a['name'] for a in registry]}")
        return registry
    except ssm.exceptions.ParameterNotFound:
        raise RuntimeError(f"Agent registry not found at {SSM_PREFIX}/agents/registry.")
    except Exception as exc:
        raise RuntimeError(f"Failed to load agent registry from SSM: {exc}")


@lru_cache(maxsize=1)
def _get_runtime_arns() -> dict[str, str]:
    ssm      = boto3.client("ssm", region_name=REGION)
    registry = _get_agent_registry()
    arns     = {}
    for agent in registry:
        name = agent["name"]
        try:
            arn = ssm.get_parameter(
                Name=f"{SSM_PREFIX}/agents/{name}/runtime_arn"
            )["Parameter"]["Value"]
            arns[name] = arn
            log.info(f"[A2A] ARN loaded for {name}")
        except Exception as exc:
            log.warning(f"[A2A] No ARN for {name}: {exc}")
    return arns


def preflight_check_arns() -> None:
    registry = _get_agent_registry()
    arns     = _get_runtime_arns()
    missing  = [a["name"] for a in registry if a["name"] not in arns]

    if missing:
        raise RuntimeError(
            f"[A2A] Pre-flight failed — missing runtime ARNs for: {missing}."
        )

    log.info(f"[A2A] Pre-flight passed — all {len(arns)} ARNs present: {list(arns.keys())}")


async def _parse_sse_stream(
    agent_name:  str,
    line_iter:   AsyncIterator[str],
    token_queue: asyncio.Queue,
) -> tuple[str, dict]:
    full_answer = ""
    span_data   = {}

    async for line in line_iter:
        line = line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            line = line[6:]
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "token":
            c = event.get("content", "")
            if c:
                full_answer += c
                await token_queue.put({"type": "token", "content": c})

        elif etype in ("tool_start", "tool_end"):
            await token_queue.put({"type": etype, "name": event.get("name", "")})

        elif etype == "chart":
            await token_queue.put({
                "type":       "chart",
                "config":     event.get("config", {}),
                "chart_type": event.get("chart_type", "bar"),
            })

        elif etype == "interrupt":
            raise HITLInterrupt(
                question       = event.get("question", "Please clarify:"),
                options        = event.get("options", []),
                allow_freetext = event.get("allow_freetext", True),
            )

        elif etype == "error":
            raise RuntimeError(f"Sub-agent '{agent_name}': {event.get('message', '')}")

        elif etype == "span":
            span_data = event.get("data", {})
            span_data["agent"] = agent_name
            log.debug(f"[A2A] Span captured from {agent_name}: {list(span_data.keys())}")

        elif etype == "done":
            done_answer = event.get("answer", "")
            if done_answer and len(done_answer) > len(full_answer):
                full_answer = done_answer

    return full_answer, span_data


async def _sse_lines_from_httpx(response) -> AsyncIterator[str]:
    async for line in response.aiter_lines():
        yield line


async def _sse_lines_from_queue(queue: asyncio.Queue) -> AsyncIterator[str]:
    while True:
        chunk = await queue.get()
        if chunk is None:
            return
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        for line in text.splitlines():
            yield line


async def _invoke_sub_agent_local(
    agent_name:  str,
    payload:     dict,
    token_queue: asyncio.Queue,
    port:        int,
) -> tuple[str, dict]:
    url = f"http://localhost:{port}/invocations"
    log.info(f"[A2A LOCAL] {agent_name} → {url}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url,
                                 json=payload,
                                 headers={"Content-Type": "application/json"}) as response:
            full_answer, span_data = await _parse_sse_stream(
                agent_name, _sse_lines_from_httpx(response), token_queue
            )

    log.info(f"[A2A LOCAL] {agent_name} done  answer_len={len(full_answer)}")
    return full_answer, span_data


async def _invoke_sub_agent_prod(
    agent_name:  str,
    payload:     dict,
    token_queue: asyncio.Queue,
) -> tuple[str, dict]:
    arns = _get_runtime_arns()
    if agent_name not in arns:
        raise RuntimeError(
            f"No runtime ARN for '{agent_name}'. Available: {list(arns.keys())}."
        )

    runtime_arn = arns[agent_name]
    session_id  = payload.get("session_id", f"{agent_name}-session")
    loop        = asyncio.get_event_loop()
    chunk_queue = asyncio.Queue()

    def _stream_in_thread():
        try:
            client   = boto3.client("bedrock-agentcore", region_name=REGION)
            response = client.invoke_agent_runtime(
                agentRuntimeArn  = runtime_arn,
                runtimeSessionId = session_id,
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
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, complete)
                if _partial:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, _partial)
            else:
                raw = response.get("body", b"")
                if raw:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, raw)
        except Exception as exc:
            err = json.dumps({"type": "error", "message": str(exc)}).encode()
            loop.call_soon_threadsafe(chunk_queue.put_nowait, err)
        finally:
            loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

    loop.run_in_executor(_executor, _stream_in_thread)

    full_answer, span_data = await _parse_sse_stream(
        agent_name, _sse_lines_from_queue(chunk_queue), token_queue
    )
    log.info(f"[A2A PROD] {agent_name} done  answer_len={len(full_answer)}  span_keys={list(span_data.keys())}")
    return full_answer, span_data


async def _invoke_sub_agent(
    agent_name:  str,
    payload:     dict,
    token_queue: asyncio.Queue,
    port:        int = 0,
) -> tuple[str, dict]:
    if LOCAL_MODE:
        return await _invoke_sub_agent_local(agent_name, payload, token_queue, port)
    return await _invoke_sub_agent_prod(agent_name, payload, token_queue)


class AgentInput(BaseModel):
    query: str = Field(description="The query or task to send to this agent")


def build_a2a_tools(
    session_id:  str,
    domain:      str,
    token_queue: asyncio.Queue,
) -> list[Any]:
    if not LOCAL_MODE:
        preflight_check_arns()

    registry = _get_agent_registry()
    tools    = []

    for agent_def in registry:
        name        = agent_def["name"]
        description = agent_def["description"]
        port        = agent_def.get("port", 0)

        def make_tool_func(agent_name: str, agent_port: int):
            async def tool_func(query: str) -> str:
                log.info(f"[Supervisor] → {agent_name}  query={query[:60]}...")

                agent_session_id = f"{session_id}__{agent_name}"
                effective_queue  = asyncio.Queue() if agent_name == "safety" else token_queue

                t0 = asyncio.get_event_loop().time()
                answer, span_data = await _invoke_sub_agent(
                    agent_name  = agent_name,
                    payload     = {"message": query, "session_id": agent_session_id, "domain": domain},
                    token_queue = effective_queue,
                    port        = agent_port,
                )
                elapsed_ms = round((asyncio.get_event_loop().time() - t0) * 1000, 2)

                if agent_name == "safety":
                    verdict       = answer.strip()
                    verdict_upper = verdict.upper()
                    if verdict_upper.startswith("BLOCKED"):
                        await token_queue.put({"type": "safety_blocked", "reason": verdict})
                        log.info(f"[A2A] Safety BLOCKED  reason={verdict[:60]}")
                    elif verdict_upper.startswith("PASSED"):
                        await token_queue.put({"type": "safety_passed"})
                        log.info("[A2A] Safety PASSED")
                    else:
                        await token_queue.put({"type": "safety_passed"})
                        log.info(f"[A2A] Safety unknown verdict — treating as PASSED: {verdict[:40]}")

                span_data.setdefault("agent",      agent_name)
                span_data.setdefault("elapsed_ms", elapsed_ms)
                span_data.setdefault("status",     "ok")
                _append_span(session_id, span_data)
                log.info(f"[A2A] Span buffered for {agent_name}  elapsed={elapsed_ms:.0f}ms")

                return answer

            tool_func.__name__ = f"{agent_name}_agent"
            return tool_func

        tool = StructuredTool.from_function(
            coroutine    = make_tool_func(name, port),
            name         = f"{name}_agent",
            description  = description,
            args_schema  = AgentInput,
        )
        tools.append(tool)
        log.info(f"[A2A] Registered tool: {name}_agent")

    tools.append(build_ask_user_tool())
    log.info(f"[A2A] Built {len(tools)} tools: {[t.name for t in tools]}")
    return tools


def register_agent(name: str, description: str, port: int = 0) -> None:
    registry = list(_get_agent_registry())
    for i, entry in enumerate(registry):
        if entry["name"] == name:
            registry[i] = {"name": name, "description": description, "port": port}
            break
    else:
        registry.append({"name": name, "description": description, "port": port})

    ssm = boto3.client("ssm", region_name=REGION)
    ssm.put_parameter(
        Name=f"{SSM_PREFIX}/agents/registry",
        Value=json.dumps(registry),
        Type="String",
        Overwrite=True,
    )
    _get_agent_registry.cache_clear()
    _get_runtime_arns.cache_clear()
    log.info(f"[A2A] Registry updated: {[a['name'] for a in registry]}")