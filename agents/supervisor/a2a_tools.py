"""
agents/supervisor/a2a_tools.py
================================
A2A tool wrappers — dynamically built from agent registry in SSM.

AGENT REGISTRY:
  SSM path: {SSM_PREFIX}/agents/registry
  Value: JSON array of agent descriptors:
  [
    {
      "name":        "research",
      "description": "Search clinical trial documents...",
      "port":        8001
    },
    ...
  ]

ADDING A NEW AGENT (no code changes needed):
  1. Deploy new AgentCore Runtime
  2. Update SSM registry:
       aws ssm put-parameter \
         --name /vs-agentcore-multiagent/prod/agents/registry \
         --value '[...existing..., {"name":"new_agent","description":"...","port":8006}]' \
         --type String --overwrite
  3. Supervisor picks up the new agent on next cold start

HOW STREAMING WORKS:
  LLM calls research_agent("query")
    → @tool runs async
    → _invoke_sub_agent() streams from Research Agent
    → puts tokens on token_queue (side channel)
    → Supervisor handler() drains token_queue simultaneously
    → user sees tokens in real time
    → @tool returns full answer to LLM when done

LOCAL_MODE:
  Set LOCAL_MODE=true to call sub-agents via HTTP on localhost instead
  of invoke_agent_runtime(). Ports come from registry descriptor.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import boto3
import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

REGION     = os.environ.get("AWS_REGION", "us-east-1")

# ── Span buffer — keyed by thread_id ─────────────────────────────────────────
# Stores sub-agent spans during a Supervisor request.
# TracerMiddleware reads this in after_agent via _pop_span_buffer().
# Using thread_id (not run_id) because run_id is only known inside middleware.
_span_buffer: dict[str, list] = {}
_span_buffer_lock = __import__("threading").Lock()

def _append_span(thread_id: str, span: dict) -> None:
    with _span_buffer_lock:
        if thread_id not in _span_buffer:
            _span_buffer[thread_id] = []
        _span_buffer[thread_id].append(span)

def pop_span_buffer(thread_id: str) -> list:
    """Called by TracerMiddleware.after_agent to get sub-agent spans."""
    with _span_buffer_lock:
        return _span_buffer.pop(thread_id, [])
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
LOCAL_MODE = os.environ.get("LOCAL_MODE", "false").lower() == "true"

_executor = ThreadPoolExecutor(max_workers=10)


# ── HITL Interrupt ────────────────────────────────────────────────────────────

class HITLInterrupt(Exception):
    def __init__(self, question: str, options: list[str], allow_freetext: bool = True):
        self.question       = question
        self.options        = options
        self.allow_freetext = allow_freetext
        super().__init__(f"HITL: {question}")


# ── Supervisor-native HITL tool ──────────────────────────────────────────────
#
# HITL is NOT a sub-agent. It's a Supervisor-level tool that:
#   1. Supervisor LLM detects vague query
#   2. Supervisor calls research/knowledge agent to get REAL candidates from DB
#   3. Supervisor calls ask_user() with real options → raises HITLInterrupt
#   4. Supervisor app.py catches interrupt → sends {"type":"interrupt"} to UI
#   5. User selects → same thread_id resumes with selection as next message
#
# Why not a sub-agent?
#   - HITL agent would need its own LLM + search tools = extra cost + complexity
#   - Options would come from GPT-4o training knowledge, NOT your actual database
#   - Supervisor already has Postgres checkpointer for pause/resume
#   - HITL is a Supervisor concern — it knows when query is vague

class AskUserInput(BaseModel):
    question: str = Field(description="The clarifying question to ask the user")
    options: list[str] = Field(description="List of specific options for the user to choose from (2-6 items). Each option should be a real trial name or specific category from the database.")
    allow_freetext: bool = Field(default=True, description="Whether to allow free-text response in addition to options")


def build_ask_user_tool() -> Any:
    """
    Build the ask_user HITL tool for the Supervisor.

    When the LLM calls this tool, it raises HITLInterrupt which is caught
    by the Supervisor app.py handler and sent to the UI as an interrupt event.

    IMPORTANT: The LLM should FIRST search for real candidates using
    research_agent or knowledge_agent, THEN call ask_user with real options.
    Never call ask_user with made-up or training-knowledge options.
    """
    async def ask_user_func(question: str, options: list[str], allow_freetext: bool = True) -> str:
        log.info(f"[Supervisor] HITL interrupt: question='{question}'  options={options}")
        raise HITLInterrupt(
            question=question,
            options=options,
            allow_freetext=allow_freetext,
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


# ── Agent Registry ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_agent_registry() -> list[dict]:
    """
    Load agent registry from SSM.

    SSM path: {SSM_PREFIX}/agents/registry
    Written by: ./scripts/deploy.sh registry

    Cached — reloads only on cold start.
    To add a new agent: update SSM registry and restart Supervisor.
    No code changes needed.

    Raises:
        RuntimeError: if SSM registry not found — run deploy.sh registry first.
    """
    ssm = boto3.client("ssm", region_name=REGION)
    try:
        value    = ssm.get_parameter(Name=f"{SSM_PREFIX}/agents/registry")["Parameter"]["Value"]
        registry = json.loads(value)
        log.info(f"[A2A] Registry loaded from SSM ({len(registry)} agents): {[a['name'] for a in registry]}")
        return registry
    except ssm.exceptions.ParameterNotFound:
        raise RuntimeError(
            f"Agent registry not found at {SSM_PREFIX}/agents/registry. "
            f"Run: ./scripts/deploy.sh registry"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load agent registry from SSM: {exc}")


@lru_cache(maxsize=1)
def _get_runtime_arns() -> dict[str, str]:
    """Load sub-agent runtime ARNs from SSM. Cached."""
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


# ── SSM registry writer (call this after deploying a new agent) ───────────────

def register_agent(name: str, description: str, port: int = 0) -> None:
    """
    Add or update an agent in the SSM registry.
    Call after deploying a new AgentCore Runtime.

    Args:
        name:        Agent name (e.g. "my_new_agent")
        description: Tool description shown to the Supervisor LLM
        port:        Local port for LOCAL_MODE testing
    """
    registry = list(_get_agent_registry())
    # Update existing or append
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
    # Clear cache so next call picks up new registry
    _get_agent_registry.cache_clear()
    _get_runtime_arns.cache_clear()
    log.info(f"[A2A] Registry updated: {[a['name'] for a in registry]}")


# ── SSE stream parser ─────────────────────────────────────────────────────────

async def _parse_sse_stream(
    agent_name:  str,
    stream_iter,
    token_queue: asyncio.Queue,
) -> tuple[str, dict]:
    """
    Parse SSE events from a sub-agent stream.

    Returns:
        (full_answer, span_data) where span_data contains observability
        metadata emitted by the sub-agent as a {"type": "span"} event.
        Supervisor collects these spans and writes ONE unified trace record.
    """
    full_answer = ""
    span_data   = {}

    async for line in stream_iter:
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
            # Sub-agent emits observability metadata for distributed tracing.
            # Supervisor collects these and writes ONE unified DynamoDB record.
            span_data = event.get("data", {})
            span_data["agent"] = agent_name
            log.debug(f"[A2A] Span captured from {agent_name}: {list(span_data.keys())}")

        elif etype == "done":
            done_answer = event.get("answer", "")
            if done_answer and len(done_answer) > len(full_answer):
                full_answer = done_answer

    return full_answer, span_data


# ── Core invoke ───────────────────────────────────────────────────────────────

async def _invoke_sub_agent_local(
    agent_name:  str,
    payload:     dict,
    token_queue: asyncio.Queue,
    port:        int,
) -> str:
    """LOCAL_MODE: call sub-agent via HTTP on localhost."""
    url = f"http://localhost:{port}/invocations"
    log.info(f"[A2A LOCAL] {agent_name} → {url}")

    async def _line_iter(response):
        async for line in response.aiter_lines():
            yield line

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url,
                                 json=payload,
                                 headers={"Content-Type": "application/json"}) as response:
            full_answer, span_data = await _parse_sse_stream(agent_name, _line_iter(response), token_queue)

    log.info(f"[A2A LOCAL] {agent_name} done  answer_len={len(full_answer)}")
    return full_answer, span_data


async def _invoke_sub_agent(
    agent_name:  str,
    payload:     dict,
    token_queue: asyncio.Queue,
    port:        int = 0,
) -> str:
    """
    Invoke a sub-agent.
    LOCAL_MODE=true → HTTP localhost
    LOCAL_MODE=false → invoke_agent_runtime()
    """
    if LOCAL_MODE:
        return await _invoke_sub_agent_local(agent_name, payload, token_queue, port)
    # prod path also returns (answer, span_data)

    arns = _get_runtime_arns()
    if agent_name not in arns:
        raise RuntimeError(f"No runtime ARN for '{agent_name}'. Available: {list(arns.keys())}")

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
                for chunk in streaming_body.iter_chunks(chunk_size=1024):
                    if chunk:
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
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
    full_answer = ""

    while True:
        chunk = await chunk_queue.get()
        if chunk is None:
            break
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        for line in text.splitlines():
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
            elif etype == "done":
                done_answer = event.get("answer", "")
                if done_answer and len(done_answer) > len(full_answer):
                    full_answer = done_answer

    log.info(f"[A2A] {agent_name} done  answer_len={len(full_answer)}")
    return full_answer, {}  # span_data not available in prod path yet


# ── Dynamic tool factory ──────────────────────────────────────────────────────

class AgentInput(BaseModel):
    query: str = Field(description="The query or task to send to this agent")


def build_a2a_tools(
    session_id:  str,
    domain:      str,
    token_queue: asyncio.Queue,
) -> list[Any]:
    """
    Dynamically build A2A tools from the agent registry.

    Reads registry from SSM (or DEFAULT_REGISTRY fallback).
    Each registry entry becomes a LangChain StructuredTool.
    Adding a new agent = update SSM registry, no code changes.

    Args:
        session_id:  User thread_id passed to every sub-agent.
        domain:      "pharma" passed to sub-agents.
        token_queue: Sub-agent events put here for Supervisor to re-stream.

    Returns:
        List of StructuredTool objects, one per registered agent.
    """
    registry = _get_agent_registry()
    tools    = []

    for agent_def in registry:
        name        = agent_def["name"]
        description = agent_def["description"]
        port        = agent_def.get("port", 0)

        # Capture loop variables in closure
        def make_tool_func(agent_name: str, agent_port: int):
            async def tool_func(query: str) -> str:
                log.info(f"[Supervisor] → {agent_name}  query={query[:60]}...")
                # Use agent-scoped session_id to prevent checkpoint cross-contamination.
                agent_session_id = f"{session_id}__{agent_name}"
                # Safety agent: use real token_queue so we can detect BLOCKED
                # but safety TOKENS (PASSED/BLOCKED text) are suppressed via throwaway
                safety_token_queue = asyncio.Queue() if agent_name == "safety" else None
                effective_queue    = safety_token_queue if agent_name == "safety" else token_queue

                t0 = asyncio.get_event_loop().time()
                answer, span_data = await _invoke_sub_agent(
                    agent_name  = agent_name,
                    payload     = {"message": query, "session_id": agent_session_id, "domain": domain},
                    token_queue = effective_queue,
                    port        = agent_port,
                )
                elapsed_ms = round((asyncio.get_event_loop().time() - t0) * 1000, 2)

                # Notify Supervisor app of safety verdict via real token_queue
                if agent_name == "safety":
                    verdict = answer.strip()
                    verdict_upper = verdict.upper()
                    if verdict_upper.startswith("BLOCKED"):
                        await token_queue.put({
                            "type":   "safety_blocked",
                            "reason": verdict,
                        })
                        log.info(f"[A2A] Safety BLOCKED  reason={verdict[:60]}")
                    elif verdict_upper.startswith("PASSED"):
                        await token_queue.put({"type": "safety_passed"})
                        log.info(f"[A2A] Safety PASSED")
                    else:
                        # Unknown verdict — treat as passed to avoid false blocks
                        await token_queue.put({"type": "safety_passed"})
                        log.info(f"[A2A] Safety unknown verdict — treating as PASSED: {verdict[:40]}")

                # Buffer span for TracerMiddleware to pick up in after_agent.
                # Uses thread_id as key since run_id is not accessible here.
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

    # Add Supervisor-native HITL tool (not from registry — it's always present)
    tools.append(build_ask_user_tool())
    log.info(f"[A2A] Built {len(tools)} tools from registry + ask_user: {[t.name for t in tools]}")
    return tools