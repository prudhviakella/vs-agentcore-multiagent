"""
agents/supervisor/a2a_tools.py
================================
A2A tool wrappers — dynamically built from agent registry in SSM.

FIXES in this file:

  FIX (Issue #4 — prod path didn't capture span_data):
    BEFORE: _invoke_sub_agent() prod path parsed SSE events inline with a
            separate loop that discarded the "span" event type entirely:
              return full_answer, {}  # span_data not available in prod path yet

            This meant TracerMiddleware._append_span() never fired in production,
            so the Supervisor's DynamoDB trace had empty agent_spans=[].
            Distributed tracing was silently broken in prod (only worked in LOCAL_MODE).

    AFTER:  The prod path now calls _parse_sse_stream() — the same function
            used by the local path. This captures "span" events and returns
            (full_answer, span_data) in both paths identically.

            _parse_sse_stream() is no longer exclusive to LOCAL_MODE. The
            only difference between the two paths is how they obtain the
            line iterator:
              LOCAL  → httpx async stream → _line_iter()
              PROD   → boto3 chunk queue  → _chunk_queue_to_lines()

  FIX (Issue #6 — no pre-flight check for runtime ARNs):
    BEFORE: Missing ARNs only surfaced as RuntimeError mid-request when a
            sub-agent was actually called for the first time. The user saw
            an error response, not a startup failure.

    AFTER:  preflight_check_arns() is called from build_a2a_tools() at cold
            start. It validates that every agent in the registry has a
            corresponding ARN entry in SSM. Missing ARNs raise RuntimeError
            at startup — fail fast, not mid-request.

            LOCAL_MODE skips the check (ports used instead of ARNs).

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
       aws ssm put-parameter \\
         --name /vs-agentcore-multiagent/prod/agents/registry \\
         --value '[...existing..., {"name":"new_agent","description":"...","port":8006}]' \\
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


# ── Span buffer — keyed by thread_id ─────────────────────────────────────────
# Stores sub-agent spans during a Supervisor request.
# TracerMiddleware reads this in after_agent via pop_span_buffer().
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


# ── HITL Interrupt ────────────────────────────────────────────────────────────

class HITLInterrupt(Exception):
    def __init__(self, question: str, options: list[str], allow_freetext: bool = True):
        self.question       = question
        self.options        = options
        self.allow_freetext = allow_freetext
        super().__init__(f"HITL: {question}")


# ── Supervisor-native HITL tool ──────────────────────────────────────────────

class AskUserInput(BaseModel):
    question:      str       = Field(description="The clarifying question to ask the user")
    options:       list[str] = Field(description="List of specific options for the user to choose from (2-6 items). Each option should be a real trial name or specific category from the database.")
    allow_freetext: bool     = Field(default=True, description="Whether to allow free-text response in addition to options")


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


# ── Agent Registry ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_agent_registry() -> list[dict]:
    """
    Load agent registry from SSM.
    Cached — reloads only on cold start.
    To add a new agent: update SSM registry and restart Supervisor.
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


def preflight_check_arns() -> None:
    """
    FIX (Issue #6): Validate that every registered agent has a runtime ARN.
    Called at cold start from build_a2a_tools() when LOCAL_MODE=false.

    WHY at cold start (not per-request):
      Missing ARNs should be surfaced as a deployment error, not a user error.
      Failing fast at startup means the container won't start in a broken state,
      which is immediately visible in AgentCore health checks and CloudWatch.
      Without this, the first user to trigger a missing agent gets an error
      response — confusing and hard to distinguish from an agent bug.

    Raises:
      RuntimeError: listing all agents that are missing ARNs. One error
                    message covers all missing agents so you don't have to
                    fix them one at a time.
    """
    registry = _get_agent_registry()
    arns     = _get_runtime_arns()
    missing  = [a["name"] for a in registry if a["name"] not in arns]

    if missing:
        raise RuntimeError(
            f"[A2A] Pre-flight failed — missing runtime ARNs for: {missing}. "
            f"Deploy these agents first, then update SSM:\n"
            + "\n".join(
                f"  aws ssm put-parameter "
                f"--name {SSM_PREFIX}/agents/{name}/runtime_arn "
                f"--value <arn> --type String --overwrite"
                for name in missing
            )
        )

    log.info(f"[A2A] Pre-flight passed — all {len(arns)} ARNs present: {list(arns.keys())}")


# ── SSE stream parser ─────────────────────────────────────────────────────────

async def _parse_sse_stream(
    agent_name:  str,
    line_iter:   AsyncIterator[str],
    token_queue: asyncio.Queue,
) -> tuple[str, dict]:
    """
    FIX (Issue #4): Shared SSE parser used by BOTH local and prod paths.

    BEFORE: prod path had its own inline SSE parsing loop that silently
            dropped "span" events, returning {} for span_data always.

    AFTER:  Both paths call this function. The only difference is the
            line_iter source:
              LOCAL → _sse_lines_from_httpx()
              PROD  → _sse_lines_from_queue()

    Returns:
        (full_answer, span_data) where span_data contains observability
        metadata emitted by the sub-agent as a {"type": "span"} event.
    """
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
            # FIX: span events now captured in BOTH paths.
            # Sub-agent emits observability metadata. Supervisor collects
            # these and writes ONE unified DynamoDB trace record.
            span_data = event.get("data", {})
            span_data["agent"] = agent_name
            log.debug(f"[A2A] Span captured from {agent_name}: {list(span_data.keys())}")

        elif etype == "done":
            done_answer = event.get("answer", "")
            if done_answer and len(done_answer) > len(full_answer):
                full_answer = done_answer

    return full_answer, span_data


async def _sse_lines_from_httpx(response) -> AsyncIterator[str]:
    """Yield SSE lines from an httpx streaming response (LOCAL_MODE)."""
    async for line in response.aiter_lines():
        yield line


async def _sse_lines_from_queue(queue: asyncio.Queue) -> AsyncIterator[str]:
    """
    Yield SSE lines from the boto3 chunk queue (PROD path).

    FIX: replaces the inline loop in the original prod path. By producing
    a line iterator, the prod path can now call _parse_sse_stream() with
    the same interface as the local path.
    """
    while True:
        chunk = await queue.get()
        if chunk is None:
            return  # sentinel — stream complete
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        for line in text.splitlines():
            yield line


# ── Core invoke ───────────────────────────────────────────────────────────────

async def _invoke_sub_agent_local(
    agent_name:  str,
    payload:     dict,
    token_queue: asyncio.Queue,
    port:        int,
) -> tuple[str, dict]:
    """LOCAL_MODE: call sub-agent via HTTP on localhost."""
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
    """
    PROD: invoke sub-agent via invoke_agent_runtime() and parse SSE stream.

    FIX (Issue #4): now calls _parse_sse_stream() via _sse_lines_from_queue()
    so "span" events are captured and returned in span_data. Previously this
    path had its own inline loop that discarded span events and returned {}.
    """
    arns = _get_runtime_arns()
    if agent_name not in arns:
        raise RuntimeError(
            f"No runtime ARN for '{agent_name}'. "
            f"Available: {list(arns.keys())}. "
            f"Run preflight_check_arns() at cold start to catch this earlier."
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
            loop.call_soon_threadsafe(chunk_queue.put_nowait, None)   # sentinel

    loop.run_in_executor(_executor, _stream_in_thread)

    # FIX: use shared _parse_sse_stream via _sse_lines_from_queue
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
    """Route to local or prod invoke path."""
    if LOCAL_MODE:
        return await _invoke_sub_agent_local(agent_name, payload, token_queue, port)
    return await _invoke_sub_agent_prod(agent_name, payload, token_queue)


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

    FIX (Issue #6): runs preflight_check_arns() at cold start (prod only)
    so missing ARNs fail fast with a clear error message instead of
    silently failing mid-request for the first affected user.

    Args:
        session_id:  User thread_id passed to every sub-agent.
        domain:      "pharma" passed to sub-agents.
        token_queue: Sub-agent events put here for Supervisor to re-stream.

    Returns:
        List of StructuredTool objects, one per registered agent + ask_user.
    """
    # FIX: pre-flight — validates all ARNs present before accepting requests
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

                # Agent-scoped session_id prevents checkpoint cross-contamination
                agent_session_id = f"{session_id}__{agent_name}"

                # Safety agent: use throwaway queue so PASSED/BLOCKED text
                # is intercepted before reaching the user stream
                effective_queue = asyncio.Queue() if agent_name == "safety" else token_queue

                t0 = asyncio.get_event_loop().time()
                answer, span_data = await _invoke_sub_agent(
                    agent_name  = agent_name,
                    payload     = {"message": query, "session_id": agent_session_id, "domain": domain},
                    token_queue = effective_queue,
                    port        = agent_port,
                )
                elapsed_ms = round((asyncio.get_event_loop().time() - t0) * 1000, 2)

                # Safety verdict forwarding
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

                # Buffer span for TracerMiddleware (after_agent reads via pop_span_buffer)
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

    # Supervisor-native HITL tool (not from registry — always present)
    tools.append(build_ask_user_tool())
    log.info(f"[A2A] Built {len(tools)} tools: {[t.name for t in tools]}")
    return tools


# ── SSM registry writer ───────────────────────────────────────────────────────

def register_agent(name: str, description: str, port: int = 0) -> None:
    """
    Add or update an agent in the SSM registry.
    Call after deploying a new AgentCore Runtime.
    """
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