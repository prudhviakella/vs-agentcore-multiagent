"""
agents/supervisor/a2a_tools.py
================================
Agent-to-Agent (A2A) Tool Layer.

ROLE
----
This module is the bridge between the supervisor LangGraph agent and the
five sub-agent AgentCore runtimes. It builds LangChain StructuredTools
that the supervisor's gpt-5.5 can call by name with typed arguments.

WHAT IT BUILDS
--------------
For each agent in the SSM registry (knowledge, research, chart):
  StructuredTool(
    name        = "knowledge_agent",
    description = "...",        ← what gpt-5.5 reads to decide when to call it
    args_schema = AgentInput,   ← Pydantic: enforces query: str
    coroutine   = tool_func,    ← async fn that calls AgentCore via boto3
  )
Plus one special tool: clarify___ask_user_input (HITL).

THE THREAD PROBLEM AND HOW WE SOLVE IT
---------------------------------------
boto3 is synchronous. AgentCore invocations are long-running (15-30s each).
If we called boto3 directly in an async coroutine it would block the entire
event loop, freezing all concurrent SSE streams.

Solution: run boto3 in a ThreadPoolExecutor. But then the streaming response
(chunks arriving over seconds) needs to cross back into async world.

  Thread world:           Async world:
  boto3 chunks            event loop (astream_events)
       │                         │
       │  loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
       └─────────────────────────►│
                                  │  await chunk_queue.get()
                                  └── _sse_lines_from_queue() yields lines

call_soon_threadsafe is the ONLY safe way to write to an asyncio.Queue
from a thread. Direct put_nowait from a thread is not thread-safe.

TOKEN QUEUE vs CHUNK QUEUE
--------------------------
Two separate queues per sub-agent call:

  chunk_queue  (asyncio.Queue, per invocation)
    Thread → chunk_queue via call_soon_threadsafe
    _sse_lines_from_queue reads raw bytes/lines from it
    Exists only for the duration of one sub-agent call

  token_queue  (asyncio.Queue, per request, passed in from app.py)
    _parse_sse_stream → token_queue for chart/interrupt/tool events
    supervisor _stream_supervisor drains this via _drain_queue()
    This is how chart events reach the SSE response to the browser

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
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

REGION     = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")

# Module-level thread pool — shared across all sub-agent invocations.
# max_workers=10 supports up to 10 concurrent sub-agent boto3 calls.
# This is well above the practical maximum (supervisor calls at most
# 4-5 tools per request, rarely in parallel).
_executor = ThreadPoolExecutor(max_workers=10)




# ── HITL Exception ─────────────────────────────────────────────────────────

class HITLInterrupt(Exception):
    """
    Raised by ask_user_input tool to signal a Human-In-The-Loop pause.

    WHY AN EXCEPTION?
    -----------------
    LangGraph's interrupt mechanism works by raising an exception inside
    a tool. LangGraph catches it, checkpoints the full message state to
    Postgres, then surfaces the interrupt to the caller.

    The supervisor handler (app.py) catches this at the top level and
    yields an {"type": "interrupt"} SSE event to the UI.
    UI shows the HITL card, disables input, waits for user selection.
    User's answer comes back via POST /resume → supervisor reloads checkpoint.

    Fields
    ------
    question      : str        — what to ask the user
    options       : list[str]  — pre-defined choices (always from real data)
    allow_freetext: bool       — whether user can type a custom answer
    """
    def __init__(self, question: str, options: list[str], allow_freetext: bool = True):
        self.question       = question
        self.options        = options
        self.allow_freetext = allow_freetext
        super().__init__(f"HITL: {question}")


# ── HITL Tool ──────────────────────────────────────────────────────────────

class AskUserInput(BaseModel):
    """
    Pydantic schema for the ask_user_input tool.

    WHY PYDANTIC?
    -------------
    gpt-5.5 generates tool call arguments as JSON. Without a schema,
    the LLM might pass wrong types (list as string, missing fields etc).
    Pydantic validates and coerces arguments before the coroutine runs.
    A bad tool call fails fast with a clear error rather than silently
    producing wrong behaviour downstream.
    """
    question:       str       = Field(description="The clarifying question to ask the user")
    options:        list[str] = Field(description="List of specific options (2-6 items) — must come from real database results, never invented")
    allow_freetext: bool      = Field(default=True, description="Whether to allow free-text response in addition to options")


def build_ask_user_tool() -> Any:
    """
    Build the HITL clarification tool.

    The tool raises HITLInterrupt which LangGraph catches and turns into
    a checkpoint + interrupt event. This is intentional — it is not an
    error path but the designed mechanism for pausing execution.

    The description is critical: it tells gpt-5.5 to search for real
    options BEFORE calling this tool. Without this instruction, the LLM
    tends to invent plausible-sounding but wrong option values.
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
        name        = "clarify___ask_user_input",
        description = (
            "Ask the user a clarifying question when the query is too vague to answer precisely. "
            "IMPORTANT: First search for real candidates using research_agent or knowledge_agent. "
            "Then call this tool with REAL options from the database — never invented names. "
            "This PAUSES execution and checkpoints state until the user responds."
        ),
        args_schema = AskUserInput,
    )


# ── Agent Registry ─────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_agent_registry() -> list[dict]:
    """
    Load the agent registry from SSM. Cached for container lifetime.

    The registry is a JSON list stored in SSM:
      [
        {"name": "knowledge", "description": "...", "port": 8081},
        {"name": "research",  "description": "...", "port": 8082},
        ...
      ]

    WHY lru_cache?
    The registry rarely changes — it's set once at deploy time.
    Caching avoids an SSM API call on every request.
    If the registry changes, the container must restart (redeploy).
    This is acceptable — registry changes happen with agent deployments.
    """
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


def _get_runtime_arns() -> dict[str, str]:
    """
    Fetch current runtime ARNs from SSM for all registered agents.

    WHY NO lru_cache HERE?
    ----------------------
    Unlike the registry (which is stable), runtime ARNs change every time
    an agent is redeployed. AgentCore generates a new ARN suffix on each
    deployment. If we cached the ARN, a redeployed sub-agent would be
    unreachable until the supervisor container also restarts.

    Fetching fresh on every call (one SSM GetParameter per agent per request)
    adds ~5-10ms per agent but guarantees we always have the current ARN.
    This is the correct trade-off for a platform where agents are redeployed
    independently during development.
    """
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





# ── SSE Stream Parsing ─────────────────────────────────────────────────────

async def _parse_sse_stream(
    agent_name:  str,
    line_iter:   AsyncIterator[str],
    token_queue: asyncio.Queue,
) -> tuple[str, dict]:
    """
    Parse SSE lines from a sub-agent response into typed events.

    This is the core protocol parser for agent-to-agent communication.
    Every sub-agent (knowledge, research, chart) speaks the same
    SSE event protocol — this one function handles all of them.

    WHAT EACH EVENT TYPE MEANS
    --------------------------
    token       → sub-agent is streaming its answer text
                  Accumulated into full_answer for the supervisor's tool result.
                  Also put into token_queue so the UI can see intermediate text.

    tool_start  → sub-agent is calling one of ITS tools (e.g. Neo4j query)
    tool_end    → sub-agent tool finished
                  Both forwarded to token_queue → UI shows nested spinner.

    chart       → chart_agent produced a Chart.js config
                  Put into token_queue → _drain_queue() in app.py picks it up
                  → UI renders the chart BEFORE the supervisor's answer tokens.
                  Critical: this must arrive in the queue while the tool is
                  still "running" so drain at tool_end catches it first.

    interrupt   → sub-agent (unlikely but possible) triggered HITL
                  Re-raised as HITLInterrupt → propagates up to handler().

    error       → sub-agent failed
                  Re-raised as RuntimeError → supervisor gets a tool error result.

    span        → observability data (latency, tokens, status)
                  Stored for sub-agent tracing (emitted by sub-agents, not processed here).

    done        → sub-agent stream complete
                  If it includes a consolidated answer, use it as final result.

    Parameters
    ----------
    agent_name  : str            — used in error messages and span tagging
    line_iter   : AsyncIterator  — yields raw SSE lines (from chunk_queue)
    token_queue : asyncio.Queue  — shared queue for chart/interrupt events

    Returns
    -------
    full_answer : str — complete text response from sub-agent (supervisor tool result)
    """
    full_answer = ""

    async for line in line_iter:
        line = line.strip()
        if not line:
            continue
        # Strip "data: " SSE prefix
        if line.startswith("data: "):
            line = line[6:]
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # malformed line — skip silently

        etype = event.get("type", "")

        if etype == "token":
            # Sub-agent streaming its answer token by token.
            # Accumulated locally (supervisor sees the full answer as tool result)
            # and forwarded to token_queue (UI can show intermediate progress).
            c = event.get("content", "")
            if c:
                full_answer += c
                await token_queue.put({"type": "token", "content": c})

        elif etype == "chart":
            # Chart.js config from chart_agent.
            # Put directly into token_queue — _drain_queue() in app.py will
            # pick this up at the next on_tool_end LangGraph event, guaranteeing
            # the chart renders BEFORE the supervisor's answer tokens.
            await token_queue.put({
                "type":       "chart",
                "config":     event.get("config", {}),
                "chart_type": event.get("chart_type", "bar"),
            })

        elif etype == "interrupt":
            # Sub-agent triggered a HITL interrupt.
            # Re-raise — propagates through _invoke_sub_agent_prod() →
            # build_a2a_tools tool_func → LangGraph ToolNode → supervisor handler.
            raise HITLInterrupt(
                question       = event.get("question", "Please clarify:"),
                options        = event.get("options", []),
                allow_freetext = event.get("allow_freetext", True),
            )

        elif etype == "error":
            # Sub-agent reported an error (timeout, Cypher failure, etc.)
            # Raise RuntimeError — LangGraph catches it as a tool error and
            # adds a ToolMessage with the error text. Supervisor can then decide
            # to retry with a different query or report failure to the user.
            raise RuntimeError(f"Sub-agent '{agent_name}': {event.get('message', '')}")

        elif etype == "done":
            pass  # stream complete — full_answer already built from token events

    return full_answer


async def _sse_lines_from_queue(queue: asyncio.Queue) -> AsyncIterator[str]:
    """
    Convert chunk_queue items into SSE text lines for _parse_sse_stream.

    The thread puts raw bytes into chunk_queue (complete lines guaranteed
    by the partial-line buffering in _stream_in_thread).
    This generator decodes bytes, splits on newlines, and yields each line.

    SENTINEL PATTERN
    ----------------
    _stream_in_thread puts None into chunk_queue when it finishes
    (normal completion or exception). This generator stops on None.
    Without the sentinel, we'd await queue.get() forever after the thread
    completes — hanging the entire request.
    """
    while True:
        chunk = await queue.get()
        if chunk is None:
            return  # sentinel received — thread finished, stop iterating
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else chunk
        for line in text.splitlines():
            yield line


# ── Production Sub-Agent Invocation ───────────────────────────────────────

async def _invoke_sub_agent_prod(
    agent_name:  str,
    payload:     dict,
    token_queue: asyncio.Queue,
) -> tuple[str, dict]:
    """
    Invoke a sub-agent AgentCore runtime and stream its response.

    THE THREAD + QUEUE ARCHITECTURE
    --------------------------------
    Problem: boto3 is synchronous. A 15-30s blocking call in an async
    coroutine would freeze the event loop — no other SSE streams, no
    keepalive pings, nothing.

    Solution:
      1. Dispatch _stream_in_thread to ThreadPoolExecutor (non-blocking)
      2. Thread uses loop.call_soon_threadsafe to safely put chunks into
         chunk_queue (an asyncio.Queue living in the event loop)
      3. _sse_lines_from_queue yields lines from chunk_queue asynchronously
      4. _parse_sse_stream processes lines and puts events into token_queue

    This pattern keeps the event loop fully responsive during the entire
    15-30s sub-agent call.

    PARTIAL LINE BUFFERING
    ----------------------
    AgentCore streams raw bytes. A single SSE line (e.g. a large Chart.js
    JSON config) can span multiple chunks. _stream_in_thread accumulates
    bytes until it sees \n, then sends complete lines only. This prevents
    _parse_sse_stream from receiving half-JSON and failing to parse it.

    BOTO3 TIMEOUT
    -------------
    read_timeout=300: allows up to 5 minutes for a sub-agent response.
    Default boto3 timeout is 60s — sub-agents on complex queries can take
    90-120s, causing ReadTimeoutError with the default setting.
    retries=0: no automatic retry on streaming calls — a retry would
    restart the stream from the beginning, causing duplicate events.
    """
    arns = _get_runtime_arns()
    if agent_name not in arns:
        raise RuntimeError(
            f"No runtime ARN for '{agent_name}'. Available: {list(arns.keys())}."
        )

    runtime_arn = arns[agent_name]
    session_id  = payload.get("session_id", f"{agent_name}-session")

    # Get the currently running event loop.
    # get_running_loop() is the correct call inside an async context.
    # (asyncio.get_event_loop() is deprecated in Python 3.10+ and may
    # return a closed or wrong loop in some contexts.)
    loop        = asyncio.get_running_loop()

    # Per-invocation queue for raw byte chunks from the boto3 thread.
    # Separate from token_queue: chunk_queue carries raw SSE bytes,
    # token_queue carries parsed typed events.
    chunk_queue = asyncio.Queue()

    def _stream_in_thread():
        """
        Run in ThreadPoolExecutor. Calls boto3 and pipes chunks to chunk_queue.

        All queue writes use loop.call_soon_threadsafe — the ONLY safe way
        to interact with an asyncio primitive from a non-async thread.
        Direct queue.put_nowait() from a thread is NOT thread-safe in asyncio.
        """
        try:
            from botocore.config import Config as _BotocoreConfig
            client = boto3.client(
                "bedrock-agentcore",
                region_name = REGION,
                config      = _BotocoreConfig(
                    read_timeout    = 300,   # 5 min — sub-agents can take 90-120s
                    connect_timeout = 10,    # fast fail on connection issues
                    retries         = {"max_attempts": 0},  # no retry on streams
                ),
            )

            response = client.invoke_agent_runtime(
                agentRuntimeArn  = runtime_arn,
                runtimeSessionId = session_id,
                payload          = json.dumps(payload).encode("utf-8"),
            )

            streaming_body = response.get("response")
            if streaming_body:
                # Partial-line buffer: accumulate bytes until we have
                # complete lines (ending with \n) before sending to queue.
                # Prevents _parse_sse_stream from receiving split JSON.
                _partial = b""
                for chunk in streaming_body.iter_chunks(chunk_size=65536):
                    if chunk:
                        _partial += chunk
                        if b"\n" in _partial:
                            last_nl  = _partial.rfind(b"\n")
                            complete = _partial[:last_nl + 1]  # everything up to last \n
                            _partial = _partial[last_nl + 1:]  # hold the rest for next iteration
                            # Schedule put_nowait on the event loop — thread-safe
                            loop.call_soon_threadsafe(chunk_queue.put_nowait, complete)

                # Flush any remaining bytes after the stream ends
                if _partial:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, _partial)
            else:
                # Non-streaming response (unexpected but handle it)
                raw = response.get("body", b"")
                if raw:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, raw)

        except Exception as exc:
            # Encode error as an SSE event so _parse_sse_stream raises RuntimeError
            # This keeps the error handling in one place (parse_sse_stream)
            err = json.dumps({"type": "error", "message": str(exc)}).encode()
            loop.call_soon_threadsafe(chunk_queue.put_nowait, err)
        finally:
            # Always send the sentinel — even on exception.
            # Without this, _sse_lines_from_queue awaits forever.
            loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

    # Dispatch to thread pool — returns immediately, does not block event loop
    loop.run_in_executor(_executor, _stream_in_thread)

    # Now stream the response asynchronously via the chunk_queue bridge
    full_answer = await _parse_sse_stream(
        agent_name, _sse_lines_from_queue(chunk_queue), token_queue
    )
    log.info(f"[A2A PROD] {agent_name} done  answer_len={len(full_answer)}")
    return full_answer


# ── Tool Factory ───────────────────────────────────────────────────────────

class AgentInput(BaseModel):
    """Pydantic schema shared by all A2A tools. Enforces typed query argument."""
    query: str = Field(description="The query or task to send to this agent")


def build_a2a_tools(
    session_id:  str,
    domain:      str,
    token_queue: asyncio.Queue,
) -> list[Any]:
    """
    Build StructuredTool instances for all registered sub-agents.

    Called per request from build_supervisor_agent() in agent.py.
    Rebuilds tools every time because tool_func closes over session_id
    and token_queue — both are unique per request.

    WHY CLOSURES?
    -------------
    Each tool_func needs to know:
      session_id  → for sub-agent session namespacing and span attribution
      token_queue → to route chart/interrupt events to the right SSE stream

    These change per request. By capturing them in a closure (via
    make_tool_func), every tool has its own isolated reference.

    WITHOUT make_tool_func (just using a lambda or def inside a loop):
      Python closures capture variables by reference, not value.
      All tools would share the SAME agent_name from the last loop iteration.
      make_tool_func forces early binding via function argument.

    SUB-AGENT SESSION ISOLATION
    ---------------------------
    session_id for sub-agents is namespaced:
      "{session_id}__{agent_name}"
      e.g. "c73e7373__knowledge"

    Why: each sub-agent has its own LangGraph graph with its own Postgres
    checkpoint. Using the same session_id across agents would mix their
    message histories. Namespacing ensures each sub-agent sees only its
    own conversation history.

    Parameters
    ----------
    session_id  : str            — supervisor session, used for namespacing + tracing
    domain      : str            — "pharma" etc, passed to sub-agents for context
    token_queue : asyncio.Queue  — shared event bridge for chart/interrupt events

    Returns
    -------
    List of StructuredTool instances ready to pass to create_agent().
    """
    registry = _get_agent_registry()
    tools    = []

    for agent_def in registry:
        name        = agent_def["name"]
        description = agent_def["description"]

        def make_tool_func(agent_name: str):
            """
            Create a tool coroutine with agent_name bound by value.

            make_tool_func is called immediately in the loop, binding
            agent_name as a function argument (value capture).
            Without this wrapper, all tools would reference the same
            `name` variable from the enclosing loop scope.
            """
            async def tool_func(query: str) -> str:
                log.info(f"[Supervisor] → {agent_name}  query={query[:60]}...")

                # Namespace the session so sub-agent checkpoint is isolated
                # from the supervisor's own checkpoint and from other sub-agents.
                agent_session_id = f"{session_id}__{agent_name}"

                t0 = asyncio.get_running_loop().time()
                answer = await _invoke_sub_agent_prod(
                    agent_name  = agent_name,
                    payload     = {
                        "message":    query,
                        "session_id": agent_session_id,
                        "domain":     domain,
                    },
                    token_queue = token_queue,
                )
                elapsed_ms = round((asyncio.get_running_loop().time() - t0) * 1000, 2)
                log.info(f"[A2A] {agent_name} done  elapsed={elapsed_ms:.0f}ms")

                # Return full answer — becomes the ToolMessage content
                # that gpt-5.5 sees in the next ReAct turn
                return answer

            tool_func.__name__ = f"{agent_name}_agent"
            return tool_func

        tool = StructuredTool.from_function(
            coroutine    = make_tool_func(name),
            name         = f"{name}_agent",
            description  = description,
            args_schema  = AgentInput,
        )
        tools.append(tool)
        log.info(f"[A2A] Registered tool: {name}_agent")

    # HITL tool always added last — ordering does not affect LLM choice
    # but putting it last makes it easy to find in logs.
    tools.append(build_ask_user_tool())
    log.info(f"[A2A] Built {len(tools)} tools: {[t.name for t in tools]}")
    return tools