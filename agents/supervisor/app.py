"""
agents/supervisor/app.py
=========================
Supervisor Agent — AgentCore Runtime Entrypoint.

ROLE IN THE PLATFORM
--------------------
The supervisor is the orchestration brain. It receives every user message,
decides which sub-agents to call, in what order, with what queries, and
synthesises their responses into a final answer.

It runs as an AWS Bedrock AgentCore Runtime — a containerised FastAPI-style
app that AgentCore manages (scaling, routing, lifecycle). The platform
invokes it via boto3 invoke_agent_runtime() and reads the streaming response.

INTERNAL ARCHITECTURE
---------------------
  User message
      │
      ▼
  handler()          ← AgentCore entrypoint (@app.entrypoint)
      │
      ├── _ensure_cold_start()   ← build LangGraph graph once, reuse forever
      │
      ├── token_queue            ← asyncio.Queue() bridges thread↔async worlds
      │
      └── _stream_supervisor()   ← drives LangGraph, drains queue, yields SSE events

STREAMING MODEL
---------------
Two event sources run simultaneously:
  1. LangGraph astream_events() — async, yields on every node/edge execution
  2. token_queue                — A2A tools (running in threads) push chart/interrupt events here

_drain_queue() is called on EVERY LangGraph event to interleave the two
streams in the correct order. Without this, chart events would arrive after
answer tokens — wrong order in the UI.

SSE EVENTS EMITTED
------------------
  {"type": "token",     "content": str}     — raw LLM token (thinking + answer mixed)
  {"type": "tool_start","name":    str}     — sub-agent starting (UI shows spinner)
  {"type": "tool_end",  "name":    str}     — sub-agent done
  {"type": "chart",     "config":  dict}    — Chart.js config from chart_agent
  {"type": "interrupt", ...}                — HITL clarification needed
  {"type": "error",     "message": str}     — something went wrong
  {"type": "done",      "latency_ms": int}  — stream complete

NOTE: tokens include <thinking> blocks — the UI state machine in app.py
separates them into reasoning cards vs answer bubbles client-side.
"""

import asyncio
import json
import logging
import os
import re
import time

import boto3
import watchtower
from bedrock_agentcore import BedrockAgentCoreApp, BedrockAgentCoreContext

from agents.supervisor.agent import build_supervisor_agent, build_supervisor_cold_start
from agents.supervisor.a2a_tools import HITLInterrupt
from agents.supervisor.middleware.output_guardrail import _FALLBACK_MARKER

# ── Logging ────────────────────────────────────────────────────────────────
# Standard Python logging goes to CloudWatch via Watchtower.
# We also write to the AgentCore-managed log group so logs appear in the
# AWS console alongside the runtime metrics.

_LOG_GROUP = os.environ.get("LOG_GROUP_NAME", "/agentcore/vs-agentcore-ma/supervisor-agent")

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
try:
    # AgentCore runtimes have a default log group named:
    #   /aws/bedrock-agentcore/runtimes/{runtime_id}-DEFAULT
    # We write to it via Watchtower so logs survive container restarts.
    _rt_id   = os.environ.get("AGENT_RUNTIME_ID", "vs_agentcore_ma_supervisor-S2wbNj9c4x")
    _lg_name = f"/aws/bedrock-agentcore/runtimes/{_rt_id}-DEFAULT"
    _cw = watchtower.CloudWatchLogHandler(
        log_group_name    = _lg_name,
        boto3_client      = boto3.client("logs", region_name=os.environ.get("AWS_REGION", "us-east-1")),
        create_log_group  = False,   # log group is created by AgentCore — do not recreate
        create_log_stream = True,    # individual stream per container instance is fine
    )
    _cw.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    root = logging.getLogger()
    # Guard against duplicate handlers if the module is imported multiple times
    if not any(isinstance(h, watchtower.CloudWatchLogHandler) for h in root.handlers):
        root.addHandler(_cw)
except Exception:
    # Never crash on logging setup — degraded logging is better than no service
    pass

log = logging.getLogger(__name__)

# ── AgentCore App ──────────────────────────────────────────────────────────
# BedrockAgentCoreApp is the AgentCore SDK entrypoint.
# It exposes @app.entrypoint which AgentCore calls for every incoming request.
# Functionally similar to @app.post() in FastAPI.
app = BedrockAgentCoreApp()

# ── Cold start cache ───────────────────────────────────────────────────────
# _cold_start_objects holds the expensive objects built once at container
# startup: compiled LangGraph graph, Postgres checkpointer, LLM client.
# None means "not yet initialised" — initialised on first request.
# Subsequent requests reuse these objects — critical for performance since
# building the LangGraph graph with Postgres connection takes 3-5 seconds.
_cold_start_objects = None

# ── EPISODIC tail stripping ────────────────────────────────────────────────
# The supervisor prompt instructs gpt-5.5 to append an episodic memory tag
# at the end of every response:
#   "...final answer text.\nEPISODIC: YES 0.95"
# This tag is for the episodic memory middleware — it should NEVER reach
# the UI. _EPISODIC_PATTERN strips it from the tail buffer before yielding.
_EPISODIC_PATTERN = re.compile(r'\s*\nEPISODIC:\s*(YES|NO)[\d.\s]*$', re.IGNORECASE)

# How many chars to hold back at the end of the token stream.
# 60 chars is enough to contain the longest EPISODIC suffix
# ("\nEPISODIC: YES 0.95" = 20 chars) with margin.
_TAIL_SIZE = 60


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_interrupt_args(iv: dict) -> dict:
    """
    Extract HITL question/options from various interrupt value shapes.

    LangGraph interrupt() values can arrive in different shapes depending
    on how the interrupt was triggered (ask_user_input tool vs graph-level
    interrupt). This normalises both into {question, options, allow_freetext}.

    Shape 1 — from ask_user_input StructuredTool:
      {"action_requests": [{"args": {"question": ..., "options": [...]}}]}

    Shape 2 — direct dict from graph interrupt:
      {"question": ..., "options": [...]}
    """
    if not iv or not isinstance(iv, dict):
        return {}
    action_requests = iv.get("action_requests", [])
    if action_requests:
        # Shape 1: StructuredTool interrupt
        return action_requests[0].get("args", {})
    # Shape 2: direct dict — return as-is
    return iv


def _load_langsmith_from_ssm():
    """
    Load LangSmith tracing credentials from AWS Secrets Manager at cold start.

    WHY SSM not env vars:
      LangSmith API keys are secrets — they must not be baked into container
      images or task definitions. We store them in Secrets Manager and load
      at cold start. The key is written to os.environ so LangChain's
      LangSmithTracer picks it up automatically via the standard env var
      LANGSMITH_API_KEY.

    Short-circuits if LANGSMITH_API_KEY is already set (e.g. local dev mode).
    """
    if os.environ.get("LANGSMITH_API_KEY"):
        return  # already set — skip SSM call
    try:
        prefix = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
        sm     = boto3.client("secretsmanager", region_name="us-east-1")
        # Secret stored as JSON: {"api_key": "lsv2_...", "project": "...", "tracing": "true"}
        secret = json.loads(sm.get_secret_value(SecretId=f"{prefix}/langsmith")["SecretString"])
        os.environ["LANGSMITH_API_KEY"] = secret.get("api_key", "")
        os.environ["LANGSMITH_PROJECT"]  = secret.get("project", "langchain-agent-experiments")
        os.environ["LANGSMITH_TRACING"]  = secret.get("tracing", "true")
        log.info("[Supervisor] LangSmith credentials loaded from SSM")
    except Exception as e:
        # Non-fatal — tracing is observability, not core functionality
        log.warning(f"[Supervisor] LangSmith secret not found — tracing disabled: {e}")


async def _ensure_cold_start():
    """
    Initialise shared expensive objects exactly once per container lifetime.

    COLD START vs WARM REQUEST
    --------------------------
    Cold start (first request after container start):
      - Load OpenAI key from Secrets Manager
      - Load LangSmith key from Secrets Manager
      - Build LangGraph compiled graph (connects to Postgres, builds tools)
      - Takes ~3-5 seconds

    Warm request (subsequent requests):
      - _cold_start_objects already set — returns immediately
      - Zero overhead

    WHY global variable and not a singleton class:
      AgentCore calls handler() as a coroutine in a single process.
      A module-level global is the simplest correct pattern here.
      No lock needed because asyncio is single-threaded — only one
      _ensure_cold_start() coroutine runs at a time.
    """
    global _cold_start_objects
    if _cold_start_objects is None:
        log.info("[Supervisor] Cold start — building shared objects")

        # Load OpenAI key from Secrets Manager if not already in env.
        # Secret stored as JSON: {"api_key": "sk-..."}
        if not os.environ.get("OPENAI_API_KEY"):
            prefix = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
            sm     = boto3.client("secretsmanager", region_name="us-east-1")
            secret = json.loads(sm.get_secret_value(SecretId=f"{prefix}/openai")["SecretString"])
            os.environ["OPENAI_API_KEY"] = secret.get("api_key", "")
            log.info("[Supervisor] OpenAI key loaded")

        # Load LangSmith before building the graph so LangChain picks up
        # the tracing env vars during graph compilation
        _load_langsmith_from_ssm()

        # Build the LangGraph compiled graph with all tools and checkpointer.
        # Returns: {graph, llm, tools, checkpointer, registry}
        # See agents/supervisor/agent.py for full build details.
        _cold_start_objects = await build_supervisor_cold_start()
        log.info("[Supervisor] Cold start complete")

    return _cold_start_objects


# ── AgentCore Entrypoint ───────────────────────────────────────────────────

@app.entrypoint
async def handler(payload: dict, context: BedrockAgentCoreContext):
    """
    Main entrypoint — called by AgentCore for every incoming request.

    This is an async generator — it yields SSE events that AgentCore
    streams back to the platform as a chunked HTTP response.

    FLOW
    ----
    1. Extract request fields from payload
    2. Ensure cold start objects are ready
    3. Create a fresh token_queue for this request
    4. Build the per-request agent (injects token_queue into tool closures)
    5. If resume: repair dangling tool calls from HITL checkpoint
    6. Stream via _stream_supervisor() — yields SSE events
    7. Always yield "done" in finally block

    RESUME FLOW
    -----------
    When user answers a HITL question, the UI sends POST /resume.
    Platform invokes supervisor with resume=True and user_answer set.
    The supervisor:
      a. Loads the LangGraph checkpoint from Postgres (thread_id is the key)
      b. Finds dangling tool calls (ask_user_input never got a ToolMessage)
      c. Inserts synthetic ToolMessages with the user's answer
      d. Resumes graph execution from the repaired state
    """
    t0          = time.perf_counter()
    message     = payload.get("message",     "")
    thread_id   = payload.get("thread_id",   "") or getattr(context, "session_id", "")
    domain      = payload.get("domain",      "pharma")
    is_resume   = payload.get("resume",      False)
    user_answer = payload.get("user_answer", "")

    log.info(f"[Supervisor] {'resume' if is_resume else 'chat'}  thread={thread_id[:8] if thread_id else 'n/a'}")

    try:
        cold = await _ensure_cold_start()

        # token_queue is the thread→async bridge.
        # A2A tools run in a ThreadPoolExecutor (boto3 is sync).
        # When a sub-agent returns a chart or interrupt event, the tool
        # puts it here via loop.call_soon_threadsafe(queue.put_nowait, event).
        # _drain_queue() in _stream_supervisor picks it up.
        # A FRESH queue per request prevents events leaking between sessions.
        token_queue = asyncio.Queue()

        # LangGraph config — thread_id is the Postgres checkpoint key.
        # Every session gets its own isolated message history and HITL state.
        config = {
            "configurable": {
                "thread_id":  thread_id,
                "user_id":    thread_id,
                "session_id": thread_id,
                "domain":     domain,
            },
            "recursion_limit": 100,  # max ReAct loop iterations before LangGraph raises
        }
        # agent_context passes session metadata into tool closures
        agent_context = {"user_id": thread_id, "session_id": thread_id, "domain": domain}

        # build_supervisor_agent() takes the compiled graph from cold start
        # and injects token_queue into all A2A tool closures for this request.
        # Cheap — no graph recompilation.
        agent = await build_supervisor_agent(
            session_id  = thread_id,
            domain      = domain,
            token_queue = token_queue,
            **cold,
        )

        if is_resume:
            # ── HITL Resume: repair dangling tool calls ────────────────────
            # When the supervisor called ask_user_input, LangGraph checkpointed
            # the state with an AIMessage containing tool_calls but no matching
            # ToolMessage (because the tool was interrupted, not completed).
            # OpenAI rejects messages where tool_calls exist without responses
            # (error 400: "tool_calls must be followed by tool messages").
            # We repair by inserting synthetic ToolMessages before resuming.
            try:
                from langchain_core.messages import ToolMessage
                state    = await agent.aget_state(config)
                messages = state.values.get("messages", [])

                # Collect IDs of tool calls that already have responses
                result_ids = {getattr(m, "tool_call_id", None) for m in messages if hasattr(m, "tool_call_id")}

                # Find tool calls with no matching response (dangling)
                dangling = [
                    tc for msg in messages
                    for tc in getattr(msg, "tool_calls", [])
                    if tc.get("id") not in result_ids
                ]

                if dangling:
                    log.info(f"[Supervisor] Repairing {len(dangling)} dangling tool call(s)")
                    # Insert a synthetic ToolMessage for each dangling call
                    # so the message history is valid before resuming
                    await agent.aupdate_state(config, {
                        "messages": [
                            ToolMessage(
                                content      = f"[Interrupted — user answered: {user_answer}]",
                                tool_call_id = tc["id"],
                                name         = tc.get("name", "unknown"),
                            )
                            for tc in dangling
                        ]
                    })
            except Exception as e:
                log.warning(f"[Supervisor] State repair failed (continuing): {e}")

            # Resume: inject user answer as a new HumanMessage.
            # The supervisor LLM sees this and continues from where it paused.
            input_data = {"messages": [{"role": "user", "content": f"[HITL Answer]: {user_answer}. Now search and answer."}]}
        else:
            # Normal chat: just the user's message
            input_data = {"messages": [{"role": "user", "content": message}]}

        # Stream all SSE events from the supervisor back to the platform
        async for event in _stream_supervisor(agent, input_data, config, agent_context, token_queue):
            yield event

    except HITLInterrupt as hitl:
        # HITLInterrupt is raised by the ask_user_input tool directly
        # (alternative HITL path — some LangGraph versions surface it this way)
        log.info("[Supervisor] HITLInterrupt — surfacing to UI")
        yield {
            "type":           "interrupt",
            "question":       hitl.question,
            "options":        hitl.options,
            "allow_freetext": hitl.allow_freetext,
        }
    except Exception as exc:
        log.exception(f"[Supervisor] Handler error: {exc}")
        yield {"type": "error", "message": str(exc)}
    finally:
        # "done" is ALWAYS yielded — even on error or interrupt.
        # The platform and UI both rely on this to know the stream is complete.
        elapsed = round((time.perf_counter() - t0) * 1_000, 2)
        log.info(f"[Supervisor] Done  latency_ms={elapsed}")
        yield {"type": "done", "latency_ms": elapsed}


# ── Core Streaming Logic ───────────────────────────────────────────────────

async def _stream_supervisor(agent, input_data, config, agent_context, token_queue):
    """
    Drive the LangGraph ReAct loop and yield typed SSE events.

    THE TWO-STREAM PROBLEM
    ----------------------
    LangGraph emits events asynchronously via astream_events().
    A2A tools (knowledge_agent, chart_agent etc.) run in a ThreadPoolExecutor
    because boto3 is synchronous. When a tool produces a chart or interrupt
    event, it cannot directly yield into this async generator. Instead it:

      loop.call_soon_threadsafe(token_queue.put_nowait, event)

    This schedules the put_nowait on the event loop's next tick.
    _drain_queue() is called on EVERY LangGraph event to pick these up
    at the earliest possible moment — guaranteeing chart events appear
    BEFORE the supervisor's answer tokens that follow them.

    TAIL BUFFER
    -----------
    All LLM tokens are buffered with a _TAIL_SIZE holdback. This allows
    _EPISODIC_PATTERN to strip the memory tag ("EPISODIC: YES 0.95")
    that gpt-5.5 appends to every response, before it reaches the UI.

    INTERRUPT PATHS
    ---------------
    HITL interrupts can surface via three different paths depending on the
    LangGraph version and how the interrupt was triggered:
      Path A: on_tool_end for "ask_user_input" — most common
      Path B: on_chain_end with __interrupt__ key — LangGraph graph-level
      Path C: checkpoint tasks with interrupts — fallback after stream ends
    All three are handled.
    """
    _tail_buffer    = ""    # holds back last _TAIL_SIZE chars for EPISODIC strip
    hitl_input      = {}    # stores ask_user_input args captured at on_tool_start
    interrupt_fired = False # ensures we don't double-emit interrupt events

    def _flush_safe():
        """
        Yield everything except the last _TAIL_SIZE chars.

        Called on every token. Holds back the tail so we can strip
        EPISODIC: YES/NO suffix when the stream ends (or at flush time).

        Example:
          _tail_buffer = "Phase 3 trials are most common.\nEPISODIC: YES 0.95"
          _flush_safe() → yields "Phase 3 trials are most common."
                        → keeps "\nEPISODIC: YES 0.95" in buffer
          flush at end  → _EPISODIC_PATTERN strips it → yields nothing extra
        """
        nonlocal _tail_buffer
        if len(_tail_buffer) > _TAIL_SIZE:
            safe         = _tail_buffer[:-_TAIL_SIZE]  # safe to emit — not near tail
            _tail_buffer = _tail_buffer[-_TAIL_SIZE:]  # hold back tail
            return safe
        return ""  # buffer not yet long enough — hold everything

    async def _drain_queue():
        """
        Non-blocking drain of token_queue.

        Uses get_nowait() (not await get()) because:
          - We only want items already in the queue right now
          - await get() would BLOCK the event loop waiting for future items
          - Blocking here freezes the entire SSE stream for all clients

        Called on EVERY LangGraph event. This is what guarantees the
        correct event ordering:

          on_tool_end (chart_agent)
            → _drain_queue() → finds chart event → yields it FIRST
          on_chat_model_stream (answer tokens)
            → _drain_queue() → empty → yields answer tokens SECOND

          Without draining on every event:
            chart event sits in queue while answer tokens stream to UI
            chart arrives LAST → wrong order in browser
        """
        while True:
            try:
                yield token_queue.get_nowait()  # returns immediately or raises QueueEmpty
            except asyncio.QueueEmpty:
                break  # queue empty — stop draining, continue with LangGraph event

    try:
        # astream_events() drives the full ReAct loop:
        #   supervisor node (LLM generates tool_calls or final answer)
        #   → tools node (executes structured tools — A2A calls)
        #   → supervisor node (LLM sees tool results, decides next step)
        #   → ... until no tool_calls → END
        #
        # version="v2" is required for LangGraph 0.2+ event schema.
        async for event in agent.astream_events(
            input_data,
            config  = config,
            version = "v2",
            context = agent_context,
        ):
            # ── DRAIN FIRST on every event ─────────────────────────────────
            # This is the core of the two-stream interleaving.
            # At on_tool_end specifically, the thread has just finished running
            # the A2A tool. call_soon_threadsafe has scheduled the chart/interrupt
            # event on the event loop. By the time on_tool_end arrives here,
            # that event is sitting in token_queue ready to be drained.
            async for item in _drain_queue():
                yield item  # chart event, interrupt event, or span data

            kind = event.get("event", "")
            name = event.get("name",  "")
            data = event.get("data",  {})

            # ── LLM token stream ───────────────────────────────────────────
            # on_chat_model_stream fires for every token the LLM generates.
            # This includes BOTH:
            #   - <thinking>...</thinking> blocks (supervisor reasoning)
            #   - Final answer tokens (what the user sees)
            # The UI state machine in app.py separates them client-side.
            if kind == "on_chat_model_stream":
                chunk   = data.get("chunk", {})
                content = getattr(chunk, "content", "")
                if isinstance(content, str) and content:
                    _tail_buffer += content   # accumulate in tail buffer
                    safe = _flush_safe()      # get safe-to-emit portion
                    if safe:
                        yield {"type": "token", "content": safe}

            # ── HITL tool starting — capture args ──────────────────────────
            # When the supervisor decides to ask the user, it calls
            # clarify___ask_user_input. We capture its args here at tool_start
            # because on_tool_end fires AFTER the tool runs (which for HITL
            # means after LangGraph has already checkpointed and interrupted).
            elif kind == "on_tool_start" and "ask_user_input" in name:
                raw        = data.get("input", {})
                hitl_input = raw.get("arguments", raw)  # normalise arg shapes
                log.info("[Supervisor] clarify___ask_user_input starting")

            # ── HITL Path A: tool_end for ask_user_input ───────────────────
            # Most common interrupt path. LangGraph fires on_tool_end when
            # the ask_user_input tool completes (which for HITL means the
            # graph has paused and checkpointed).
            # We flush the tail buffer (in case any answer tokens were
            # generated before the interrupt decision) then emit the interrupt.
            elif kind == "on_tool_end" and "ask_user_input" in name:
                if _tail_buffer.strip():
                    clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
                    if clean:
                        yield {"type": "token", "content": clean}
                    _tail_buffer = ""
                yield {
                    "type":           "interrupt",
                    "question":       hitl_input.get("question", "Please clarify:"),
                    "options":        hitl_input.get("options", []),
                    "allow_freetext": hitl_input.get("allow_freetext", True),
                }
                return  # stop streaming — UI takes over with HITL card

            # ── HITL Path B: graph-level __interrupt__ ─────────────────────
            # In some LangGraph versions, interrupt() surfaces via on_chain_end
            # with an "__interrupt__" key in the output dict.
            # This is the graph-level interrupt, not the tool-level one.
            elif kind == "on_chain_end":
                output = data.get("output", {})
                if isinstance(output, dict) and "__interrupt__" in output:
                    interrupts = output["__interrupt__"]
                    if interrupts:
                        iv   = interrupts[0]
                        val  = iv.value if hasattr(iv, "value") else iv
                        args = _extract_interrupt_args(val)
                        interrupt_fired = True
                        if _tail_buffer.strip():
                            clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
                            if clean:
                                yield {"type": "token", "content": clean}
                            _tail_buffer = ""
                        yield {
                            "type":           "interrupt",
                            "question":       args.get("question", "Please clarify:"),
                            "options":        args.get("options", []),
                            "allow_freetext": args.get("allow_freetext", True),
                        }
                        return

    except HITLInterrupt:
        # HITLInterrupt is raised directly by the tool in some code paths.
        # Re-raise to handler() which catches it at the top level.
        raise

    except Exception as exc:
        exc_type = type(exc).__name__
        if "Interrupt" in exc_type or "GraphInterrupt" in exc_type:
            # LangGraph raises GraphInterrupt internally in some versions.
            # Treat as HITL interrupt — same handling as Path A/B.
            interrupt_fired = True
            if _tail_buffer.strip():
                clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
                if clean:
                    yield {"type": "token", "content": clean}
                _tail_buffer = ""
            yield {
                "type":           "interrupt",
                "question":       hitl_input.get("question", "Please clarify:"),
                "options":        hitl_input.get("options", []),
                "allow_freetext": hitl_input.get("allow_freetext", True),
            }
        else:
            # Real error — log and surface to UI
            log.exception(f"[Supervisor] Stream error: {exc}")
            if _tail_buffer.strip():
                clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
                if clean:
                    yield {"type": "token", "content": clean}
                _tail_buffer = ""
            yield {"type": "error", "message": str(exc)}
        return

    # ── End of stream: flush tail buffer ──────────────────────────────────
    # All LangGraph events processed. Flush whatever remains in the tail buffer.
    # _EPISODIC_PATTERN strips the memory tag before yielding.
    if _tail_buffer:
        clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
        if clean:
            yield {"type": "token", "content": clean}

    # ── End of stream: final queue drain ──────────────────────────────────
    # Catch any A2A events that arrived after the last LangGraph event.
    # Rare but possible if call_soon_threadsafe fires on the very last tick.
    async for item in _drain_queue():
        yield item

    # ── HITL Path C: checkpoint-based interrupt detection ─────────────────
    # Fallback for LangGraph versions where the interrupt is only visible
    # in the checkpoint state AFTER streaming completes. We check the
    # checkpoint tasks for any pending interrupts we may have missed.
    if not interrupt_fired:
        try:
            state = await agent.aget_state(config)
            for task in (state.tasks if hasattr(state, "tasks") else []):
                for iv in getattr(task, "interrupts", []):
                    val  = iv.value if hasattr(iv, "value") else iv
                    args = _extract_interrupt_args(val)
                    log.info("[Supervisor] Path C — interrupt in checkpoint")
                    yield {
                        "type":           "interrupt",
                        "question":       args.get("question", "Please clarify:"),
                        "options":        args.get("options", []),
                        "allow_freetext": args.get("allow_freetext", True),
                    }
                    return
        except Exception as e:
            log.warning(f"[Supervisor] Checkpoint check failed: {e}")

    # ── OutputGuardrail fallback ───────────────────────────────────────────
    # If the output guardrail blocked the supervisor's response and replaced
    # it with a fallback message, the final AIMessage contains _FALLBACK_MARKER.
    # We surface it here so the UI shows the safe fallback instead of silence.
    try:
        final_state = await agent.aget_state(config)
        final_msgs  = final_state.values.get("messages", [])
        if final_msgs:
            from langchain_core.messages import AIMessage as _AI
            last = final_msgs[-1]
            if isinstance(last, _AI) and _FALLBACK_MARKER in str(last.content):
                log.warning("[Supervisor] OutputGuardrail blocked — surfacing fallback")
                yield {"type": "token", "content": str(last.content)}
    except Exception as e:
        log.warning(f"[Supervisor] Guardrail state check failed: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("AGENT_PORT", "8080")))