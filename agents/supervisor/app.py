"""
supervisor/app.py
==================
AgentCore Runtime Entrypoint — the front door of the Supervisor Agent.

WHAT THIS FILE DOES
--------------------
This is the ONLY file AgentCore calls directly. It receives every incoming
request, delegates to the right module, and yields SSE events back.

Think of it like a restaurant host:
  - Greets every customer (receives the request)
  - Seats them at the right table (routes to cold_start, streaming etc.)
  - Does NOT cook the food (no business logic here)

WHAT IT DELEGATES TO
--------------------
  logging_setup.py  → sets up CloudWatch logging (called once at startup)
  cold_start.py     → loads secrets + builds shared objects (once per container)
  agent.py          → builds the LangGraph agent for this request
  streaming.py      → drives LangGraph and emits SSE events
  a2a_tools/        → all sub-agent invocation logic

WHY KEEP THIS FILE THIN
------------------------
When something breaks in production, you want to find the bug quickly.
A thin entrypoint means: if the bug is in streaming logic → look in streaming.py.
If it's in tool invocation → look in a2a_tools/invoke.py.
Clear module boundaries = faster debugging.

HOW AGENTCORE CALLS THIS
-------------------------
AgentCore SDK calls handler() as an async generator for every HTTP request.
The @app.entrypoint decorator registers it. handler() yields dicts which
AgentCore serialises as SSE (Server-Sent Events) back to the platform.
"""

import asyncio
import logging
import os
import time
import uuid

from bedrock_agentcore import BedrockAgentCoreApp, BedrockAgentCoreContext

from agents.supervisor.a2a_tools    import HITLInterrupt
from agents.supervisor.agent        import build_supervisor_agent
from agents.supervisor.cold_start   import ensure_cold_start
from agents.supervisor.logging_setup import setup_logging
from agents.supervisor.streaming    import stream_supervisor, repair_hitl_state

# ── Logging setup ──────────────────────────────────────────────────────────
# Called once when the module is first imported (i.e. container startup).
# All other modules just do: log = logging.getLogger(__name__)
# They automatically inherit the CloudWatch handler configured here.
setup_logging()
log = logging.getLogger(__name__)

# ── AgentCore app ──────────────────────────────────────────────────────────
# BedrockAgentCoreApp is the AgentCore SDK's entry point.
# @app.entrypoint registers handler() so AgentCore knows which function
# to call when a request arrives. Think of it like @app.post() in FastAPI.
app = BedrockAgentCoreApp()


@app.entrypoint
async def handler(payload: dict, context: BedrockAgentCoreContext):
    """
    Main entrypoint — called by AgentCore for every incoming request.

    This is an async generator. Every "yield" sends one SSE event to the
    platform, which forwards it to the UI. The UI renders each event as it
    arrives — this is what creates the live streaming experience.

    REQUEST TYPES
    -------------
    Normal chat  (payload.resume = False):
      User typed a message → supervisor thinks → calls sub-agents → answers

    HITL resume  (payload.resume = True):
      User answered a clarification question → supervisor continues from
      where it paused (loads checkpoint from Postgres via thread_id)

    PARAMETERS FROM PAYLOAD
    -----------------------
    message     : str  — the user's question
    thread_id   : str  — unique session ID; also the Postgres checkpoint key
    domain      : str  — "pharma" (used for tool context and memory routing)
    resume      : bool — True when this is a HITL resume request
    user_answer : str  — the user's HITL selection (only set when resume=True)

    ALWAYS YIELDS "done"
    --------------------
    The finally block ALWAYS yields {"type": "done"} even if an error occurs.
    The platform and UI both rely on this to know the stream has ended.
    Without it, the UI would wait forever for more events.
    """
    t0          = time.perf_counter()
    message     = payload.get("message",     "")
    thread_id   = payload.get("thread_id",   "") or getattr(context, "session_id", "")
    domain      = payload.get("domain",      "pharma")
    is_resume   = payload.get("resume",      False)
    user_answer = payload.get("user_answer", "")
    # Unique per request — used as DynamoDB key for feedback
    # thread_id is session-level (reused across messages); request_id is per-message
    request_id  = str(uuid.uuid4())

    log.info(f"[Supervisor] {'resume' if is_resume else 'chat'}  thread={thread_id[:8]}  req={request_id[:8]}")

    try:
        # ── Step 1: Cold start ─────────────────────────────────────────────
        # First request: loads secrets + builds Postgres connection, Pinecone
        # index, semantic cache, LangGraph graph. Takes ~3-5 seconds.
        # All subsequent requests: returns cached objects instantly.
        cold = await ensure_cold_start()

        # ── Step 2: Create token_queue ─────────────────────────────────────
        # A fresh queue PER REQUEST — critical for isolation.
        # A2A tools (running in threads) push chart/interrupt events here.
        # stream_supervisor() drains it on every LangGraph event.
        # If we reused one queue across requests, events from one user's
        # chart_agent would appear in another user's SSE stream.
        token_queue = asyncio.Queue()

        # ── Step 3: LangGraph config ───────────────────────────────────────
        # thread_id is the Postgres checkpoint key.
        # Every session has its own isolated message history and HITL state.
        # recursion_limit prevents infinite ReAct loops (LangGraph raises
        # GraphRecursionError if the agent exceeds this many tool calls).
        config = {
            "configurable": {
                "thread_id":  thread_id,
                "user_id":    thread_id,
                "session_id": thread_id,
                "domain":     domain,
            },
            "recursion_limit": 100,
        }
        agent_context = {"user_id": thread_id, "session_id": thread_id, "domain": domain}

        # ── Step 4: Build per-request agent ───────────────────────────────
        # Takes the compiled LangGraph graph from cold start and injects
        # token_queue into all A2A tool closures. Cheap — no recompilation.
        # Token_queue injection is why this is per-request: each request
        # needs its own queue reference inside the tool closures.
        agent = await build_supervisor_agent(
            session_id  = thread_id,
            domain      = domain,
            token_queue = token_queue,
            **cold,
        )

        # ── Step 5: Prepare input ──────────────────────────────────────────
        if is_resume:
            # Before resuming, fix the broken message history.
            # When HITL interrupted, the checkpoint has an AIMessage with
            # tool_calls but no matching ToolMessage. OpenAI rejects this.
            # repair_hitl_state() inserts the missing ToolMessage.
            await repair_hitl_state(agent, config, user_answer)
            input_data = {"messages": [{"role": "user", "content": f"[HITL Answer]: {user_answer}. Now search and answer."}]}
        else:
            input_data = {"messages": [{"role": "user", "content": message}]}

        # ── Step 6: Stream ─────────────────────────────────────────────────
        # Delegate entirely to stream_supervisor().
        # Every event it yields gets forwarded to the platform as SSE.
        async for event in stream_supervisor(agent, input_data, config, agent_context, token_queue):
            yield event

    except HITLInterrupt as hitl:
        # HITLInterrupt bubbles up from ask_user_func in some LangGraph
        # versions before stream_supervisor catches it. Handle here as fallback.
        log.info("[Supervisor] HITLInterrupt caught at handler level")
        yield {
            "type":           "interrupt",
            "question":       hitl.question,
            "options":        hitl.options,
            "allow_freetext": hitl.allow_freetext,
        }
    except Exception as exc:
        log.exception(f"[Supervisor] Unhandled error: {exc}")
        yield {"type": "error", "message": str(exc)}
    finally:
        # Always yield "done" — platform and UI wait for this
        elapsed = round((time.perf_counter() - t0) * 1_000, 2)
        log.info(f"[Supervisor] Done  latency_ms={elapsed}  req={request_id[:8]}")
        # run_id = thread_id — matches TracerMiddleware's DynamoDB partition key
        # so feedback UpdateItem hits the same item as the trace PutItem
        yield {"type": "done", "latency_ms": elapsed, "run_id": thread_id}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("AGENT_PORT", "8080")))