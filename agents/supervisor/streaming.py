"""
supervisor/streaming.py
========================
LangGraph event loop driver + SSE event emitter.

THIS MODULE DOES THREE THINGS
------------------------------
1. repair_hitl_state()  — fixes broken message history before HITL resume
2. stream_supervisor()  — drives the LangGraph ReAct loop
3. _drain_queue()       — interleaves A2A tool events into the SSE stream

WHY THIS IS A SEPARATE MODULE
------------------------------
app.py should only know WHAT to do (call these functions).
This module knows HOW to do it (drive LangGraph, handle events, flush buffers).
This separation means you can test streaming logic without touching the
AgentCore entrypoint.

THE TWO-STREAM PROBLEM (important to understand)
-------------------------------------------------
Two things produce events at the same time:

  Stream 1: LangGraph astream_events()
    Yields on every graph node execution — LLM tokens, tool start/end etc.
    Runs in the async event loop. We iterate it with "async for".

  Stream 2: token_queue (asyncio.Queue)
    A2A tools call sub-agents via httpx async streaming. When chart_agent
    produces a Chart.js config, parse_sse_stream() puts it directly into
    token_queue from the async context — no threads needed.

  Problem: if we only yield LangGraph events, chart events arrive AFTER
  answer tokens → wrong order in the browser (answer first, chart later).

  Solution: _drain_queue() is called on EVERY LangGraph event so queued
  chart events are emitted at exactly the right moment (after tool_end,
  before the next on_chat_model_stream).

EPISODIC TAIL STRIPPING
------------------------
gpt-5.5 appends a memory tag to every response:
  "...final answer text.\nEPISODIC: YES 0.95"

This tag is for internal episodic memory middleware — the UI must never
see it. We hold back the last 60 chars of every token stream and strip
the pattern at flush time before yielding.
"""
import asyncio
import logging
import re

from agents.supervisor.a2a_tools import HITLInterrupt

log = logging.getLogger(__name__)

# Strips the episodic memory tag that gpt-5.5 appends to every response.
# Pattern: optional whitespace + newline + "EPISODIC: YES/NO" + optional score
_EPISODIC_PATTERN = re.compile(r'\s*\nEPISODIC:\s*(YES|NO)[\d.\s]*$', re.IGNORECASE)

# How many chars to hold back at the end of the token stream.
# 60 chars safely contains the longest possible EPISODIC suffix.
_TAIL_SIZE = 60


# ── HITL State Repair ──────────────────────────────────────────────────────

async def repair_hitl_state(agent, config: dict, user_answer: str) -> None:
    """
    Fix broken message history before resuming after a HITL interrupt.

    WHY THIS IS NEEDED
    ------------------
    When the supervisor calls ask_user_input and gets interrupted, LangGraph
    saves the checkpoint to Postgres. The saved state has:

      AIMessage(tool_calls=[{id: "call_abc", name: "clarify___ask_user_input"}])
      ← NO ToolMessage for call_abc — the tool never completed

    OpenAI's API strictly requires that every tool_call in an AIMessage has
    a matching ToolMessage before the next HumanMessage. Without one, the
    next API call fails with:
      "Error 400: tool_calls must be followed by tool messages"

    This function inserts a synthetic ToolMessage that "closes" the open
    tool call with the user's answer, making the history valid again.

    HOW IT WORKS
    ------------
    1. Load the checkpoint from Postgres (keyed by thread_id)
    2. Find all tool_call IDs that have no matching ToolMessage (dangling)
    3. Insert a synthetic ToolMessage for each dangling call
    4. LangGraph saves the repaired state — next invocation succeeds

    Parameters
    ----------
    agent       : compiled LangGraph graph
    config      : LangGraph config dict (contains thread_id for checkpoint lookup)
    user_answer : str — the user's HITL selection, injected into the ToolMessage
    """
    try:
        from langchain_core.messages import ToolMessage

        # Load checkpoint from Postgres
        state    = await agent.aget_state(config)
        messages = state.values.get("messages", [])

        # Collect IDs of tool calls that already have a ToolMessage response
        result_ids = {
            getattr(m, "tool_call_id", None)
            for m in messages
            if hasattr(m, "tool_call_id")
        }

        # Find tool calls with no matching ToolMessage (these are "dangling")
        dangling = [
            tc
            for msg in messages
            for tc in getattr(msg, "tool_calls", [])
            if tc.get("id") not in result_ids
        ]

        if dangling:
            log.info(f"[Supervisor] Repairing {len(dangling)} dangling tool call(s)")
            # Insert a synthetic ToolMessage for each dangling call.
            # Content includes the user's answer so gpt-5.5 sees it in
            # context when deciding what to do next.
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
        # State repair is best-effort — log and continue.
        # Without repair the next API call may fail with 400,
        # but that's handled by the error path in stream_supervisor.
        log.warning(f"[Supervisor] State repair failed (continuing): {e}")


# ── Core Streaming Function ────────────────────────────────────────────────

async def stream_supervisor(
    agent,
    input_data:   dict,
    config:       dict,
    agent_context: dict,
    token_queue:  asyncio.Queue,
):
    """
    Drive the LangGraph ReAct loop and yield typed SSE events.

    This is an async generator — it yields dicts that become SSE events:
      {"type": "token",     "content": str}    — LLM token (thinking or answer)
      {"type": "tool_start","name":    str}    — sub-agent starting
      {"type": "tool_end",  "name":    str}    — sub-agent done
      {"type": "chart",     "config":  dict}   — chart event from chart_agent
      {"type": "interrupt", ...}               — HITL: pause and ask user
      {"type": "error",     "message": str}    — something went wrong

    THE REACT LOOP
    --------------
    LangGraph drives the supervisor in a loop:
      1. supervisor node: gpt-5.5 generates tokens → may include tool_calls
      2. tools node: executes the tool (calls the sub-agent)
      3. back to supervisor: gpt-5.5 sees the tool result, decides next step
      4. repeat until gpt-5.5 generates no tool_calls → stream ends

    astream_events() yields events for each step. We process:
      on_chat_model_stream → LLM tokens (buffered for EPISODIC stripping)
      on_tool_start        → HITL arg capture
      on_tool_end          → HITL interrupt emission

    Parameters
    ----------
    agent         : compiled LangGraph graph (from build_supervisor_agent)
    input_data    : dict — {"messages": [...]} to feed into the graph
    config        : dict — LangGraph config with thread_id, recursion_limit etc.
    agent_context : dict — session metadata passed into tool closures
    token_queue   : asyncio.Queue — receives chart/interrupt events from A2A threads
    """
    _tail_buffer = ""   # accumulates LLM tokens; last _TAIL_SIZE chars held back
    hitl_input   = {}   # HITL args captured at on_tool_start, emitted at on_tool_end

    def _flush_safe() -> str:
        """
        Return the safe-to-emit portion of the tail buffer.

        Holds back the last _TAIL_SIZE chars so we can strip the EPISODIC
        suffix at the end of the stream without emitting it to the UI.

        Example:
          buffer = "Phase 3 trials lead.\nEPISODIC: YES 0.95"
          _flush_safe() → emits "Phase 3 trials lead."
                        → holds "\nEPISODIC: YES 0.95" until stream end
          _flush_tail() → strips pattern → emits nothing extra
        """
        nonlocal _tail_buffer
        if len(_tail_buffer) > _TAIL_SIZE:
            safe         = _tail_buffer[:-_TAIL_SIZE]  # safe portion
            _tail_buffer = _tail_buffer[-_TAIL_SIZE:]  # hold back tail
            return safe
        return ""  # not enough chars yet — hold everything

    def _flush_tail() -> str:
        """Flush and clean the entire tail buffer at end of stream."""
        nonlocal _tail_buffer
        clean        = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
        _tail_buffer = ""
        return clean

    async def _drain_queue():
        """
        Non-blocking drain of token_queue.

        IMPORTANT: uses get_nowait() not await get().
          - get_nowait() returns immediately if queue is empty (raises QueueEmpty)
          - await get() would BLOCK the event loop waiting for future items
          - Blocking here would freeze ALL concurrent SSE streams

        Called on EVERY LangGraph event — this is the mechanism that
        guarantees chart events arrive BEFORE the answer tokens that follow.

        Timeline example:
          on_tool_end (chart_agent)    ← drain → finds chart event → yield it
          on_chat_model_stream (token) ← drain → empty → yield token

          Without draining on every event:
            chart sits in queue, answer tokens stream past it,
            chart arrives LAST in the browser → wrong order
        """
        while True:
            try:
                yield token_queue.get_nowait()
            except asyncio.QueueEmpty:
                break  # nothing left — continue with LangGraph event

    try:
        # astream_events() drives the full ReAct loop asynchronously.
        # version="v2" is required for LangGraph 1.0+ event schema.
        async for event in agent.astream_events(
            input_data,
            config  = config,
            version = "v2",
            context = agent_context,
        ):
            # ── Drain A2A queue FIRST on every event ──────────────────────
            # A2A tools (running in threads) may have put chart/interrupt
            # events into token_queue since the last LangGraph event.
            # Drain before processing so events appear in the correct order.
            async for item in _drain_queue():
                yield item

            kind = event.get("event", "")
            name = event.get("name",  "")
            data = event.get("data",  {})

            # ── LLM token stream ───────────────────────────────────────────
            # Fires for every token gpt-5.5 generates — including <thinking>
            # blocks. We buffer all tokens and flush safely to strip EPISODIC.
            # The UI state machine in ui/app.py separates thinking vs answer.
            if kind == "on_chat_model_stream":
                content = getattr(data.get("chunk", {}), "content", "")
                if isinstance(content, str) and content:
                    _tail_buffer += content
                    safe = _flush_safe()
                    if safe:
                        yield {"type": "token", "content": safe}

            # ── HITL tool starting — capture args before tool runs ─────────
            # We capture the question/options here at on_tool_start because
            # on_tool_end fires AFTER the tool has already interrupted.
            # By that point the args are no longer available in the event data.
            elif kind == "on_tool_start" and "ask_user_input" in name:
                raw        = data.get("input", {})
                hitl_input = raw.get("arguments", raw)
                log.info("[Supervisor] HITL tool starting — args captured")

            # ── HITL interrupt: emit and stop streaming ────────────────────
            # LangGraph fires on_tool_end after the interrupt is checkpointed.
            # We flush any remaining answer tokens, emit the interrupt event,
            # then return — the UI takes over with the HITL card.
            elif kind == "on_tool_end" and "ask_user_input" in name:
                clean = _flush_tail()
                if clean:
                    yield {"type": "token", "content": clean}
                yield {
                    "type":           "interrupt",
                    "question":       hitl_input.get("question", "Please clarify:"),
                    "options":        hitl_input.get("options", []),
                    "allow_freetext": hitl_input.get("allow_freetext", True),
                }
                return  # stop — UI handles next step via POST /resume

    except HITLInterrupt:
        # HITLInterrupt propagates up from ask_user_func in some LangGraph
        # versions before on_tool_end fires. Re-raise to handler() which
        # catches it and emits the interrupt event from there.
        raise

    except Exception as exc:
        exc_type = type(exc).__name__
        if "Interrupt" in exc_type or "GraphInterrupt" in exc_type:
            # LangGraph raises GraphInterrupt in some versions instead of
            # surfacing it via on_tool_end. Treat identically to Path A.
            clean = _flush_tail()
            if clean:
                yield {"type": "token", "content": clean}
            yield {
                "type":           "interrupt",
                "question":       hitl_input.get("question", "Please clarify:"),
                "options":        hitl_input.get("options", []),
                "allow_freetext": hitl_input.get("allow_freetext", True),
            }
        else:
            # Genuine error — log full traceback, surface message to UI
            log.exception(f"[Supervisor] Stream error: {exc}")
            clean = _flush_tail()
            if clean:
                yield {"type": "token", "content": clean}
            yield {"type": "error", "message": str(exc)}
        return

    # ── End of stream ──────────────────────────────────────────────────────
    # Flush tail buffer (strips EPISODIC suffix) and drain any remaining
    # A2A events that arrived after the last LangGraph event.
    clean = _flush_tail()
    if clean:
        yield {"type": "token", "content": clean}

    async for item in _drain_queue():
        yield item