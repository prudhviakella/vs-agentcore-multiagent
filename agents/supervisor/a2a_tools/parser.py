"""
a2a_tools/parser.py
====================
SSE event parser for sub-agent responses.

WHY A DEDICATED PARSER?
------------------------
Every sub-agent (knowledge, research, chart) speaks the same SSE protocol.
One parser handles all of them — no duplicated parsing logic per agent.

THE SSE PROTOCOL BETWEEN AGENTS
---------------------------------
When the supervisor calls a sub-agent (e.g. knowledge_agent), the sub-agent
streams its response as Server-Sent Events (SSE). Each line looks like:
  data: {"type": "token", "content": "The trial enrolled..."}
  data: {"type": "chart", "config": {...}}
  data: {"type": "done"}

This parser reads those lines and routes each event type correctly.

EVENT ROUTING
-------------
  token     → two things happen:
                1. Accumulated in full_answer (returned to supervisor as tool result)
                2. Forwarded to token_queue (UI sees sub-agent streaming in real time)

  chart     → put into token_queue DIRECTLY
                _drain_queue() in streaming.py picks it up at on_tool_end,
                guaranteeing the chart renders BEFORE the supervisor's answer tokens.
                This is the fix for "chart appearing after the answer" bug.

  interrupt → re-raise as HITLInterrupt
                Propagates up through invoke.py → tools.py → LangGraph ToolNode
                → streaming.py exception handler → UI shows HITL card.

  error     → re-raise as RuntimeError
                LangGraph catches this as a tool error and adds a ToolMessage
                with the error text. gpt-5.5 sees "tool failed" and can retry
                with a different query or report the failure to the user.

  done      → stream complete, return full_answer

"""
import asyncio
import json
import logging
from typing import AsyncIterator

log = logging.getLogger(__name__)


async def parse_sse_stream(
    agent_name:  str,
    line_iter:   AsyncIterator,
    token_queue: asyncio.Queue,
) -> str:
    """
    Parse SSE lines from a sub-agent response into typed events.

    Parameters
    ----------
    agent_name  : str            — e.g. "knowledge". Used in error messages.
    line_iter   : AsyncIterator  — yields raw SSE text lines (from httpx aiter_lines())
    token_queue : asyncio.Queue  — shared queue for chart/interrupt events.
                                   These get picked up by _drain_queue() in
                                   streaming.py and forwarded to the UI.

    Returns
    -------
    str — the complete answer from the sub-agent.
    This string becomes the ToolMessage content that gpt-5.5 reads
    in the next ReAct turn to decide what to do next.
    """
    # Import here to avoid circular imports
    # (hitl imports from this package, this imports from hitl)
    from agents.supervisor.a2a_tools.hitl import HITLInterrupt

    full_answer = ""  # accumulated answer, built token by token

    async for line in line_iter:
        line = line.strip()
        if not line:
            continue  # blank lines are valid SSE separators — skip them

        # Strip the "data: " SSE prefix before JSON parsing
        if line.startswith("data: "):
            line = line[6:]

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Malformed line (e.g. partial chunk, keepalive comment) — skip
            continue

        etype = event.get("type", "")

        if etype == "token":
            # Sub-agent is streaming its answer token by token.
            # We accumulate the full answer for the supervisor's tool result,
            # AND forward each token to token_queue so the UI can show
            # the sub-agent's progress in real time.
            c = event.get("content", "")
            if c:
                full_answer += c
                await token_queue.put({"type": "token", "content": c})

        elif etype == "chart":
            # Chart.js config from chart_agent.
            # Goes directly into token_queue — NOT into full_answer.
            # Reason: the supervisor's LLM doesn't need to "read" the chart
            # config; it just needs to know the chart was generated.
            # The UI renders the chart from the config in token_queue.
            #
            # ORDER GUARANTEE: _drain_queue() in streaming.py drains this
            # at on_tool_end, BEFORE the next on_chat_model_stream tokens.
            # This is why charts appear before the answer in the browser.
            await token_queue.put({
                "type":   "chart",
                "config": event.get("config", {}),
            })

        elif etype == "interrupt":
            # Sub-agent triggered a HITL interrupt (rare — usually the supervisor does this).
            # Re-raise so it propagates to the supervisor's exception handlers.
            raise HITLInterrupt(
                question       = event.get("question", "Please clarify:"),
                options        = event.get("options", []),
                allow_freetext = event.get("allow_freetext", True),
            )

        elif etype == "error":
            # Sub-agent reported an error (e.g. Neo4j timeout, Pinecone failure).
            # Raise RuntimeError — LangGraph's ToolNode catches it and creates a
            # ToolMessage with the error text. gpt-5.5 then sees "tool failed"
            # and can decide to retry or tell the user what went wrong.
            raise RuntimeError(f"Sub-agent '{agent_name}': {event.get('message', 'Unknown error')}")

        elif etype == "done":
            # Sub-agent stream complete.
            # full_answer is already built from token events above.
            pass

    return full_answer