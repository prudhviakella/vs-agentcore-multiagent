"""
a2a_tools/parser.py
====================
SSE event parser for sub-agent responses.
Returns (answer, span_data) so callers can capture observability metadata
(e.g. rag_metrics from the research agent's done event).
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
) -> tuple[str, dict]:
    """
    Parse SSE lines from a sub-agent response into typed events.

    Returns
    -------
    (answer, span_data)
        answer    : complete answer text (ToolMessage content for the LLM)
        span_data : observability metadata from the done event
                    (e.g. rag_metrics from research agent)
    """
    from agents.supervisor.a2a_tools.hitl import HITLInterrupt

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

        elif etype == "chart":
            await token_queue.put({
                "type":   "chart",
                "config": event.get("config", {}),
            })

        elif etype == "interrupt":
            raise HITLInterrupt(
                question       = event.get("question", "Please clarify:"),
                options        = event.get("options", []),
                allow_freetext = event.get("allow_freetext", True),
            )

        elif etype == "error":
            raise RuntimeError(f"Sub-agent '{agent_name}': {event.get('message', 'Unknown error')}")

        elif etype == "span":
            # Sub-agent observability metadata (elapsed_ms, agent name, etc.)
            span_data.update(event.get("data", {}))

        elif etype == "done":
            # Capture rag_metrics embedded by research agent in done event
            if "rag_metrics" in event:
                span_data["rag_metrics"] = event["rag_metrics"]
                log.info(f"[A2A] RAG metrics from done  agent={agent_name}  keys={list(event['rag_metrics'].keys())}")

    return full_answer, span_data