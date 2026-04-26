"""
agents/hitl/app.py
===================
HITL Agent — AgentCore Runtime Entrypoint.

FIX (Issue #1 — Postgres connection per invocation):
  BEFORE: _ensure_agent() called build_hitl_agent() which called
          psycopg.AsyncConnection.connect() on every cold start check.
          Because _agent was cached globally, this only ran once —
          BUT the checkpointer connection was built inside agent.py
          on every build_hitl_agent() call, so if _agent was ever
          invalidated or rebuilt, a new connection would be opened.

  AFTER:  _cold_start holds {"checkpointer": AsyncPostgresSaver} built
          once via build_hitl_cold_start(). On every request, the same
          checkpointer is passed to build_hitl_agent(). This guarantees
          exactly one Postgres connection for the lifetime of the container.

IDENTICAL to original app.py in all other respects:
  - Tool filter : HITL_TOOLS
  - HITL paths  : A/B/C/D preserved
  - EPISODIC strip
  - Span emission for distributed tracing
"""

import json
import logging
import os
import re
import time

import boto3
from bedrock_agentcore import BedrockAgentCoreApp, BedrockAgentCoreContext

from agents.hitl.agent import build_hitl_agent, build_hitl_cold_start
from core.mcp_client import get_mcp_tools

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

app = BedrockAgentCoreApp()

# ── Cold start globals ────────────────────────────────────────────────────────
# FIX: split into _cold_start (built once) and _agent (assembled per call).
# _cold_start["checkpointer"] is a single long-lived Postgres connection.
# _agent is rebuilt when tools change (MCP reconnect after cold restart).
_cold_start: dict | None = None
_agent = None

HITL_TOOLS = {
    "tool-hitl___ask_user_input",
}

_EPISODIC_PATTERN = re.compile(r'\s*\nEPISODIC:\s*(YES|NO)[\d.\s]*$', re.IGNORECASE)
_TAIL_SIZE = 60


def _extract_hitl_input(raw_input: dict) -> dict:
    """Extract HITL args — same as single agent app.py."""
    if not raw_input:
        return {}
    if "arguments" in raw_input:
        return raw_input["arguments"]
    if "question" in raw_input or "options" in raw_input:
        return raw_input
    return raw_input


def _extract_interrupt_args(iv: dict) -> dict:
    """Extract interrupt args from LangGraph checkpoint — same as single agent."""
    if not iv or not isinstance(iv, dict):
        return {}
    action_requests = iv.get("action_requests", [])
    if action_requests:
        return action_requests[0].get("args", {})
    return iv


async def _ensure_agent():
    """
    Build cold start objects once, then assemble agent per call.

    Cold start (once per container lifetime):
      - OpenAI API key loaded from Secrets Manager
      - Postgres connection + AsyncPostgresSaver built via build_hitl_cold_start()

    Per request:
      - MCP tools filtered to HITL_TOOLS
      - Agent assembled with pre-built checkpointer

    WHY _agent is also cached globally:
      get_mcp_tools() is an async I/O call. Caching _agent avoids paying
      this cost on every request. If tools change (MCP reconnect), the
      container will restart naturally via AgentCore's health check.
    """
    global _cold_start, _agent

    # ── One-time cold start ───────────────────────────────────────────────
    if _cold_start is None:
        log.info("[HITL] Cold start — building shared objects")

        if not os.environ.get("OPENAI_API_KEY"):
            prefix = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
            sm     = boto3.client("secretsmanager", region_name="us-east-1")
            secret = json.loads(sm.get_secret_value(SecretId=f"{prefix}/openai")["SecretString"])
            os.environ["OPENAI_API_KEY"] = secret.get("api_key", "")
            log.info("[HITL] OpenAI key loaded")

        _cold_start = await build_hitl_cold_start()
        log.info("[HITL] Cold start complete")

    # ── Agent assembly (cached after first build) ─────────────────────────
    if _agent is None:
        all_tools = await get_mcp_tools()
        tools     = [t for t in all_tools if t.name in HITL_TOOLS]
        log.info(f"[HITL] Tools filtered: {[t.name for t in tools]}")

        _agent = await build_hitl_agent(
            tools        = tools,
            checkpointer = _cold_start["checkpointer"],   # FIX: passed in, not built here
        )
        log.info("[HITL] Agent ready")

    return _agent


@app.entrypoint
async def handler(payload: dict, context: BedrockAgentCoreContext):
    """
    Invoked by Supervisor via invoke_agent_runtime().

    payload (new call):
        message    — broad query needing clarification
        session_id — user thread_id
        domain     — "pharma"
        resume     — False

    payload (resume call after user picks option):
        message     — "[HITL Answer]: NCI-MATCH. Now search and answer."
        session_id  — same thread_id
        domain      — "pharma"
        resume      — True
        user_answer — "NCI-MATCH study"

    yields:
        token, tool_start, tool_end — normal streaming
        interrupt                   — HITL card data (Supervisor catches this)
        span                        — observability span for distributed tracing
        done                        — with answer or empty if interrupted
        error                       — on failure
    """
    t0          = time.perf_counter()
    message     = payload.get("message",     "")
    session_id  = payload.get("session_id",  "") or getattr(context, "session_id", "")
    domain      = payload.get("domain",      "pharma")
    is_resume   = payload.get("resume",      False)
    user_answer = payload.get("user_answer", "")

    log.info(
        f"[HITL] {'resume' if is_resume else 'new'}"
        f"  session={session_id[:8] if session_id else 'n/a'}"
        f"  query={message[:60]}..."
    )

    full_answer     = ""
    _tail_buffer    = ""
    hitl_input      = {}
    interrupt_fired = False

    def _flush_safe():
        nonlocal _tail_buffer
        if len(_tail_buffer) > _TAIL_SIZE:
            safe         = _tail_buffer[:-_TAIL_SIZE]
            _tail_buffer = _tail_buffer[-_TAIL_SIZE:]
            return safe
        return ""

    try:
        agent     = await _ensure_agent()
        thread_id = session_id or f"hitl-{time.time()}"

        config = {
            "configurable": {
                "thread_id":  thread_id,
                "user_id":    thread_id,
                "session_id": thread_id,
                "domain":     domain,
            },
            "recursion_limit": 20,
        }
        agent_context = {"user_id": thread_id, "session_id": thread_id, "domain": domain}

        # ── Resume: repair dangling tool calls ────────────────────────────
        if is_resume:
            try:
                from langchain_core.messages import ToolMessage
                state    = await agent.aget_state(config)
                messages = state.values.get("messages", [])

                result_ids = {
                    getattr(m, "tool_call_id", None)
                    for m in messages if hasattr(m, "tool_call_id")
                }
                dangling = [
                    tc
                    for msg in messages
                    for tc in getattr(msg, "tool_calls", [])
                    if tc.get("id") not in result_ids
                ]

                if dangling:
                    log.info(f"[HITL] Repairing {len(dangling)} dangling tool call(s)")
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
                log.warning(f"[HITL] State repair failed (continuing): {e}")

            log.info(f"[HITL] Resume  answer='{user_answer[:60]}'")
            input_data = {
                "messages": [{
                    "role":    "user",
                    "content": f"[HITL Answer]: {user_answer}. Now search and answer.",
                }]
            }
        else:
            input_data = {"messages": [{"role": "user", "content": message}]}

        # ── Stream events — HITL paths A / B / C / D ──────────────────────
        try:
            async for event in agent.astream_events(
                input_data,
                config  = config,
                version = "v2",
                context = agent_context,
            ):
                kind = event.get("event", "")
                name = event.get("name",  "")
                data = event.get("data",  {})

                if kind == "on_chat_model_stream":
                    chunk   = data.get("chunk", {})
                    content = getattr(chunk, "content", "")
                    if isinstance(content, str) and content:
                        _tail_buffer += content
                        safe = _flush_safe()
                        if safe:
                            full_answer += safe
                            yield {"type": "token", "content": safe}

                elif kind == "on_tool_start":
                    is_hitl_tool = "ask_user_input" in name or "tool-hitl" in name
                    if is_hitl_tool:
                        raw        = data.get("input", {})
                        hitl_input = _extract_hitl_input(raw)
                        log.info(f"[HITL] Tool starting  input={hitl_input}")
                    else:
                        yield {"type": "tool_start", "name": name}

                elif kind == "on_tool_end":
                    is_hitl_tool = "ask_user_input" in name or "tool-hitl" in name
                    if is_hitl_tool:
                        # Path A — tool ran to completion, fire interrupt
                        log.info("[HITL] Path A — tool ended")
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
                        break
                    else:
                        yield {"type": "tool_end", "name": name}

                elif kind == "on_chain_end":
                    # Path B — __interrupt__ in chain end output
                    output = data.get("output", {})
                    if isinstance(output, dict) and "__interrupt__" in output:
                        interrupts = output["__interrupt__"]
                        if interrupts:
                            iv   = interrupts[0]
                            val  = iv.value if hasattr(iv, "value") else iv
                            args = _extract_interrupt_args(val)
                            log.info("[HITL] Path B — __interrupt__ in chain")
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

        except Exception as exc:
            exc_type = type(exc).__name__
            if "Interrupt" in exc_type or "GraphInterrupt" in exc_type:
                # Path D — GraphInterrupt exception
                log.info("[HITL] Path D — GraphInterrupt exception")
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
                log.exception(f"[HITL] Stream error: {exc}")
                if _tail_buffer.strip():
                    clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
                    if clean:
                        yield {"type": "token", "content": clean}
                    _tail_buffer = ""
                yield {"type": "error", "message": str(exc)}
            return

        # Flush tail buffer
        if _tail_buffer:
            clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
            if clean:
                full_answer += clean
                yield {"type": "token", "content": clean}
            _tail_buffer = ""

        # Path C — check Postgres checkpoint for pending interrupt
        if not interrupt_fired:
            try:
                state = await agent.aget_state(config)
                tasks = state.tasks if hasattr(state, "tasks") else []
                for task in tasks:
                    task_interrupts = getattr(task, "interrupts", [])
                    if task_interrupts:
                        iv   = task_interrupts[0]
                        val  = iv.value if hasattr(iv, "value") else iv
                        args = _extract_interrupt_args(val)
                        log.info("[HITL] Path C — interrupt in state checkpoint")
                        yield {
                            "type":           "interrupt",
                            "question":       args.get("question", "Please clarify:"),
                            "options":        args.get("options", []),
                            "allow_freetext": args.get("allow_freetext", True),
                        }
                        return
            except Exception as e:
                log.warning(f"[HITL] State interrupt check failed: {e}")

    except Exception as exc:
        log.exception(f"[HITL] Handler error: {exc}")
        yield {"type": "error", "message": str(exc)}

    finally:
        elapsed = round((time.perf_counter() - t0) * 1_000, 2)
        log.info(f"[HITL] Done  latency_ms={elapsed}")
        # Emit observability span for Supervisor distributed tracing
        yield {
            "type": "span",
            "data": {
                "agent":      "hitl",
                "elapsed_ms": elapsed,
            }
        }
        yield {"type": "done", "latency_ms": elapsed, "answer": full_answer}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8004)