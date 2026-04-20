"""
agents/supervisor/app.py
=========================
Supervisor Agent — AgentCore Runtime Entrypoint.

IDENTICAL to vs-agentcore-platform-aws/agent/app.py with these differences:
  1. Cold start builds heavy objects (checkpointer, store, cache, safety_llm)
     once via build_supervisor_cold_start() — stored in module-level globals.
  2. Per request: builds A2A tools with session_id + token_queue in closure,
     then calls build_supervisor_agent() to assemble the agent.
  3. token_queue created per request — Supervisor reads from it and yields
     events to Platform API (re-streaming sub-agent tokens).
  4. HITLInterrupt caught here — yields interrupt SSE same as single agent.
  5. HITL paths A/B/C/D preserved — Supervisor intercepts if LLM calls
     hitl_agent tool and it returns an interrupt.

WHY token_queue INSTEAD OF DIRECT YIELDING IN A2A TOOLS?
  @tool functions are called synchronously by LangChain's tool executor.
  They can't yield directly to the handler() generator.
  Solution: @tool puts events on token_queue, handler() reads and yields.
  This decouples the streaming from the tool execution.

COLD START vs PER-REQUEST:
  Cold start (once):
    - OpenAI API key loaded
    - Pinecone connected (PineconeStore + SemanticCache)
    - Postgres connected (AsyncPostgresSaver + checkpointer.setup())
    - Safety LLM (gpt-4o-mini) initialised
    - Sub-agent runtime ARNs loaded from SSM (_get_runtime_arns())

  Per request:
    - token_queue created (asyncio.Queue)
    - A2A tools built with session_id + token_queue in closure
    - Supervisor agent assembled (create_agent with tools + middleware)
    - astream_events() runs — LLM calls A2A tools as needed
    - token_queue drained and events yielded to Platform API

AGENT_NAME: "supervisor-agent" set by Terraform
  → SSM: /supervisor-agent/prod/bedrock/prompt_id
         /supervisor-agent/prod/bedrock/prompt_version
"""

import asyncio
import json
import logging
import os
import re
import time

import boto3
from bedrock_agentcore import BedrockAgentCoreApp, BedrockAgentCoreContext

from agents.supervisor.agent import build_supervisor_agent, build_supervisor_cold_start
from agents.supervisor.a2a_tools import HITLInterrupt

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

app = BedrockAgentCoreApp()

# ── Cold start globals (built once, reused per request) ───────────────────────
_cold_start_objects = None   # dict: checkpointer, store, cache, safety_llm

_EPISODIC_PATTERN = re.compile(r'\s*\nEPISODIC:\s*(YES|NO)[\d.\s]*$', re.IGNORECASE)
_TAIL_SIZE = 60


def _extract_interrupt_args(iv: dict) -> dict:
    """Same as single agent app.py."""
    if not iv or not isinstance(iv, dict):
        return {}
    action_requests = iv.get("action_requests", [])
    if action_requests:
        return action_requests[0].get("args", {})
    return iv


async def _ensure_cold_start():
    """Build heavy objects once at cold start, return cached on warm calls."""
    global _cold_start_objects
    if _cold_start_objects is None:
        log.info("[Supervisor] Cold start — building shared objects")

        # Load OpenAI key
        if not os.environ.get("OPENAI_API_KEY"):
            prefix = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
            sm     = boto3.client("secretsmanager", region_name="us-east-1")
            secret = json.loads(sm.get_secret_value(SecretId=f"{prefix}/openai")["SecretString"])
            os.environ["OPENAI_API_KEY"] = secret.get("api_key", "")
            log.info("[Supervisor] OpenAI key loaded")

        _cold_start_objects = await build_supervisor_cold_start()
        log.info("[Supervisor] Cold start complete")

    return _cold_start_objects


@app.entrypoint
async def handler(payload: dict, context: BedrockAgentCoreContext):
    """
    Main handler — called by Platform API via invoke_agent_runtime().

    PAYLOAD (same structure as single agent):
        message    — user query
        thread_id  — user session
        domain     — "pharma"
        resume     — True if HITL resume
        user_answer — selected HITL option (resume only)

    YIELDS (same event types as single agent):
        token, tool_start, tool_end, chart, interrupt, done, error
    """
    t0          = time.perf_counter()
    message     = payload.get("message",     "")
    thread_id   = payload.get("thread_id",   "") or getattr(context, "session_id", "")
    domain      = payload.get("domain",      "pharma")
    is_resume   = payload.get("resume",      False)
    user_answer = payload.get("user_answer", "")

    log.info(
        f"[Supervisor] {'resume' if is_resume else 'chat'}"
        f"  thread={thread_id[:8] if thread_id else 'n/a'}"
    )

    try:
        cold = await _ensure_cold_start()

        # Per-request token queue — A2A tools put events here, we yield them
        token_queue = asyncio.Queue()

        # Per-request config (same as single agent)
        config = {
            "configurable": {
                "thread_id":  thread_id,
                "user_id":    thread_id,
                "session_id": thread_id,
                "domain":     domain,
            },
            "recursion_limit": 50,
        }
        agent_context = {"user_id": thread_id, "session_id": thread_id, "domain": domain}

        # Build agent with A2A tools (per request — session_id + token_queue in closure)
        agent = await build_supervisor_agent(
            session_id  = thread_id,
            domain      = domain,
            token_queue = token_queue,
            **cold,
        )

        # HITL resume — repair dangling tool calls (identical to single agent)
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
                    log.info(f"[Supervisor] Repairing {len(dangling)} dangling tool call(s)")
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

            input_data = {
                "messages": [{
                    "role":    "user",
                    "content": f"[HITL Answer]: {user_answer}. Now search and answer.",
                }]
            }
        else:
            input_data = {"messages": [{"role": "user", "content": message}]}

        # ── Stream events ─────────────────────────────────────────────────────
        async for event_or_queue_item in _stream_supervisor(
            agent, input_data, config, agent_context, token_queue
        ):
            yield event_or_queue_item

    except HITLInterrupt as hitl:
        log.info(f"[Supervisor] HITLInterrupt caught — surfacing to UI")
        yield {
            "type":          "interrupt",
            "question":      hitl.question,
            "options":       hitl.options,
            "allow_freetext": hitl.allow_freetext,
        }
    except Exception as exc:
        log.exception(f"[Supervisor] Handler error: {exc}")
        yield {"type": "error", "message": str(exc)}
    finally:
        elapsed = round((time.perf_counter() - t0) * 1_000, 2)
        log.info(f"[Supervisor] Done  latency_ms={elapsed}")
        yield {"type": "done", "latency_ms": elapsed}


async def _stream_supervisor(agent, input_data, config, agent_context, token_queue):
    """
    Run astream_events() and interleave with token_queue events.

    Two sources of events:
      1. astream_events() — Supervisor LLM tokens + chain events
      2. token_queue      — Sub-agent tokens/tool events (put there by A2A tools)

    We yield from both simultaneously using asyncio.

    HITL paths A/B/C/D preserved from single agent (same code).
    """
    _tail_buffer     = ""
    hitl_input       = {}
    interrupt_fired  = False
    _safety_blocked  = False   # set True when safety_agent returns BLOCKED
    _block_reason    = ""      # reason string from safety_agent
    _answer_complete = False   # set True when safety_passed — suppress Supervisor LLM tokens
    _safety_blocked = False   # set True when safety_agent returns BLOCKED
    _block_reason   = ""      # reason string from safety_agent

    def _flush_safe():
        nonlocal _tail_buffer
        if len(_tail_buffer) > _TAIL_SIZE:
            safe         = _tail_buffer[:-_TAIL_SIZE]
            _tail_buffer = _tail_buffer[-_TAIL_SIZE:]
            return safe
        return ""

    # Drain token_queue while astream_events runs
    async def _drain_queue():
        nonlocal _safety_blocked, _block_reason, _answer_complete
        while True:
            try:
                item = token_queue.get_nowait()
                # Intercept safety verdict events — don't yield to user
                if isinstance(item, dict):
                    if item.get("type") == "safety_blocked":
                        _safety_blocked = True
                        _block_reason   = item.get("reason", "")
                        log.info(f"[Supervisor] Safety BLOCKED intercepted — suppressing LLM tokens")
                        continue
                    elif item.get("type") == "safety_passed":
                        _answer_complete = True
                        log.info("[Supervisor] Safety PASSED — suppressing Supervisor LLM tokens")
                        continue   # don't yield safety_passed to user
                yield item
            except asyncio.QueueEmpty:
                break

    try:
        async for event in agent.astream_events(
            input_data,
            config  = config,
            version = "v2",
            context = agent_context,
        ):
            # Drain any sub-agent events that arrived via token_queue
            async for item in _drain_queue():
                yield item

            # Break out of event loop if safety blocked (prevents LLM retry)
            # For _answer_complete we let the natural loop end — no break needed
            if _safety_blocked:
                break

            kind = event.get("event", "")
            name = event.get("name",  "")
            data = event.get("data",  {})

            if kind == "on_chat_model_stream":
                # Suppress Supervisor LLM tokens if:
                # a) safety BLOCKED — we'll yield the standard refusal
                # b) safety PASSED — answer already streamed by sub-agent, don't repeat
                if _safety_blocked or _answer_complete:
                    continue
                chunk   = data.get("chunk", {})
                content = getattr(chunk, "content", "")
                if isinstance(content, str) and content:
                    _tail_buffer += content
                    safe = _flush_safe()
                    if safe:
                        yield {"type": "token", "content": safe}

            elif kind == "on_tool_start":
                if "ask_user_input" in name:
                    raw        = data.get("input", {})
                    hitl_input = raw.get("arguments", raw)
                    log.info(f"[Supervisor] HITL tool starting")
                # Don't yield tool_start for A2A tools — sub-agent yields its own

            elif kind == "on_tool_end":
                if "ask_user_input" in name:
                    # Path A
                    interrupt_fired = True
                    if _tail_buffer.strip():
                        clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
                        if clean:
                            yield {"type": "token", "content": clean}
                        _tail_buffer = ""
                    yield {
                        "type":          "interrupt",
                        "question":      hitl_input.get("question", "Please clarify:"),
                        "options":       hitl_input.get("options", []),
                        "allow_freetext": hitl_input.get("allow_freetext", True),
                    }
                    return

            elif kind == "on_chain_end":
                # Path B
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
                            "type":          "interrupt",
                            "question":      args.get("question", "Please clarify:"),
                            "options":       args.get("options", []),
                            "allow_freetext": args.get("allow_freetext", True),
                        }
                        return

    except HITLInterrupt:
        # Re-raise — caught in handler()
        raise

    except Exception as exc:
        exc_type = type(exc).__name__
        if "Interrupt" in exc_type or "GraphInterrupt" in exc_type:
            # Path D
            interrupt_fired = True
            if _tail_buffer.strip():
                clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
                if clean:
                    yield {"type": "token", "content": clean}
                _tail_buffer = ""
            yield {
                "type":          "interrupt",
                "question":      hitl_input.get("question", "Please clarify:"),
                "options":       hitl_input.get("options", []),
                "allow_freetext": hitl_input.get("allow_freetext", True),
            }
        else:
            log.exception(f"[Supervisor] Stream error: {exc}")
            if _tail_buffer.strip():
                clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
                if clean:
                    yield {"type": "token", "content": clean}
                _tail_buffer = ""
            yield {"type": "error", "message": str(exc)}
        return

    # Flush tail + drain remaining queue
    if _tail_buffer:
        clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
        if clean:
            yield {"type": "token", "content": clean}

    async for item in _drain_queue():
        yield item

    # If safety blocked — append a warning note to the answer already streamed
    # (we cannot retract tokens already sent to the user in streaming mode)
    if _safety_blocked:
        reason_short = _block_reason.replace("BLOCKED:", "").strip()[:120]
        log.info(f"[Supervisor] Appending safety warning  reason={reason_short}")
        yield {
            "type":    "token",
            "content": (
                f"\n\n⚠️ **Safety Note:** This answer could not be fully verified "
                f"against retrieved sources. Please verify independently via "
                f"ClinicalTrials.gov or consult a qualified professional."
                f"\n*Reason: {reason_short}*"
            )
        }
        return

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
                    log.info("[Supervisor] Path C — interrupt in state checkpoint")
                    yield {
                        "type":          "interrupt",
                        "question":      args.get("question", "Please clarify:"),
                        "options":       args.get("options", []),
                        "allow_freetext": args.get("allow_freetext", True),
                    }
                    return
        except Exception as e:
            log.warning(f"[Supervisor] State interrupt check failed: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)