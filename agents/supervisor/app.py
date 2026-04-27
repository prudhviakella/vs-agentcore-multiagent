"""
agents/supervisor/app.py
=========================
Supervisor Agent — AgentCore Runtime Entrypoint.

ARCHITECTURE:
  Cold start (once per container):
    - OpenAI API key loaded from Secrets Manager
    - Pinecone, Postgres, SemanticCache initialised
    - Sub-agent runtime ARNs loaded from SSM

  Per request:
    - token_queue created (asyncio.Queue)
    - A2A tools built with session_id + token_queue in closure
    - Supervisor agent assembled and run via astream_events()
    - Sub-agent tokens arrive on token_queue and are re-streamed to the client

SAFETY (360° coverage):
  Input  — Bedrock Guardrail at platform gateway (input_guardrail.py):
             PROMPT_ATTACK, OffTopicQuery, HATE/VIOLENCE/MISCONDUCT, PII
  Output — OutputGuardrailMiddleware (Bedrock Guardrail + regex):
             PersonalMedicalAdvice denied topic, content filters,
             contextual grounding, check_medical_action_output regex

HITL (4 paths — all preserved):
  Path A — on_tool_end fires for clarify___ask_user_input
  Path B — on_chain_end contains __interrupt__ key
  Path C — Postgres checkpoint has pending interrupt (post-stream state check)
  Path D — GraphInterrupt exception propagates through astream_events
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

_LOG_GROUP = os.environ.get("LOG_GROUP_NAME", "/agentcore/vs-agentcore-ma/supervisor-agent")

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
try:
    _cw = watchtower.CloudWatchLogHandler(
        log_group_name   = _LOG_GROUP,
        boto3_client     = boto3.client("logs", region_name=os.environ.get("AWS_REGION", "us-east-1")),
        create_log_group = True,
    )
    _cw.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logging.getLogger().addHandler(_cw)
except Exception:
    pass  # local dev — no CloudWatch credentials

log = logging.getLogger(__name__)

app = BedrockAgentCoreApp()

_cold_start_objects = None

# Strip episodic memory tag that EpisodicMemoryMiddleware appends to prompts
_EPISODIC_PATTERN = re.compile(r'\s*\nEPISODIC:\s*(YES|NO)[\d.\s]*$', re.IGNORECASE)
_TAIL_SIZE = 60  # chars held back to strip episodic tag before flushing


def _extract_interrupt_args(iv: dict) -> dict:
    """Unwrap interrupt value from AgentCore or LangGraph interrupt format."""
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
    Main handler — invoked by Platform API via invoke_agent_runtime().

    Payload keys:
        message     — user query (empty string on HITL resume)
        thread_id   — user session identifier
        domain      — "pharma"
        resume      — True if this is a HITL resume call
        user_answer — selected HITL option (resume only)

    Yields SSE event dicts:
        token, tool_start, tool_end, chart, interrupt, done, error
    """
    t0          = time.perf_counter()
    message     = payload.get("message",     "")
    thread_id   = payload.get("thread_id",   "") or getattr(context, "session_id", "")
    domain      = payload.get("domain",      "pharma")
    is_resume   = payload.get("resume",      False)
    user_answer = payload.get("user_answer", "")

    log.info(f"[Supervisor] {'resume' if is_resume else 'chat'}  thread={thread_id[:8] if thread_id else 'n/a'}")

    try:
        cold        = await _ensure_cold_start()
        token_queue = asyncio.Queue()

        config = {
            "configurable": {
                "thread_id":  thread_id,
                "user_id":    thread_id,
                "session_id": thread_id,
                "domain":     domain,
            },
            "recursion_limit": 100,  # 20 tool calls × ~5 graph steps each
        }
        agent_context = {"user_id": thread_id, "session_id": thread_id, "domain": domain}

        agent = await build_supervisor_agent(
            session_id  = thread_id,
            domain      = domain,
            token_queue = token_queue,
            **cold,
        )

        # ── HITL resume: repair any dangling tool calls left from the interrupt ──
        if is_resume:
            try:
                from langchain_core.messages import ToolMessage
                state    = await agent.aget_state(config)
                messages = state.values.get("messages", [])
                result_ids = {getattr(m, "tool_call_id", None) for m in messages if hasattr(m, "tool_call_id")}
                dangling   = [
                    tc for msg in messages
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

            input_data = {"messages": [{"role": "user", "content": f"[HITL Answer]: {user_answer}. Now search and answer."}]}
        else:
            input_data = {"messages": [{"role": "user", "content": message}]}

        async for event in _stream_supervisor(agent, input_data, config, agent_context, token_queue):
            yield event

    except HITLInterrupt as hitl:
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
        elapsed = round((time.perf_counter() - t0) * 1_000, 2)
        log.info(f"[Supervisor] Done  latency_ms={elapsed}")
        yield {"type": "done", "latency_ms": elapsed}


async def _stream_supervisor(agent, input_data, config, agent_context, token_queue):
    """
    Stream astream_events() and interleave with token_queue sub-agent events.

    Two event sources run concurrently:
      astream_events() — Supervisor LLM turns and chain events
      token_queue      — Sub-agent tokens and events (put there by A2A tools)

    HITL interrupts surface through four paths (A/B/C/D) depending on how
    LangGraph raises or records the interrupt for clarify___ask_user_input.

    OutputGuardrailMiddleware runs in after_agent (post-stream). If it blocks
    the answer, the final state check at the end surfaces the fallback message.
    """
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

    async def _drain_queue():
        """Yield all events currently sitting on the token_queue."""
        while True:
            try:
                yield token_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    try:
        async for event in agent.astream_events(
            input_data,
            config  = config,
            version = "v2",
            context = agent_context,
        ):
            # Drain sub-agent events that arrived while LangGraph was processing
            async for item in _drain_queue():
                yield item

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
                        yield {"type": "token", "content": safe}

            elif kind == "on_tool_start" and "ask_user_input" in name:
                # Capture HITL args — used by Path A and Path D
                raw        = data.get("input", {})
                hitl_input = raw.get("arguments", raw)
                log.info("[Supervisor] clarify___ask_user_input starting")

            elif kind == "on_tool_end" and "ask_user_input" in name:
                # Path A — tool completed, surface interrupt to UI
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
                return

            elif kind == "on_chain_end":
                # Path B — interrupt embedded in chain output dict
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
        raise  # re-raised and caught in handler()

    except Exception as exc:
        exc_type = type(exc).__name__
        if "Interrupt" in exc_type or "GraphInterrupt" in exc_type:
            # Path D — interrupt raised as exception by LangGraph
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
            log.exception(f"[Supervisor] Stream error: {exc}")
            if _tail_buffer.strip():
                clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
                if clean:
                    yield {"type": "token", "content": clean}
                _tail_buffer = ""
            yield {"type": "error", "message": str(exc)}
        return

    # ── Flush tail buffer ─────────────────────────────────────────────────
    if _tail_buffer:
        clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
        if clean:
            yield {"type": "token", "content": clean}

    # ── Drain any remaining sub-agent events ──────────────────────────────
    async for item in _drain_queue():
        yield item

    # ── Path C: check Postgres checkpoint for pending interrupt ───────────
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

    # ── OutputGuardrailMiddleware post-stream check ───────────────────────
    # OutputGuardrailMiddleware.after_agent runs after the graph completes.
    # If Bedrock blocked the answer it replaces the last message with a
    # fallback. We surface that fallback here so the user sees the reason.
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