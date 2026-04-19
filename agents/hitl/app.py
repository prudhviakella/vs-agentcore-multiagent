"""
agents/hitl/app.py
===================
HITL Agent — AgentCore Runtime Entrypoint.

IDENTICAL pattern to single agent app.py WITH HITL paths A/B/C/D kept.
This is the only sub-agent that can interrupt and needs resume handling.

DIFFERENCES from research/knowledge/safety app.py:
  1. Tool filter   : clarify___ask_user_input only
  2. HITL paths    : A/B/C/D preserved from single agent app.py
  3. EPISODIC strip: kept (precaution)
  4. AGENT_NAME    : "hitl-agent" set by Terraform
  5. Port          : 8004 (local dev)

RESPONSIBILITY:
  When Supervisor detects a broad/vague query it calls hitl_agent @tool.
  HITL Agent:
    1. Searches for candidate trials (using clarify tool internally)
    2. Generates a clarification card with 5 real trial names as options
    3. Fires NodeInterrupt via HumanInTheLoopMiddleware
    4. Yields interrupt event back to Supervisor
    5. Supervisor propagates interrupt → Platform API → UI (HITL card shown)
    6. User selects option → Platform API calls POST /resume
    7. Supervisor calls hitl_agent @tool again with user_answer
    8. HITL Agent resumes from Postgres checkpoint with user selection

HOW HITL PROPAGATES BACK TO UI:
  HITL Agent yields: {"type": "interrupt", "question": "...", "options": [...]}
  Supervisor's _invoke_sub_agent() reads this event and raises HITLInterrupt.
  Supervisor's handler() catches HITLInterrupt and yields the interrupt event.
  Platform API forwards interrupt SSE to UI → HITL card displayed.

RESUME FLOW:
  Supervisor calls hitl_agent @tool with payload:
    {"message": "[HITL Answer]: NCI-MATCH. Now search and answer.", "resume": True}
  HITL Agent repairs dangling tool calls (same as single agent),
  injects user_answer, continues from Postgres checkpoint.

CALLED BY:
  Supervisor's hitl_agent @tool in a2a_tools.py via invoke_agent_runtime().
"""

import json
import logging
import os
import re
import time

import boto3
from bedrock_agentcore import BedrockAgentCoreApp, BedrockAgentCoreContext

from agents.hitl.agent import build_hitl_agent
from core.mcp_client import get_mcp_tools

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

app    = BedrockAgentCoreApp()
_agent = None

HITL_TOOLS = {
    "clarify___ask_user_input",
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
    global _agent
    if _agent is None:
        log.info("[HITL] Cold start — building agent")

        if not os.environ.get("OPENAI_API_KEY"):
            prefix = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
            sm     = boto3.client("secretsmanager", region_name="us-east-1")
            secret = json.loads(sm.get_secret_value(SecretId=f"{prefix}/openai")["SecretString"])
            os.environ["OPENAI_API_KEY"] = secret.get("api_key", "")
            log.info("[HITL] OpenAI key loaded")

        all_tools = await get_mcp_tools()
        tools     = [t for t in all_tools if t.name in HITL_TOOLS]
        log.info(f"[HITL] Tools: {[t.name for t in tools]}")

        _agent = await build_hitl_agent(tools=tools)
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
        message    — "[HITL Answer]: NCI-MATCH. Now search and answer."
        session_id — same thread_id
        domain     — "pharma"
        resume     — True
        user_answer — "NCI-MATCH study"

    yields:
        token, tool_start, tool_end — normal streaming
        interrupt                   — HITL card data (Supervisor catches this)
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

    full_answer  = ""
    _tail_buffer = ""
    hitl_input   = {}
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

        # ── Resume: repair dangling tool calls (identical to single agent) ──
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

        # ── Stream events — HITL paths A/B/C/D (identical to single agent) ──
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
                    is_hitl_tool = "ask_user_input" in name
                    if is_hitl_tool:
                        raw        = data.get("input", {})
                        hitl_input = _extract_hitl_input(raw)
                        log.info(f"[HITL] Tool starting  input={hitl_input}")
                    else:
                        yield {"type": "tool_start", "name": name}

                elif kind == "on_tool_end":
                    is_hitl_tool = "ask_user_input" in name
                    if is_hitl_tool:
                        # Path A — tool ran to completion
                        log.info(f"[HITL] Path A — tool ended")
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
                        break
                    else:
                        yield {"type": "tool_end", "name": name}

                elif kind == "on_chain_end":
                    # Path B — __interrupt__ in chain end
                    output = data.get("output", {})
                    if isinstance(output, dict) and "__interrupt__" in output:
                        interrupts = output["__interrupt__"]
                        if interrupts:
                            iv   = interrupts[0]
                            val  = iv.value if hasattr(iv, "value") else iv
                            args = _extract_interrupt_args(val)
                            log.info(f"[HITL] Path B — __interrupt__ in chain")
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
                    "type":          "interrupt",
                    "question":      hitl_input.get("question", "Please clarify:"),
                    "options":       hitl_input.get("options", []),
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

        # Flush tail
        if _tail_buffer:
            clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
            if clean:
                full_answer += clean
                yield {"type": "token", "content": clean}
            _tail_buffer = ""

        # Path C — check Postgres checkpoint for pending interrupt (PRIMARY)
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
                        log.info(f"[HITL] Path C — interrupt in state checkpoint")
                        yield {
                            "type":          "interrupt",
                            "question":      args.get("question", "Please clarify:"),
                            "options":       args.get("options", []),
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
        yield {"type": "done", "latency_ms": elapsed, "answer": full_answer}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8004)