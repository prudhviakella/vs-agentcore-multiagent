"""
agents/research/app.py
=======================
Research Agent — AgentCore Runtime Entrypoint.

IDENTICAL pattern to vs-agentcore-platform-aws/agent/app.py with these
differences:
  1. Tool filter  : only search_tool + summariser_tool
  2. No middleware : Supervisor owns all cross-cutting concerns
  3. No HITL paths: Research Agent never interrupts
  4. AGENT_NAME   : "research-agent" set by Terraform
                    → drives SSM prompt path via core/aws.get_bedrock_prompt()

CALLED BY:
  Supervisor's research_agent @tool in a2a_tools.py via invoke_agent_runtime().
"""

import json
import logging
import os
import re
import time

import boto3
from bedrock_agentcore import BedrockAgentCoreApp, BedrockAgentCoreContext

from agents.research.agent import build_research_agent
from core.mcp_client import get_mcp_tools

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

app    = BedrockAgentCoreApp()
_agent = None

RESEARCH_TOOLS = {
    "tool-search___search_tool",
    "tool-summariser___summariser_tool",
}

# Same tail-buffer + EPISODIC strip as single agent (precaution)
_EPISODIC_PATTERN = re.compile(r'\s*\nEPISODIC:\s*(YES|NO)[\d.\s]*$', re.IGNORECASE)
_TAIL_SIZE = 60


async def _ensure_agent():
    global _agent
    if _agent is None:
        log.info("[Research] Cold start — building agent")

        if not os.environ.get("OPENAI_API_KEY"):
            prefix = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
            sm     = boto3.client("secretsmanager", region_name="us-east-1")
            secret = json.loads(sm.get_secret_value(SecretId=f"{prefix}/openai")["SecretString"])
            os.environ["OPENAI_API_KEY"] = secret.get("api_key", "")
            log.info("[Research] OpenAI key loaded")

        all_tools = await get_mcp_tools()
        tools     = [t for t in all_tools if t.name in RESEARCH_TOOLS]
        log.info(f"[Research] Tools: {[t.name for t in tools]}")

        _agent = await build_research_agent(tools=tools)
        log.info("[Research] Agent ready")

    return _agent


@app.entrypoint
async def handler(payload: dict, context: BedrockAgentCoreContext):
    """
    Invoked by Supervisor via invoke_agent_runtime().

    payload:
        message    — research query
        session_id — user thread_id
        domain     — "pharma"

    yields:
        token, tool_start, tool_end, done, error
        (same event types as single agent — Supervisor re-streams them)
    """
    t0         = time.perf_counter()
    message    = payload.get("message",    "")
    session_id = payload.get("session_id", "") or getattr(context, "session_id", "")
    domain     = payload.get("domain",     "pharma")

    log.info(f"[Research] session={session_id[:8] if session_id else 'n/a'}  query={message[:80]}...")

    full_answer  = ""
    _tail_buffer = ""

    def _flush_safe():
        nonlocal _tail_buffer
        if len(_tail_buffer) > _TAIL_SIZE:
            safe         = _tail_buffer[:-_TAIL_SIZE]
            _tail_buffer = _tail_buffer[-_TAIL_SIZE:]
            return safe
        return ""

    try:
        agent     = await _ensure_agent()
        thread_id = session_id or f"research-{time.time()}"

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
        input_data    = {"messages": [{"role": "user", "content": message}]}

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
                log.info(f"[Research] → {name}")
                yield {"type": "tool_start", "name": name}

            elif kind == "on_tool_end":
                log.info(f"[Research] ✓ {name}")
                yield {"type": "tool_end", "name": name}

    except Exception as exc:
        log.exception(f"[Research] Error: {exc}")
        if _tail_buffer.strip():
            clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
            if clean:
                full_answer += clean
                yield {"type": "token", "content": clean}
        yield {"type": "error", "message": str(exc)}
        return

    # Flush tail
    if _tail_buffer:
        clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
        if clean:
            full_answer += clean
            yield {"type": "token", "content": clean}

    elapsed = round((time.perf_counter() - t0) * 1_000, 2)
    log.info(f"[Research] Done  latency_ms={elapsed}  answer_len={len(full_answer)}")
    yield {"type": "done", "latency_ms": elapsed, "answer": full_answer}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)