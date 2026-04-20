"""
agents/safety/app.py
=====================
Safety Agent — AgentCore Runtime Entrypoint.

IDENTICAL pattern to research/app.py with these differences:
  1. No MCP tools   : pure LLM judge — no get_mcp_tools() call
  2. AGENT_NAME     : "safety-agent" set by Terraform
  3. Port           : 8003 (local dev)
  4. No tail buffer : no tool calls, answer is short (PASSED / BLOCKED)

RESPONSIBILITY:
  Evaluates answer drafts for faithfulness and consistency
  before the Supervisor returns them to the user.

  Called by Supervisor's safety_agent @tool AFTER research/knowledge
  agents return results and BEFORE the final answer is streamed.

  Two checks:
    Faithfulness  — does the answer contradict the retrieved evidence?
    Consistency   — does the answer contradict prior answers in this thread?

  Returns one of:
    "PASSED"              — safe to return to user
    "BLOCKED: <reason>"   — Supervisor replaces answer with safe refusal

CALLED BY:
  Supervisor's safety_agent @tool in a2a_tools.py via invoke_agent_runtime().
"""

import json
import logging
import os
import time

import boto3
from bedrock_agentcore import BedrockAgentCoreApp, BedrockAgentCoreContext

from agents.safety.agent import build_safety_agent

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

app    = BedrockAgentCoreApp()
_agent = None


async def _ensure_agent():
    global _agent
    if _agent is None:
        log.info("[Safety] Cold start — building agent")

        if not os.environ.get("OPENAI_API_KEY"):
            prefix = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
            sm     = boto3.client("secretsmanager", region_name="us-east-1")
            secret = json.loads(sm.get_secret_value(SecretId=f"{prefix}/openai")["SecretString"])
            os.environ["OPENAI_API_KEY"] = secret.get("api_key", "")
            log.info("[Safety] OpenAI key loaded")

        # No get_mcp_tools() — Safety Agent uses no MCP tools
        _agent = await build_safety_agent()
        log.info("[Safety] Agent ready")

    return _agent


@app.entrypoint
async def handler(payload: dict, context: BedrockAgentCoreContext):
    """
    Invoked by Supervisor via invoke_agent_runtime().

    payload:
        message    — the answer draft to evaluate
                     Supervisor sends: "<answer>\n\nEvidence:\n<retrieved chunks>"
        session_id — user thread_id (for consistency check against prior answers)
        domain     — "pharma"

    yields:
        token      — "PASSED" or "BLOCKED: <reason>"
        done       — with full verdict in answer field
        error      — if evaluation fails (Supervisor treats as PASSED to avoid blocking)
    """
    t0         = time.perf_counter()
    message    = payload.get("message",    "")
    session_id = payload.get("session_id", "") or getattr(context, "session_id", "")
    domain     = payload.get("domain",     "pharma")

    log.info(f"[Safety] session={session_id[:8] if session_id else 'n/a'}  "
             f"evaluating answer_len={len(message)}")

    full_answer = ""

    try:
        agent     = await _ensure_agent()
        thread_id = session_id or f"safety-{time.time()}"

        config = {
            "configurable": {
                "thread_id":  thread_id,
                "user_id":    thread_id,
                "session_id": thread_id,
                "domain":     domain,
            },
            "recursion_limit": 10,
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
            data = event.get("data",  {})

            if kind == "on_chat_model_stream":
                chunk   = data.get("chunk", {})
                content = getattr(chunk, "content", "")
                if isinstance(content, str) and content:
                    full_answer += content
                    yield {"type": "token", "content": content}

    except Exception as exc:
        log.exception(f"[Safety] Error: {exc}")
        # On error yield PASSED — better to let an answer through than
        # block the user because the safety check itself failed
        yield {"type": "error",  "message": str(exc)}
        yield {"type": "token",  "content": "PASSED"}
        full_answer = "PASSED"

    elapsed = round((time.perf_counter() - t0) * 1_000, 2)
    verdict = "PASSED" if "BLOCKED" not in full_answer.upper() else "BLOCKED"
    log.info(f"[Safety] Done  verdict={verdict}  latency_ms={elapsed}")
    # Emit observability span for Supervisor distributed tracing
    yield {
        "type": "span",
        "data": {
            "agent":      "safety",
            "elapsed_ms": elapsed,
        }
    }
    yield {"type": "done", "latency_ms": elapsed, "answer": full_answer}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8003)