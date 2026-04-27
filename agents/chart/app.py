"""
agents/chart/app.py
====================
Chart Agent — AgentCore Runtime Entrypoint.

IDENTICAL pattern to research/app.py with these differences:
  1. Tool filter  : search_tool + summariser_tool + chart___chart_tool
  2. AGENT_NAME   : "chart-agent" set by Terraform
  3. Port         : 8005 (local dev)
  4. Chart event  : detects {"type": "chart"} in tool output and yields it

RESPONSIBILITY:
  Called by Supervisor when the query involves numerical comparison or
  visualisation. The agent:
    1. Searches Pinecone for numerical evidence (search_tool)
    2. Extracts and synthesises the numbers (summariser_tool)
    3. Generates a Chart.js JSON config (chart___chart_tool → chart_lambda)
    4. Yields the chart config as a special event for the UI to render

CHART EVENT:
  {"type": "chart", "config": {...Chart.js config...}, "chart_type": "bar"}
  UI detects this event and renders a Chart.js canvas inline in the chat bubble.
  Supervisor re-streams this event through to Platform API → UI.

EXAMPLE QUERIES ROUTED HERE:
  "Compare efficacy across COVID-19 vaccine trials"
  "Show me enrollment numbers for cancer trials"
  "Visualise Phase distribution across all trials"

CALLED BY:
  Supervisor's chart_agent @tool in a2a_tools.py via invoke_agent_runtime().
"""

import json
import logging
import os
import re
import time

import boto3
import watchtower
from bedrock_agentcore import BedrockAgentCoreApp, BedrockAgentCoreContext

from agents.chart.agent import build_chart_agent
from core.mcp_client import get_mcp_tools

_LOG_GROUP = os.environ.get("LOG_GROUP_NAME", "/agentcore/vs-agentcore-ma/chart-agent")

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

app    = BedrockAgentCoreApp()
_agent = None

CHART_TOOLS = {
    "tool-search___search_tool",
    "tool-summariser___summariser_tool",
    "tool-chart___chart_tool",
}

_EPISODIC_PATTERN = re.compile(r'\s*\nEPISODIC:\s*(YES|NO)[\d.\s]*$', re.IGNORECASE)
_TAIL_SIZE = 60


async def _ensure_agent():
    global _agent
    if _agent is None:
        log.info("[Chart] Cold start — building agent")

        if not os.environ.get("OPENAI_API_KEY"):
            prefix = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
            sm     = boto3.client("secretsmanager", region_name="us-east-1")
            secret = json.loads(sm.get_secret_value(SecretId=f"{prefix}/openai")["SecretString"])
            os.environ["OPENAI_API_KEY"] = secret.get("api_key", "")
            log.info("[Chart] OpenAI key loaded")

        all_tools = await get_mcp_tools()
        tools     = [t for t in all_tools if t.name in CHART_TOOLS]
        log.info(f"[Chart] Tools: {[t.name for t in tools]}")

        _agent = await build_chart_agent(tools=tools)
        log.info("[Chart] Agent ready")

    return _agent


@app.entrypoint
async def handler(payload: dict, context: BedrockAgentCoreContext):
    """
    Invoked by Supervisor via invoke_agent_runtime().

    payload:
        message    — chart query e.g. "Compare COVID vaccine efficacy"
        session_id — user thread_id
        domain     — "pharma"

    yields:
        token      — streamed answer text
        tool_start — when search/summariser/chart tool starts
        tool_end   — when tool completes
        chart      — Chart.js config for UI to render inline
        done       — with full answer
        error      — on failure
    """
    t0         = time.perf_counter()
    message    = payload.get("message",    "")
    session_id = payload.get("session_id", "") or getattr(context, "session_id", "")
    domain     = payload.get("domain",     "pharma")

    log.info(f"[Chart] session={session_id[:8] if session_id else 'n/a'}  query={message[:80]}...")

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
        thread_id = session_id or f"chart-{time.time()}"

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
                log.info(f"[Chart] → {name}")
                yield {"type": "tool_start", "name": name}

            elif kind == "on_tool_end":
                log.info(f"[Chart] ✓ {name}")
                # Check if chart_tool returned a chart config
                if "chart" in name:
                    import json as _json
                    output = data.get("output", {})
                    log.info(f"[Chart] tool output type={type(output).__name__}  preview={str(output)[:200]}")

                    # MCP tools can return: string, dict, ToolMessage, or list of content blocks
                    # Unwrap all possible formats
                    raw = output

                    # If it's a LangChain message object with content
                    if hasattr(raw, "content"):
                        raw = raw.content

                    # If it's a list (content blocks), join text blocks
                    if isinstance(raw, list):
                        raw = " ".join(
                            block.get("text", "") if isinstance(block, dict) else str(block)
                            for block in raw
                        )

                    # If it's a string, try JSON parse
                    if isinstance(raw, str):
                        try:
                            raw = _json.loads(raw)
                        except Exception:
                            pass

                    log.info(f"[Chart] parsed output type={type(raw).__name__}  has_chart={'chart' in raw if isinstance(raw, dict) else False}")

                    # chart_lambda returns {"chart": {...}, "chart_type": "bar"}
                    if isinstance(raw, dict) and "chart" in raw:
                        log.info(f"[Chart] ✅ Chart config found: type={raw.get('chart_type')}")
                        yield {
                            "type":       "chart",
                            "config":     raw["chart"],
                            "chart_type": raw.get("chart_type", "bar"),
                        }
                yield {"type": "tool_end", "name": name}

    except Exception as exc:
        log.exception(f"[Chart] Error: {exc}")
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
    log.info(f"[Chart] Done  latency_ms={elapsed}  answer_len={len(full_answer)}")
    # Emit observability span for Supervisor distributed tracing
    yield {
        "type": "span",
        "data": {
            "agent":      "chart",
            "elapsed_ms": elapsed,
        }
    }
    yield {"type": "done", "latency_ms": elapsed, "answer": full_answer}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("AGENT_PORT", "8080")))