"""
agents/research/app.py
====================
Research Agent — AgentCore Runtime Entrypoint.

Called by the Supervisor via invoke_agent_runtime().
Streams SSE events back: token, tool_start, tool_end, chart (chart only), done, error.

AgentCore automatically pipes stdout to CloudWatch:
  /aws/bedrock-agentcore/runtimes/{runtime_id}-DEFAULT
No Watchtower handler needed — logging.basicConfig() to stdout is sufficient.
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

# Stdout → AgentCore → CloudWatch (automatic — no Watchtower needed)
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger(__name__)

app    = BedrockAgentCoreApp()
_agent = None

# Prevents two concurrent first requests both triggering cold start
_agent_lock = __import__("asyncio").Lock()

RESEARCH_TOOLS = {
    "tool-search___search_tool",
    "tool-summariser___summariser_tool",
}

# Strips the EPISODIC memory tag gpt-5.5 appends to every response.
# Must never reach the Supervisor or UI.
_EPISODIC_PATTERN = re.compile(r'\s*\nEPISODIC:\s*(YES|NO)[\d.\s]*$', re.IGNORECASE)
_TAIL_SIZE = 60


def _load_langsmith_from_ssm() -> None:
    """
    Load LangSmith tracing credentials from Secrets Manager.
    Called at cold start after IAM credentials are available.
    Short-circuits if LANGSMITH_API_KEY already set (local dev).
    """
    if os.environ.get("LANGSMITH_API_KEY"):
        return
    try:
        prefix = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
        sm     = boto3.client("secretsmanager", region_name="us-east-1")
        secret = json.loads(sm.get_secret_value(SecretId=f"{prefix}/langsmith")["SecretString"])
        os.environ["LANGSMITH_API_KEY"] = secret.get("api_key", "")
        os.environ["LANGSMITH_PROJECT"]  = secret.get("project", "langchain-agent-experiments")
        os.environ["LANGSMITH_TRACING"]  = secret.get("tracing", "true")
        log.info(f"[Research] LangSmith loaded from SSM")
    except Exception as e:
        log.warning(f"[Research] LangSmith not available — tracing disabled: {e}")



async def _ensure_agent():
    """
    Build the agent once on first request, reuse on all subsequent requests.

    Uses a double-checked lock:
      Fast path  — agent already built, return immediately (no lock)
      Slow path  — first request, acquire lock, build agent, release lock
      Guard      — second request waiting on lock sees agent already built
    """
    global _agent
    if _agent is not None:
        return _agent  # fast path — already initialised

    import asyncio
    async with _agent_lock:
        if _agent is not None:
            return _agent  # another coroutine built it while we waited

        log.info(f"[Research] Cold start — building agent")

        if not os.environ.get("OPENAI_API_KEY"):
            prefix = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
            sm     = boto3.client("secretsmanager", region_name="us-east-1")
            secret = json.loads(sm.get_secret_value(SecretId=f"{prefix}/openai")["SecretString"])
            os.environ["OPENAI_API_KEY"] = secret.get("api_key", "")
            log.info(f"[Research] OpenAI key loaded")

        # Load LangSmith now — IAM credentials available at this point
        _load_langsmith_from_ssm()

        all_tools = await get_mcp_tools()
        tools     = [t for t in all_tools if t.name in RESEARCH_TOOLS]
        log.info(f"[Research] Tools: {[t.name for t in tools]}")

        _agent = await build_research_agent(tools=tools)
        log.info(f"[Research] Agent ready")

    return _agent


@app.entrypoint
async def handler(payload: dict, context: BedrockAgentCoreContext):
    """
    Invoked by Supervisor via invoke_agent_runtime().

    payload:
        message    — the query
        session_id — user thread_id (namespaced by supervisor: {thread_id}__research)
        domain     — "pharma"

    yields:
        token      — streamed LLM tokens
        tool_start — MCP tool starting
        tool_end   — MCP tool complete
        done       — stream complete with full answer
        error      — on failure
    """
    t0         = time.perf_counter()
    message    = payload.get("message",    "")
    session_id = payload.get("session_id", "") or getattr(context, "session_id", "")
    domain     = payload.get("domain",     "pharma")

    log.info(f"[Research] session={session_id[:8] if session_id else 'n/a'}  query={message[:80]}...")

    full_answer  = ""
    _tail_buffer = ""

    def _flush_safe() -> str:
        nonlocal _tail_buffer
        if len(_tail_buffer) > _TAIL_SIZE:
            safe         = _tail_buffer[:-_TAIL_SIZE]
            _tail_buffer = _tail_buffer[-_TAIL_SIZE:]
            return safe
        return ""

    _rag_metrics: dict = {}  # accumulated from search_tool/summariser_tool JSON envelopes

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
                # Parse JSON envelope from search_tool / summariser_tool
                _tool_output = data.get("output", "")
                # Extract text from ToolMessage(content=[{'type':'text','text':'...'}])
                import json as _j
                try:
                    _content = getattr(_tool_output, "content", _tool_output)
                    _raw = ""
                    if isinstance(_content, list) and _content:
                        _block = _content[0]
                        if isinstance(_block, dict):
                            _raw = _block.get("text", _block.get("content", ""))
                        elif isinstance(_block, str):
                            _raw = _block
                    elif isinstance(_content, str):
                        _raw = _content
                    # Unwrap double-encoded string if needed
                    _raw = _raw.strip()
                    if _raw.startswith('"') and _raw.endswith('"'):
                        try:
                            _raw = _j.loads(_raw)
                        except Exception:
                            pass
                    _raw = _raw.strip() if isinstance(_raw, str) else ""

                    # New format: "RAG_METRICS:{...}\n\n<chunks text>"
                    if _raw.startswith("RAG_METRICS:"):
                        _lines = _raw.split("\n\n", 1)
                        _metrics_str = _lines[0][len("RAG_METRICS:"):]
                        try:
                            _metrics = _j.loads(_metrics_str)
                            _rag_metrics.update(_metrics)
                            log.info(f"[Research] RAG metrics from {name}: {_metrics}")
                        except Exception as _je:
                            log.info(f"[Research] RAG_METRICS parse error: {_je}")
                    else:
                        log.info(f"[Research] No RAG_METRICS header in {name} output")
                except Exception as _je:
                    log.info(f"[Research] tool output parse error: {_je}")
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

    # Flush tail buffer — strips EPISODIC suffix before yielding final tokens
    if _tail_buffer:
        clean = _EPISODIC_PATTERN.sub("", _tail_buffer).rstrip()
        if clean:
            full_answer += clean
            yield {"type": "token", "content": clean}

    elapsed = round((time.perf_counter() - t0) * 1_000, 2)
    log.info(f"[Research] Done  latency_ms={elapsed}  answer_len={len(full_answer)}")

    # Observability span — consumed by TracerMiddleware in Supervisor
    # Span event — supervisor timing/routing data
    yield {"type": "span", "data": {"agent": "research", "elapsed_ms": elapsed}}
    # Embed rag_metrics in done event.
    # a2a_tools reads done event → merges into span_data → tracer extracts from span buffer.
    # This avoids span buffer cross-module issues entirely.
    _done_event: dict = {"type": "done", "latency_ms": elapsed, "answer": full_answer}
    if _rag_metrics:
        _done_event["rag_metrics"] = _rag_metrics
        log.info(f"[Research] RAG metrics in done event: {list(_rag_metrics.keys())}")
    yield _done_event


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("AGENT_PORT", "8080")))