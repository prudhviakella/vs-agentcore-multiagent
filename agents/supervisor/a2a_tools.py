"""
agents/supervisor/a2a_tools.py
================================
A2A tool wrappers — LangChain @tool functions that call sub-agent
AgentCore Runtimes via invoke_agent_runtime().

HOW THIS WORKS:
  Supervisor's create_agent() receives these as regular LangChain tools.
  When the LLM calls research_agent("BNT162b2 efficacy"), this @tool:
    1. Invokes Research Agent's AgentCore Runtime via invoke_agent_runtime()
    2. Reads streaming chunks from the response body
    3. Parses SSE events (token/tool_start/tool_end/chart/interrupt/done)
    4. Puts events on token_queue (Supervisor re-streams to Platform API)
    5. Returns full answer string to the LLM as ToolMessage

STREAMING:
  One token_queue per request, shared by all 5 A2A tools via closure.
  Supervisor handler() reads the queue and yields events to Platform API.
  The user sees a continuous token stream even as work moves between agents.

HITL PROPAGATION:
  HITL Agent yields {"type": "interrupt", ...}
  _invoke_sub_agent() raises HITLInterrupt
  Supervisor handler() catches HITLInterrupt → yields interrupt SSE
  Platform API → UI → HITL card shown

RUNTIME ARNs from SSM (written by deploy.sh step_ssm_arns):
  /vs-agentcore-multiagent/prod/agents/research/runtime_arn
  /vs-agentcore-multiagent/prod/agents/knowledge/runtime_arn
  /vs-agentcore-multiagent/prod/agents/hitl/runtime_arn
  /vs-agentcore-multiagent/prod/agents/safety/runtime_arn
  /vs-agentcore-multiagent/prod/agents/chart/runtime_arn
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import boto3
from langchain_core.tools import tool

log = logging.getLogger(__name__)

REGION     = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")

_executor = ThreadPoolExecutor(max_workers=10)


# ── HITL Interrupt ────────────────────────────────────────────────────────────

class HITLInterrupt(Exception):
    """
    Raised when HITL Agent returns an interrupt event.
    Caught by Supervisor handler() which yields interrupt SSE to Platform API.
    """
    def __init__(self, question: str, options: list[str], allow_freetext: bool = True):
        self.question       = question
        self.options        = options
        self.allow_freetext = allow_freetext
        super().__init__(f"HITL: {question}")


# ── Runtime ARN loader ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_runtime_arns() -> dict[str, str]:
    """Load sub-agent runtime ARNs from SSM at cold start. Cached."""
    ssm  = boto3.client("ssm", region_name=REGION)
    arns = {}
    for name in ["research", "knowledge", "hitl", "safety", "chart"]:
        try:
            arn = ssm.get_parameter(
                Name=f"{SSM_PREFIX}/agents/{name}/runtime_arn"
            )["Parameter"]["Value"]
            arns[name] = arn
            log.info(f"[A2A] ARN loaded for {name}: {arn[:50]}...")
        except Exception as exc:
            log.error(f"[A2A] Failed to load ARN for {name}: {exc}")
    return arns


# ── Core invoke ───────────────────────────────────────────────────────────────

async def _invoke_sub_agent(
    agent_name:  str,
    payload:     dict,
    token_queue: asyncio.Queue,
) -> str:
    """
    Invoke a sub-agent runtime and stream its response.

    Runs invoke_agent_runtime() in thread pool (boto3 is sync).
    Parses SSE events and puts them on token_queue for re-streaming.

    Returns:
        Full answer string from sub-agent.

    Raises:
        HITLInterrupt: on interrupt event.
        RuntimeError:  on error event.
    """
    arns = _get_runtime_arns()
    if agent_name not in arns:
        raise RuntimeError(f"No runtime ARN for '{agent_name}'. Available: {list(arns.keys())}")

    runtime_arn = arns[agent_name]
    session_id  = payload.get("session_id", f"{agent_name}-session")
    loop        = asyncio.get_event_loop()
    chunk_queue = asyncio.Queue()

    def _stream_in_thread():
        try:
            client   = boto3.client("bedrock-agentcore", region_name=REGION)
            response = client.invoke_agent_runtime(
                agentRuntimeArn  = runtime_arn,
                runtimeSessionId = session_id,
                payload          = json.dumps(payload).encode("utf-8"),
            )
            streaming_body = response.get("response")
            if streaming_body:
                for chunk in streaming_body.iter_chunks(chunk_size=1024):
                    if chunk:
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, chunk)
            else:
                raw = response.get("body", b"")
                if raw:
                    loop.call_soon_threadsafe(chunk_queue.put_nowait, raw)
        except Exception as exc:
            err = json.dumps({"type": "error", "message": str(exc)}).encode()
            loop.call_soon_threadsafe(chunk_queue.put_nowait, err)
        finally:
            loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

    loop.run_in_executor(_executor, _stream_in_thread)

    full_answer = ""

    while True:
        chunk = await chunk_queue.get()
        if chunk is None:
            break

        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk

        for line in text.splitlines():
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
                content = event.get("content", "")
                if content:
                    full_answer += content
                    await token_queue.put({"type": "token", "content": content})

            elif etype in ("tool_start", "tool_end"):
                await token_queue.put({
                    "type": etype,
                    "name": event.get("name", ""),
                })

            elif etype == "chart":
                await token_queue.put({
                    "type":       "chart",
                    "config":     event.get("config", {}),
                    "chart_type": event.get("chart_type", "bar"),
                })

            elif etype == "interrupt":
                raise HITLInterrupt(
                    question       = event.get("question", "Please clarify:"),
                    options        = event.get("options", []),
                    allow_freetext = event.get("allow_freetext", True),
                )

            elif etype == "error":
                raise RuntimeError(f"Sub-agent '{agent_name}': {event.get('message', '')}")

            elif etype == "done":
                done_answer = event.get("answer", "")
                if done_answer and len(done_answer) > len(full_answer):
                    full_answer = done_answer

    log.info(f"[A2A] {agent_name} done  answer_len={len(full_answer)}")
    return full_answer


# ── Tool factory ──────────────────────────────────────────────────────────────

def build_a2a_tools(
    session_id:  str,
    domain:      str,
    token_queue: asyncio.Queue,
) -> list[Any]:
    """
    Build 5 A2A tools for the Supervisor's create_agent().

    session_id, domain, token_queue captured in closure — LLM calls
    tool("query") cleanly without needing to pass session context.

    Args:
        session_id:  User thread_id — passed to every sub-agent.
        domain:      "pharma" — passed to sub-agents.
        token_queue: Sub-agent events put here for Supervisor to re-stream.
    """

    @tool
    async def research_agent(query: str) -> str:
        """
        Search clinical trial documents and return a cited evidence-based answer.

        Use for specific trial questions — efficacy, safety, dosage, eligibility.
        Searches 5,772 Pinecone document chunks and synthesises with GPT-4o.

        Args:
            query: Research question to answer from clinical trial documents.
        """
        log.info(f"[Supervisor] → Research  query={query[:60]}...")
        return await _invoke_sub_agent("research",
            {"message": query, "session_id": session_id, "domain": domain},
            token_queue)

    @tool
    async def knowledge_agent(query: str) -> str:
        """
        Query the biomedical knowledge graph for structured trial information.

        Use for trial discovery, sponsor lookups, drug/disease relationships,
        and geographic queries. Queries Neo4j with Cypher.

        Args:
            query: Graph query in natural language.
        """
        log.info(f"[Supervisor] → Knowledge  query={query[:60]}...")
        return await _invoke_sub_agent("knowledge",
            {"message": query, "session_id": session_id, "domain": domain},
            token_queue)

    @tool
    async def hitl_agent(query: str) -> str:
        """
        Ask user a clarification question when query is too broad.

        Use when query is vague ("show me trials", "search for cancer studies").
        PAUSES execution — user sees a card with numbered options to click.

        Args:
            query: Broad query needing clarification.
        """
        log.info(f"[Supervisor] → HITL  query={query[:60]}...")
        return await _invoke_sub_agent("hitl",
            {"message": query, "session_id": session_id, "domain": domain},
            token_queue)

    @tool
    async def safety_agent(answer: str) -> str:
        """
        Evaluate answer draft for faithfulness and consistency.

        Always call BEFORE returning a final answer.
        Returns "PASSED" or "BLOCKED: <reason>".

        Args:
            answer: Draft answer to evaluate.
        """
        log.info(f"[Supervisor] → Safety  answer_len={len(answer)}")
        return await _invoke_sub_agent("safety",
            {"message": answer, "session_id": session_id, "domain": domain},
            token_queue)

    @tool
    async def chart_agent(query: str) -> str:
        """
        Generate a Chart.js visualisation from numerical clinical trial data.

        Use when answer has comparative numbers or user asks for a chart.
        Renders inline in the chat bubble as an interactive canvas.

        Args:
            query: Visualisation query e.g. "Compare COVID vaccine efficacy".
        """
        log.info(f"[Supervisor] → Chart  query={query[:60]}...")
        return await _invoke_sub_agent("chart",
            {"message": query, "session_id": session_id, "domain": domain},
            token_queue)

    return [research_agent, knowledge_agent, hitl_agent, safety_agent, chart_agent]