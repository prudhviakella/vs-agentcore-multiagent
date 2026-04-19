"""
agents/supervisor/agent.py
===========================
Supervisor Agent assembly.

IDENTICAL to vs-agentcore-platform-aws/agent/agent.py with ONE difference:
  tools = build_a2a_tools(session_id, domain, token_queue)
          instead of MCP tools from get_mcp_tools()

Everything else is identical:
  - Full 9-layer middleware stack via build_stack()
  - AsyncPostgresSaver (HITL resume lives on Supervisor)
  - SemanticCache + PineconeStore (cache + episodic memory on Supervisor)
  - Safety LLM gpt-4o-mini for OutputGuardrailMiddleware
  - AgentContext schema
  - get_bedrock_prompt(AGENT_NAME) where AGENT_NAME="supervisor-agent"

WHY FULL MIDDLEWARE ON SUPERVISOR ONLY?
  The 9-layer middleware stack handles cross-cutting concerns:
    TracerMiddleware      — observability (DynamoDB traces)
    PIIMiddleware         — strip PII before anything touches it
    ContentFilter         — block off-topic queries
    SemanticCache         — return cached answers (skip all sub-agents)
    EpisodicMemory        — inject past context
    Summarization         — compress long history
    HumanInTheLoop        — NOT HERE (lives on HITL Agent)
    OutputGuardrail       — faithfulness + consistency on final answer
                           (Safety Agent handles this via A2A)

  Sub-agents are lean — they only do their specialised task.
  This avoids running 9 middleware layers × 5 sub-agents per request.

NOTE ON build_agent SIGNATURE:
  Unlike the single agent, build_supervisor_agent() takes session_id,
  domain, and token_queue as arguments because the A2A tools need them
  at construction time (captured in closure).
  This means build_supervisor_agent() is called PER REQUEST not once
  at cold start — but only the tool closures are rebuilt, the heavy
  objects (checkpointer, store, cache, LLM) are cached globally in app.py.
"""

import logging
import os
from typing import Any

import psycopg
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agents.supervisor.middleware import build_stack
from agents.supervisor.a2a_tools import build_a2a_tools
from core.aws import (
    get_bedrock_prompt,
    get_trace_table_name,
    init_pinecone_index,
    init_postgres_url,
)
from core.cache import SemanticCache
from core.pinecone_store import PineconeStore
from core.schema import AgentContext

log = logging.getLogger(__name__)

AGENT_NAME = os.environ.get("AGENT_NAME", "supervisor-agent")


async def build_supervisor_agent(
    session_id:  str,
    domain:      str,
    token_queue: Any,  # asyncio.Queue
    # Heavy objects passed in from app.py (built once at cold start)
    checkpointer: Any,
    store:        Any,
    cache:        Any,
    safety_llm:   Any,
) -> Any:
    """
    Assemble the Supervisor Agent for one request.

    Heavy objects (checkpointer, store, cache, safety_llm) are built
    once at cold start in app.py and passed in here.
    A2A tools are rebuilt per request to capture session_id + token_queue.

    Args:
        session_id:   User thread_id for this request.
        domain:       "pharma".
        token_queue:  asyncio.Queue — sub-agent tokens put here for re-streaming.
        checkpointer: AsyncPostgresSaver built at cold start.
        store:        PineconeStore built at cold start.
        cache:        SemanticCache built at cold start.
        safety_llm:   ChatOpenAI(gpt-4o-mini) built at cold start.

    Returns:
        Compiled LangChain agent ready for astream_events().
    """
    # A2A tools — rebuilt per request (session_id + token_queue in closure)
    tools = build_a2a_tools(
        session_id  = session_id,
        domain      = domain,
        token_queue = token_queue,
    )

    # Middleware stack — same as single agent, same build_stack() call
    middleware = build_stack(
        domain     = domain,
        store      = store,
        safety_llm = safety_llm,
        cache      = cache,
    )

    # System prompt from Bedrock Prompt Management
    # AGENT_NAME="supervisor-agent" reads /supervisor-agent/prod/bedrock/prompt_id
    system_prompt = get_bedrock_prompt(AGENT_NAME)

    agent = create_agent(
        model          = "gpt-4o",
        tools          = tools,
        system_prompt  = system_prompt,
        middleware     = middleware,
        store          = store,
        checkpointer   = checkpointer,
        context_schema = AgentContext,
    )

    log.info(
        f"[Supervisor Agent] Built"
        f"  agent_name={AGENT_NAME}"
        f"  tools={[t.name for t in tools]}"
        f"  middleware={len(middleware)} layers"
        f"  session={session_id[:8]}"
    )
    return agent


async def build_supervisor_cold_start() -> dict:
    """
    Build all heavy objects at cold start. Called once in app.py _ensure_agent().
    Returns a dict of reusable objects passed to build_supervisor_agent() per request.

    Returns:
        {
          "checkpointer": AsyncPostgresSaver,
          "store":        PineconeStore,
          "cache":        SemanticCache,
          "safety_llm":   ChatOpenAI,
        }
    """
    # Embedder — used by SemanticCache + PineconeStore (episodic memory)
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    # Pinecone — same index, different namespaces for cache vs episodic memory
    pinecone_index = init_pinecone_index()
    store          = PineconeStore(index=pinecone_index, embedder=embedder)

    # SemanticCache — pharma: 0.97 threshold (strict for clinical safety)
    cache = SemanticCache(
        index                = pinecone_index,
        embedder             = embedder,
        similarity_threshold = 0.97,
        namespace            = "cache_pharma",
    )

    # Postgres checkpointer — HITL resume + thread history
    conn = await psycopg.AsyncConnection.connect(
        init_postgres_url(),
        autocommit=True,
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()

    # Safety LLM — gpt-4o-mini for OutputGuardrailMiddleware
    safety_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    log.info("[Supervisor] Cold start objects built: checkpointer + store + cache + safety_llm")

    return {
        "checkpointer": checkpointer,
        "store":        store,
        "cache":        cache,
        "safety_llm":   safety_llm,
    }