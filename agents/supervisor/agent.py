"""
agents/supervisor/agent.py
===========================
Supervisor Agent assembly.

Builds the Supervisor Agent per request from cold-start objects.
Heavy objects (checkpointer, store, cache) are built once and reused.
A2A tools are rebuilt per request — they capture session_id + token_queue
in a closure so streaming events route back to the right handler.

Middleware stack (5 layers):
  TracerMiddleware         — DynamoDB observability, per-request tracing
  SemanticCacheMiddleware  — short-circuit on cache hit
  EpisodicMemoryMiddleware — inject relevant past session context
  SummarizationMiddleware  — compress long conversation history
  OutputGuardrailMiddleware — Bedrock guardrail check on final answer

PII filtering, content filtering, and prompt injection are handled upstream
by the Bedrock Guardrail at the platform gateway (input_guardrail.py).
"""

import logging
import os
from typing import Any

import psycopg
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
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
    session_id:   str,
    domain:       str,
    token_queue:  Any,    # asyncio.Queue — sub-agent events routed here
    checkpointer: Any,    # AsyncPostgresSaver — HITL resume + thread history
    store:        Any,    # PineconeStore — episodic memory
    cache:        Any,    # SemanticCache — semantic cache
) -> Any:
    """
    Assemble the Supervisor Agent for one request.

    Called per request — only A2A tool closures are rebuilt.
    Heavy objects (checkpointer, store, cache) come from cold start.

    Returns:
        Compiled LangGraph agent ready for astream_events().
    """
    tools = build_a2a_tools(
        session_id  = session_id,
        domain      = domain,
        token_queue = token_queue,
    )

    middleware = build_stack(
        domain = domain,
        store  = store,
        cache  = cache,
    )

    # Fetch prompt from Bedrock Prompt Management.
    # SSM is read on every call so a version bump takes effect without restart.
    system_prompt = get_bedrock_prompt(AGENT_NAME)

    # Runtime execution policy appended separately so changing retry limits
    # does not require a prompt version bump in Bedrock.
    system_prompt += (
        "\n\n---\n"
        "## Execution policy\n"
        "Maximum attempts per sub-task: 3\n"
        "Maximum total tool calls this request: 20\n"
        "After 3 failed attempts on a sub-task, move on and state what could not be verified.\n"
        "Never retry a sub-agent with an identical query that already failed.\n"
    )

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
        f"[Supervisor] Agent built"
        f"  tools={[t.name for t in tools]}"
        f"  middleware={len(middleware)}"
        f"  session={session_id[:8]}"
    )
    return agent


async def build_supervisor_cold_start() -> dict:
    """
    Build heavy objects once at container cold start.
    Returned dict is unpacked via **kwargs into build_supervisor_agent() per request.

    Returns:
        {"checkpointer": AsyncPostgresSaver, "store": PineconeStore, "cache": SemanticCache}
    """
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    pinecone_index = init_pinecone_index()
    store = PineconeStore(index=pinecone_index, embedder=embedder)

    cache = SemanticCache(
        index                = pinecone_index,
        embedder             = embedder,
        similarity_threshold = 0.97,
        namespace            = "cache_pharma",
    )

    conn = await psycopg.AsyncConnection.connect(
        init_postgres_url(),
        autocommit=True,
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()

    log.info("[Supervisor] Cold start complete — checkpointer + store + cache ready")

    return {
        "checkpointer": checkpointer,
        "store":        store,
        "cache":        cache,
    }