"""
agents/supervisor/agent.py
===========================
Supervisor Agent Assembly — Cold Start + Per-Request Build.

RESPONSIBILITY
--------------
This module has two jobs:

1. build_supervisor_cold_start()  — runs ONCE at container startup
   Builds everything expensive: Postgres connection, Pinecone index,
   embedding model, semantic cache. These are reused across all requests
   for the lifetime of the container.

2. build_supervisor_agent()  — runs ONCE per request
   Takes the cold-start objects and assembles a ready-to-run agent.
   Only the A2A tool closures are rebuilt per request because they capture
   session_id and token_queue which are different for every request.

WHY SPLIT INTO TWO FUNCTIONS?
------------------------------
If we built everything per request:
  - Postgres connection: ~500ms
  - Pinecone index init: ~300ms
  - Embedding model load: ~200ms
  - Total: ~1s of pure overhead BEFORE any LLM call

With cold start:
  - Per-request overhead: ~10ms (just tool closure creation)
  - Everything else reused from memory

AGENT ARCHITECTURE
------------------
The supervisor is a LangChain ReAct agent compiled into a LangGraph graph.

  LangChain layer:
    ChatOpenAI(gpt-5.5) + structured tools → ReAct agent
    "Given this conversation, which tool should I call next?"

  LangGraph layer:
    Wraps the ReAct agent in a state machine graph:
      supervisor node → tools node → supervisor node → ... → END
    Adds checkpointing (HITL resume) and streaming (astream_events).

  Middleware layer:
    5 middleware wrap each tool call:
    Tracer → SemanticCache → EpisodicMemory → Summarization → OutputGuardrail

STRUCTURED TOOLS
----------------
Each sub-agent is exposed to gpt-5.5 as a typed LangChain StructuredTool:
  - knowledge_agent  : Neo4j knowledge graph queries
  - research_agent   : Pinecone document search
  - chart_agent      : Chart.js visualisation generation
  - safety_agent     : Safety/compliance verification
  - clarify___ask_user_input : HITL — pause and ask the user

Pydantic input schemas ensure gpt-5.5 passes correctly typed arguments.
The tool coroutine calls the actual AgentCore runtime via boto3.

MIDDLEWARE STACK (5 layers, applied in order per tool call)
-----------------------------------------------------------
  1. TracerMiddleware         — writes span to DynamoDB (request_id, tool, latency, result)
  2. SemanticCacheMiddleware  — vector similarity check; returns cached result if hit (threshold 0.97)
  3. EpisodicMemoryMiddleware — injects relevant past session summaries into tool context
  4. SummarizationMiddleware  — compresses long conversation history before LLM call
  5. OutputGuardrailMiddleware — Bedrock guardrail check on supervisor's final answer

  PII filtering and prompt injection are handled UPSTREAM at the platform
  gateway (input_guardrail.py) — not here.
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

# AGENT_NAME is used to look up the Bedrock Prompt Management version.
# Changing this env var (and the corresponding SSM parameter) lets you
# swap prompts without rebuilding the container image.
AGENT_NAME = os.environ.get("AGENT_NAME", "supervisor-agent")


# ── Per-Request Agent Build ────────────────────────────────────────────────

async def build_supervisor_agent(
    session_id:   str,
    domain:       str,
    token_queue:  Any,    # asyncio.Queue — A2A tool events route back here
    checkpointer: Any,    # AsyncPostgresSaver — HITL pause/resume + history
    store:        Any,    # PineconeStore — episodic memory read/write
    cache:        Any,    # SemanticCache — semantic deduplication cache
) -> Any:
    """
    Assemble the Supervisor Agent for one request.

    WHAT IS REBUILT PER REQUEST
    ---------------------------
    A2A tools: rebuilt every time because they close over session_id and
    token_queue, both of which are unique per request.

      session_id  — identifies which Postgres checkpoint to load/save.
                    Two concurrent users must have completely isolated state.

      token_queue — the asyncio.Queue that bridges the boto3 thread back to
                    the async SSE stream. Each request has its own queue;
                    events from one user's chart_agent must not appear in
                    another user's SSE stream.

    WHAT IS REUSED (comes from cold start via **kwargs)
    ---------------------------------------------------
    checkpointer  — one Postgres connection, shared across all requests.
                    AsyncPostgresSaver is thread-safe for concurrent reads.
    store         — Pinecone index client, stateless, safe to share.
    cache         — SemanticCache wraps the same Pinecone index, stateless.

    Parameters
    ----------
    session_id   : str  — maps to LangGraph thread_id and Postgres checkpoint key
    domain       : str  — "pharma", "oncology" etc — routes tool context
    token_queue  : asyncio.Queue — chart/interrupt events go here from threads
    checkpointer : AsyncPostgresSaver — from cold start
    store        : PineconeStore — from cold start
    cache        : SemanticCache — from cold start

    Returns
    -------
    Compiled LangGraph agent ready for astream_events().
    """

    # ── A2A Structured Tools ───────────────────────────────────────────────
    # build_a2a_tools() creates AsyncStructuredTool instances.
    # Each tool:
    #   - Has a Pydantic args_schema so gpt-5.5 passes typed arguments
    #   - Closes over session_id (for tracing) and token_queue (for events)
    #   - Coroutine calls _invoke_sub_agent_prod() → boto3 → AgentCore runtime
    #
    # Tools built here: knowledge_agent, research_agent, chart_agent,
    #                   safety_agent, clarify___ask_user_input
    tools = build_a2a_tools(
        session_id  = session_id,
        domain      = domain,
        token_queue = token_queue,
    )

    # ── Middleware Stack ───────────────────────────────────────────────────
    # build_stack() returns the 5-layer middleware in call order:
    #   Tracer → SemanticCache → EpisodicMemory → Summarization → OutputGuardrail
    #
    # Each layer is a callable that wraps the next layer, forming a pipeline.
    # On every tool call:
    #   1. Tracer starts a span (DynamoDB write at end)
    #   2. SemanticCache checks vector similarity against past queries
    #      → hit (similarity >= 0.97): returns cached result, skips tool call
    #      → miss: continues to next layer
    #   3. EpisodicMemory injects relevant past session context into the prompt
    #   4. Summarization compresses messages if conversation history is too long
    #   5. OutputGuardrail checks the supervisor's final response via Bedrock
    middleware = build_stack(
        domain = domain,
        store  = store,    # PineconeStore used by EpisodicMemory layer
        cache  = cache,    # SemanticCache used by SemanticCache layer
    )

    # ── System Prompt ──────────────────────────────────────────────────────
    # Fetched from Bedrock Prompt Management on every request.
    # WHY not cache this?
    #   Prompt Management versioning lets you update the prompt without
    #   redeploying the container. By fetching on every request, a prompt
    #   change takes effect immediately on the next request.
    #   SSM stores the prompt_id → Bedrock returns the versioned content.
    system_prompt = get_bedrock_prompt(AGENT_NAME)

    # Execution policy appended as a separate block.
    # WHY separate from the main prompt?
    #   Operational limits (max retries, max tool calls) change more
    #   frequently than the reasoning instructions. Keeping them separate
    #   means a limit change does not require a Bedrock prompt version bump —
    #   just a code change and container redeploy.
    system_prompt += (
        "\n\n---\n"
        "## Execution policy\n"
        "Maximum attempts per sub-task: 3\n"
        "Maximum total tool calls this request: 20\n"
        "After 3 failed attempts on a sub-task, move on and state what could not be verified.\n"
        "Never retry a sub-agent with an identical query that already failed.\n"
    )

    # ── LangChain + LangGraph Agent ───────────────────────────────────────
    # create_agent() does two things:
    #
    # 1. LangChain ReAct agent:
    #      llm = ChatOpenAI(model="gpt-5.5", streaming=True)
    #      llm_with_tools = llm.bind_tools(tools)
    #      This gives gpt-5.5 the tool schemas so it knows exactly which
    #      tools exist and what arguments each expects (Pydantic-enforced).
    #
    # 2. LangGraph StateGraph:
    #      Wraps the ReAct agent in a graph with two nodes:
    #        "supervisor" node: runs llm_with_tools
    #          → if tool_calls in output → route to "tools" node
    #          → if no tool_calls → route to END
    #        "tools" node: ToolNode executes all tool_calls in parallel
    #          → always routes back to "supervisor" node
    #
    #      graph.compile(checkpointer=checkpointer) adds:
    #        - Postgres-backed checkpoint on every node transition
    #        - Required for HITL: state is saved before the interrupt
    #          and restored when the user answers
    #        - Also gives conversation history across turns (same thread_id)
    agent = create_agent(
        model          = "gpt-5.5",
        tools          = tools,
        system_prompt  = system_prompt,
        middleware     = middleware,
        store          = store,          # episodic memory read/write
        checkpointer   = checkpointer,   # Postgres — HITL + history
        context_schema = AgentContext,   # Pydantic schema for LangGraph state
    )

    log.info(
        f"[Supervisor] Agent built"
        f"  tools={[t.name for t in tools]}"
        f"  middleware={len(middleware)}"
        f"  session={session_id[:8]}"
    )
    return agent


# ── Cold Start ─────────────────────────────────────────────────────────────

async def build_supervisor_cold_start() -> dict:
    """
    Build expensive shared objects once at container startup.

    Called by _ensure_cold_start() in app.py the first time a request
    arrives. The returned dict is cached in _cold_start_objects and passed
    to build_supervisor_agent() as **kwargs on every subsequent request.

    OBJECTS BUILT AND WHY THEY ARE EXPENSIVE
    -----------------------------------------
    1. OpenAIEmbeddings
         Initialises the OpenAI embedding client (text-embedding-3-small).
         Used by both SemanticCache and PineconeStore.
         Cost: HTTP client setup + model validation ~100ms.

    2. Pinecone index
         init_pinecone_index() reads the index name from SSM and connects
         to the Pinecone hosted index.
         Cost: network round-trip to Pinecone API ~200-300ms.

    3. PineconeStore
         Wraps the Pinecone index for episodic memory read/write.
         Stores session summaries as vectors — supervisor retrieves relevant
         past sessions at the start of each request via EpisodicMemoryMiddleware.

    4. SemanticCache
         Also wraps the Pinecone index but in a separate namespace ("cache_pharma").
         On every tool call, computes embedding of the query and checks cosine
         similarity against cached queries. If similarity >= 0.97, returns
         the cached result — no AgentCore invocation needed.
         similarity_threshold=0.97: very high by design. Clinical research
         queries must be semantically almost identical to get a cache hit.
         A lower threshold risks returning wrong trial data.

    5. AsyncPostgresSaver (LangGraph checkpointer)
         Opens a persistent async Postgres connection (psycopg3 async).
         checkpointer.setup() creates the LangGraph checkpoint tables if
         they don't exist (messages, writes, migrations).
         This connection is reused across ALL requests — Postgres handles
         concurrent reads from the single connection safely.

         WHY Postgres for checkpointing?
           HITL requires persisting the full LangGraph message state between
           two HTTP requests (the interrupt request and the resume request).
           In-memory state would be lost. Postgres gives durable, fast,
           structured storage for message history and tool call state.
           thread_id is the checkpoint key — each user session is isolated.

    Returns
    -------
    dict with keys: checkpointer, store, cache
    Unpacked via **kwargs into build_supervisor_agent() on every request.
    """

    # Single embedding model instance shared by cache and store.
    # text-embedding-3-small: 1536 dimensions, fast, cost-effective.
    # Used for both semantic cache lookups and episodic memory retrieval.
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    # Connect to Pinecone — one index serves both store and cache
    # (separated by namespace: "pharma" for documents, "cache_pharma" for cache)
    pinecone_index = init_pinecone_index()

    # PineconeStore: episodic memory backend
    # Stores: session summaries, key decisions, past query-answer pairs
    # Read by EpisodicMemoryMiddleware to inject context into new sessions
    store = PineconeStore(index=pinecone_index, embedder=embedder)

    # SemanticCache: avoids redundant AgentCore invocations
    # Query → embedding → cosine similarity check against cached queries
    # Hit (>= 0.97 similarity): return cached answer directly
    # Miss: execute tool normally, cache the result for future hits
    # namespace="cache_pharma": isolated from document vectors in same index
    cache = SemanticCache(
        index                = pinecone_index,
        embedder             = embedder,
        similarity_threshold = 0.97,   # high threshold — clinical data precision
        namespace            = "cache_pharma",
    )

    # AsyncPostgresSaver: LangGraph checkpoint backend
    # autocommit=True: each checkpoint write is immediately durable.
    # Without autocommit, a crash mid-request could roll back checkpoint
    # writes and corrupt the HITL resume state.
    conn = await psycopg.AsyncConnection.connect(
        init_postgres_url(),   # reads from SSM: host, port, dbname, user, password
        autocommit=True,
    )
    checkpointer = AsyncPostgresSaver(conn)

    # setup() is idempotent — creates LangGraph tables if missing:
    #   checkpoints, checkpoint_writes, checkpoint_migrations
    # Safe to call on every cold start.
    await checkpointer.setup()

    log.info("[Supervisor] Cold start complete — checkpointer + store + cache ready")

    # Return as dict so build_supervisor_agent() can receive via **kwargs.
    # New objects added here automatically flow through without changing
    # the function signature.
    return {
        "checkpointer": checkpointer,
        "store":        store,
        "cache":        cache,
    }