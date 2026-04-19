"""
core/schema.py — Runtime Context Schema
========================================
Moved to core/ so ALL agents (supervisor + all sub-agents) share it.

ONLY CHANGE from vs-agentcore-platform-aws/agent/schema.py:
  Location: agent/schema.py → core/schema.py
  Import:   from agent.schema import AgentContext
          → from core.schema import AgentContext

Everything else is identical — same TypedDict, same fields, same behaviour.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS "CONTEXT" IN LANGCHAIN 1.0?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangChain agents have two kinds of per-request data:

  1. STATE (config["configurable"]) — LangGraph's built-in mechanism.
     Stores: thread_id, user_id, session_id, domain.
     Used by: LangGraph checkpointer (thread_id), LangChain internals.
     Accessed via: config["configurable"]["user_id"]

  2. CONTEXT (context= kwarg) — LangChain 1.0's middleware mechanism.
     Stores: the same user_id, session_id, domain fields.
     Used by: middleware hooks (before_agent, after_agent).
     Accessed via: runtime.context["user_id"]

WHY DO WE NEED BOTH?
  LangGraph reads thread_id from config["configurable"] for checkpointing.
  Middleware reads user_id from runtime.context for episodic memory namespacing.
  Without both, either LangGraph breaks (no thread_id in configurable) or
  middleware breaks (no user_id in context).

  In every agent's app.py we set BOTH:
    config        = {"configurable": {"thread_id": t, "user_id": t, "session_id": t}}
    agent_context = {"user_id": t, "session_id": t, "domain": "pharma"}
    agent.astream_events(input_data, config=config, context=agent_context)

WHY TypedDict AND NOT Pydantic BaseModel?
  LangChain 1.0 hard requirement:
    "Custom context schemas must be TypedDict types.
     Pydantic models and dataclasses are no longer supported."
  Passing a Pydantic model to create_agent(context_schema=...) raises TypeError.

total=False — WHY ALL KEYS OPTIONAL?
  - Tests that don't set up a full context don't crash
  - Callers that only set user_id don't need session_id and domain
  - Middleware uses .get() with defaults everywhere — no KeyError risk

USED IN EVERY AGENT:
  from core.schema import AgentContext

  agent = create_agent(
      ...
      context_schema = AgentContext,
  )
"""

from typing import TypedDict


class AgentContext(TypedDict, total=False):
    """
    Per-request runtime configuration injected via the context= kwarg.

    All fields optional (total=False) — middleware uses .get() with defaults.

    Set in every agent's app.py:
        agent_context = {
            "user_id":    thread_id,
            "session_id": thread_id,
            "domain":     "pharma",
        }
        agent.astream_events(input_data, config=config, context=agent_context)

    Read in middleware (Supervisor only):
        runtime.context.get("user_id",    "anonymous")
        runtime.context.get("session_id", "")
        runtime.context.get("domain",     "general")
    """

    user_id: str
    # WHO is making this request.
    # Used by EpisodicMemoryMiddleware to namespace Pinecone vectors:
    #   namespace = f"episodic__{user_id}"
    # Used by TracerMiddleware to tag DynamoDB trace records.
    # In production: set to thread_id (AgentCore runtimeSessionId).

    session_id: str
    # WHICH conversation session this belongs to.
    # Used by middleware to correlate before_agent() / after_agent() hooks
    # via a stable run_id for the duration of the session.
    # In production: set to thread_id (same as user_id for single-user sessions).

    domain: str
    # WHAT domain this agent is operating in.
    # Values: "pharma" | "general"
    # Used by SemanticCacheMiddleware for cache namespace:
    #   "cache_pharma"  (threshold 0.97 — strict for clinical safety)
    #   "cache_general" (threshold 0.88 — lenient for broader queries)
    # Sub-agents receive this from Supervisor payload and pass it through.
