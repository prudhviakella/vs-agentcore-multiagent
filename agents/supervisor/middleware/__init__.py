"""
agents/supervisor/middleware/__init__.py — Supervisor Middleware Stack
=======================================================================
IDENTICAL to vs-agentcore-platform-aws/agent/middleware/__init__.py
with ONE change:

  from agent.middleware.xxx  →  from agents.supervisor.middleware.xxx

The domain-agnostic middleware (tracer, cache, episodic memory, summarization)
still imports from core/ — unchanged.

The pharma-domain middleware (pii, content_filter, output_guardrail, hitl)
now lives in agents/supervisor/middleware/ — copy from single agent:

  cp agent/middleware/pii.py              agents/supervisor/middleware/
  cp agent/middleware/content_filter.py   agents/supervisor/middleware/
  cp agent/middleware/action_guardrail.py agents/supervisor/middleware/
  cp agent/middleware/output_guardrail.py agents/supervisor/middleware/
  cp agent/middleware/hitl.py             agents/supervisor/middleware/

WHY ONLY SUPERVISOR HAS MIDDLEWARE?
  The 9-layer middleware stack handles cross-cutting concerns.
  Sub-agents are lean — they only do their specialised task.
  Supervisor owns: PII scrubbing, content filter, semantic cache,
  episodic memory, summarization, output guardrail, tracing.
  HITL middleware lives on the HITL Agent (not Supervisor).

For full documentation on middleware ordering and design decisions see:
  vs-agentcore-platform-aws/agent/middleware/__init__.py
"""

from langchain.agents.middleware import HumanInTheLoopMiddleware, SummarizationMiddleware

# ── Domain-agnostic middleware from core/ ─────────────────────────────────
# Unchanged from single agent — same import paths
from core.aws import get_trace_table_name
from core.middleware.tracer import TracerMiddleware
from core.middleware.semantic_cache import SemanticCacheMiddleware
from core.middleware.episodic_memory import EpisodicMemoryMiddleware
from core.cache import SemanticCache

# ── Pharma-domain middleware ──────────────────────────────────────────────
# CHANGED: agent.middleware.* → agents.supervisor.middleware.*
from agents.supervisor.middleware.pii import DomainPIIMiddleware
from agents.supervisor.middleware.content_filter import ContentFilterMiddleware
from agents.supervisor.middleware.action_guardrail import ActionGuardrailMiddleware
from agents.supervisor.middleware.output_guardrail import OutputGuardrailMiddleware


def build_stack(domain: str, store, safety_llm, cache: SemanticCache) -> list:
    """
    Assemble the 9-layer middleware stack for the Supervisor Agent.

    IDENTICAL to single agent build_stack() — same layers, same order,
    same thresholds. Only import paths changed.

    Args:
        domain:     "pharma" | "general"
        store:      PineconeStore (episodic memory)
        safety_llm: ChatOpenAI(gpt-4o-mini) (output guardrail)
        cache:      SemanticCache (semantic cache)

    Returns:
        Ordered list of middleware instances for create_agent(middleware=...).
    """
    return [
        # Layer 1: Tracer — FIRST, records every request including cache hits
        TracerMiddleware(dynamodb_table_name=get_trace_table_name()),

        # Layer 2: PII — EARLY, scrub before LLM, cache, or memory touches it
        DomainPIIMiddleware(),

        # Layer 3: Content Filter — block off-topic/toxic before LLM
        ContentFilterMiddleware(),

        # Layer 4: Semantic Cache — SHORT-CIRCUIT, skip layers 5-9 on hit
        SemanticCacheMiddleware(cache=cache),

        # Layer 5: Episodic Memory — inject relevant past context
        EpisodicMemoryMiddleware(store=store),

        # Layer 6: Summarization — compress history at 8,000 tokens
        SummarizationMiddleware(
            model   = "openai:gpt-4o-mini",
            trigger = ("tokens", 8_000),
            keep    = ("messages", 10),
        ),

        # Layer 7: HITL — NOTE: HumanInTheLoopMiddleware is NOT here.
        # HITL lives on the HITL Agent. The Supervisor routes to HITL Agent
        # via hitl_agent @tool in a2a_tools.py.
        # HumanInTheLoopMiddleware(interrupt_on={"clarify___ask_user_input": True}),

        # Layer 8: Action Guardrail — DISABLED (same as single agent)
        # ActionGuardrailMiddleware(),

        # Layer 9: OutputGuardrailMiddleware DISABLED on Supervisor.
        # Safety Agent handles faithfulness + consistency via A2A tool call.
        # Running it here too causes double-evaluation and score leakage in stream.
        # Re-enable after calibration with faithfulness_threshold=0.70.
        # OutputGuardrailMiddleware(
        #     llm                    = safety_llm,
        #     faithfulness_threshold = 0.00,
        #     confidence_threshold   = 0.00,
        # ),
    ]


__all__ = [
    "TracerMiddleware",
    "DomainPIIMiddleware",
    "ContentFilterMiddleware",
    "SemanticCacheMiddleware",
    "EpisodicMemoryMiddleware",
    "ActionGuardrailMiddleware",
    "OutputGuardrailMiddleware",
    "build_stack",
]