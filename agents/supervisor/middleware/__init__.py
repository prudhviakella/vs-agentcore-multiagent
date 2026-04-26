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

# ── Domain middleware ─────────────────────────────────────────────────────
from agents.supervisor.middleware.output_guardrail import OutputGuardrailMiddleware

# NOTE: ContentFilterMiddleware and DomainPIIMiddleware have been removed.
# Bedrock Guardrails (configured in deploy.sh step_guardrails) now handles
# both at the gateway layer before any message reaches the agent:
#
#   ContentFilterMiddleware (check_toxic regex)
#     → replaced by Bedrock VIOLENCE + MISCONDUCT content filters (HIGH)
#
#   DomainPIIMiddleware (email/CC regex on input, email regex on output)
#     → replaced by Bedrock sensitive info policy:
#         INPUT:  EMAIL, PHONE, NAME, ADDRESS, DOB, SSN — ANONYMIZED/BLOCK
#         OUTPUT: EMAIL — ANONYMIZED (added to guardrail config)
#
# This eliminates redundant checks and ensures a single enforcement point.


def build_stack(domain: str, store, cache: SemanticCache) -> list:
    """
    Assemble the middleware stack for the Supervisor Agent.

    Layers removed vs original:
      - DomainPIIMiddleware    → Bedrock input + output sensitive info policy
      - ContentFilterMiddleware → Bedrock VIOLENCE/MISCONDUCT content filters
      - OutputGuardrailMiddleware → enabled below (Bedrock output guardrail)

    Args:
        domain:  "pharma" | "general"
        store:   PineconeStore (episodic memory)
        cache:   SemanticCache (semantic cache)

    Returns:
        Ordered list of middleware instances for create_agent(middleware=...).
    """
    return [
        # Layer 1: Tracer — FIRST so every request is recorded
        TracerMiddleware(dynamodb_table_name=get_trace_table_name()),

        # Layer 2: Semantic Cache — short-circuit on cache hit
        SemanticCacheMiddleware(cache=cache),

        # Layer 3: Episodic Memory — inject relevant past context
        EpisodicMemoryMiddleware(store=store),

        # Layer 4: Summarization — compress long history
        SummarizationMiddleware(
            model   = "openai:gpt-4o-mini",
            trigger = ("tokens", 8_000),
            keep    = ("messages", 10),
        ),

        # Layer 5: Output Guardrail — Bedrock guardrail on final answer
        OutputGuardrailMiddleware(),
    ]


__all__ = [
    "TracerMiddleware",
    "SemanticCacheMiddleware",
    "EpisodicMemoryMiddleware",
    "OutputGuardrailMiddleware",
    "build_stack",
]