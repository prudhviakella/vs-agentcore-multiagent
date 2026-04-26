"""
semantic_cache.py — SemanticCacheMiddleware
=============================================
Semantic cache check (before) and store (after).
Cache REPLACES work — HIT means NO episodic search, NO tools, NO LLM.

Cache is layer 4 — after PII scrubbing and content filtering,
before episodic memory, tools, and the LLM.

SEMANTIC CACHING:
  Embeds query as vector, finds cached vectors above cosine similarity
  threshold (0.97 pharma, 0.88 general). Same question rephrased still hits.

FIX (Issue #2 — instance state breaks under concurrency):
  BEFORE: Scalar instance state — one value per middleware INSTANCE.
    self._human_message: Optional[str] = None
    self._user_id:       str           = "anonymous"

    AgentCore runs one request per container → scalars were safe there.
    But if this middleware is used in FastAPI (concurrent requests on the
    same process), two requests sharing the same middleware instance would
    corrupt each other's bridge values:

      Request A sets  self._human_message = "metformin dose?"
      Request B sets  self._human_message = "trial enrollment?"
      Request A's after_agent reads "trial enrollment?" → writes wrong answer to cache
      Request B's after_agent reads None (already cleared) → skips cache write

  AFTER: Dict bridge keyed by run_id — one entry per in-flight request.
    self._bridge: dict[str, dict] = {}   # run_id → {question, user_id}

    Each request stores and retrieves its own values using the stable
    run_id from BaseAgentMiddleware._get_run_id(). Concurrent requests
    can't see each other's values.

    The dict is cleaned up by .pop() in after_agent — no memory leak.

  This fix makes the middleware safe in BOTH AgentCore (1 request/container)
  AND FastAPI/Gunicorn (concurrent requests per process).

FIRE-AND-FORGET WRITE:
  Pinecone write runs in background daemon thread after response is sent.
  daemon=True prevents blocking container shutdown.

TTL=3600s (1 hour):
  Balances cache effectiveness vs clinical data freshness risk.

NEVER CACHE GUARDRAIL FALLBACKS:
  Caching "did not meet safety standards" would permanently block valid queries.

TRACER INTEGRATION:
  On cache HIT, annotates trace with:
    TracerMiddleware.update_trace(run_id, {
        "cache_hit":       True,
        "cache_namespace": "cache_pharma",
    })
  On cache MISS, cache_hit=False is the default in tracer — no annotation needed.
"""

import logging
import threading
from typing import Any

from langchain.agents.middleware import AgentState, hook_config
from core.middleware.base import BaseAgentMiddleware
from core.middleware.tracer import TracerMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from core.cache import SemanticCache

log = logging.getLogger(__name__)

_GUARDRAIL_FALLBACK_MARKER = "did not meet safety and accuracy standards"


class SemanticCacheMiddleware(BaseAgentMiddleware):
    """
    Two-hook middleware: cache lookup (before_agent) and cache write (after_agent).

    FIX: Instance state is now a dict keyed by run_id instead of scalars.
    This makes the middleware safe under concurrent requests (FastAPI/Gunicorn)
    while remaining backward-compatible with AgentCore's one-request-per-container model.

    Instance state:
      _cache   — SemanticCache (Pinecone-backed, shared across requests)
      _bridge  — dict[run_id, {question, user_id}] — per-request bridge data
    """

    def __init__(self, cache: SemanticCache):
        super().__init__()
        self._cache:  SemanticCache       = cache
        self._bridge: dict[str, dict]     = {}   # FIX: was two scalars, now one dict

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Cache lookup — short-circuit on HIT, annotate trace.

        SKIP CONDITIONS (return None, proceed):
          1. No human message
          2. Multi-turn conversation (answer is context-dependent)
          3. Empty question
          4. Lookup error (treat as MISS)

        HIT → return cached AIMessage + jump_to="end"
              Annotates trace: cache_hit=True, cache_namespace
        MISS → return None, store question in self._bridge[run_id] for after_agent
        """
        run_id   = self._get_run_id(runtime)
        messages = state.get("messages", [])

        human_messages = [m for m in messages if getattr(m, "type", None) == "human"]

        if not human_messages:
            # No bridge entry → after_agent will skip write
            return None

        if len(human_messages) > 1:
            # Multi-turn: cache lookup without full context would return wrong answer
            log.debug(f"[CACHE_MW] skip — multi-turn ({len(human_messages)} human messages)  run={run_id[:8]}")
            return None

        question = str(human_messages[0].content).strip()
        if not question:
            return None

        ctx     = getattr(runtime, "context", None) or {}
        user_id = ctx.get("user_id", "anonymous")

        log.debug(f"[CACHE_MW] lookup  run={run_id[:8]}  user={user_id}  question='{question[:80]}'")

        try:
            cached = self._cache.lookup(question, user_id=user_id)

            if cached:
                log.info(f"[CACHE_MW] HIT  run={run_id[:8]}  user={user_id}  answer_len={len(cached)}")
                # No bridge entry needed on HIT — after_agent must not write to cache again
                TracerMiddleware.update_trace(run_id, {
                    "cache_hit":       True,
                    "cache_namespace": getattr(self._cache, "namespace", "unknown"),
                })
                return {"messages": [AIMessage(content=cached)], "jump_to": "end"}

            log.debug(f"[CACHE_MW] MISS  run={run_id[:8]}  user={user_id}")

        except Exception as exc:
            log.warning(f"[CACHE_MW] lookup error  run={run_id[:8]}  user={user_id}  error={exc} — treating as MISS")

        # MISS — store bridge data so after_agent can write the answer to cache
        self._bridge[run_id] = {"question": question, "user_id": user_id}
        return None

    @hook_config(can_jump_to=[])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """
        Cache write — store LLM answer for future hits.

        FIX: reads bridge data from self._bridge.pop(run_id) instead of
        scalar instance attributes. .pop() cleans up the entry immediately
        — no leak even if after_agent is called without a matching before_agent.

        SKIP CONDITIONS (no write):
          1. No bridge entry for this run_id (multi-turn, HIT, or empty question)
          2. No AI answer in state
          3. Answer is a guardrail fallback (never cache error responses)
        """
        run_id = self._get_run_id(runtime)

        # FIX: pop is atomic — prevents stale data leaking to other requests
        bridge = self._bridge.pop(run_id, None)
        if not bridge:
            log.debug(f"[CACHE_MW] after_agent skip — no bridge entry  run={run_id[:8]}")
            return None

        question = bridge["question"]
        user_id  = bridge["user_id"]

        answer = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                answer = str(msg.content)
                break

        if not answer:
            log.debug(f"[CACHE_MW] after_agent skip — no AI answer  run={run_id[:8]}  user={user_id}")
            return None

        # Never cache guardrail fallbacks — would permanently block similar queries
        if _GUARDRAIL_FALLBACK_MARKER in answer:
            log.info(f"[CACHE_MW] skip — guardrail fallback, not caching  run={run_id[:8]}  user={user_id}")
            return None

        log.debug(f"[CACHE_MW] storing  run={run_id[:8]}  user={user_id}  question='{question[:80]}'")

        # daemon=True: does not block container shutdown
        threading.Thread(
            target = self._store_sync,
            args   = (question, answer, user_id),
            daemon = True,
        ).start()
        return None

    def _store_sync(self, question: str, answer: str, user_id: str, ttl: int = 3_600) -> None:
        """
        Write question-answer pair to Pinecone cache. Background daemon thread.

        ttl=3600 (1 hour): clinical data changes — new results, FDA decisions.
        1 hour balances freshness vs cache effectiveness for clinical domain.
        """
        try:
            self._cache.store(question, answer, user_id=user_id, ttl=ttl)
            log.info(f"[CACHE_MW] store complete  user={user_id}  answer_len={len(answer)}")
        except Exception as exc:
            # Non-fatal: failed write → next identical query goes to LLM
            log.warning(f"[CACHE_MW] store failed  user={user_id}  error={exc}")