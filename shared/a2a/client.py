"""
shared/a2a/client.py
====================
A2A HTTP client used exclusively by the Supervisor Agent to:
  1. Discover sub-agents by reading their Agent Cards at cold start
  2. Send TaskRequests to individual sub-agents
  3. Send multiple TaskRequests in parallel (Research + Knowledge simultaneously)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY ONLY THE SUPERVISOR USES THIS?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Sub-agents are called BY the Supervisor — they never call each other.
The communication pattern is strictly hub-and-spoke:

    Supervisor ──► Research Agent
    Supervisor ──► Knowledge Agent   (parallel with Research)
    Supervisor ──► HITL Agent
    Supervisor ──► Safety Agent
    Supervisor ──► Chart Agent

Sub-agents only need shared/a2a/server.py (the server side).
Only the Supervisor needs this client.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARALLEL EXECUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For specific queries the Supervisor calls Research + Knowledge in parallel
using asyncio.gather. This cuts latency roughly in half:

    Sequential: research(8s) + knowledge(3s) = 11s
    Parallel:   max(research(8s), knowledge(3s)) = 8s

send_tasks_parallel() handles this. Exceptions in one task don't block
the other — each is caught individually and returned as a failed TaskResult.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENT CARD CACHING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Agent Cards are fetched once at cold start and cached in memory.
The cache lives for the lifetime of the Supervisor container.
If a sub-agent is redeployed with new skills, the Supervisor container
must be restarted to pick up the new Agent Card.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE IN SUPERVISOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    from shared.a2a.client import A2AClient

    client = A2AClient()

    # Cold start — discover all sub-agents
    await client.discover_agents({
        "research":  "https://<research-runtime-url>",
        "knowledge": "https://<knowledge-runtime-url>",
        "hitl":      "https://<hitl-runtime-url>",
        "safety":    "https://<safety-runtime-url>",
        "chart":     "https://<chart-runtime-url>",
    })

    # Single task
    result = await client.send_task(
        agent_url   = "https://<research-runtime-url>",
        message     = "What are Phase 3 results for BNT162b2?",
        session_id  = "abc-123",
        skill_id    = "search_clinical_trials",
        metadata    = {"domain": "pharma", "intent": "specific"}
    )

    # Parallel tasks — Research + Knowledge simultaneously
    results = await client.send_tasks_parallel([
        {
            "url":        "https://<research-runtime-url>",
            "message":    "BNT162b2 Phase 3 efficacy data",
            "session_id": "abc-123",
            "skill_id":   "search_clinical_trials",
            "metadata":   {"domain": "pharma"}
        },
        {
            "url":        "https://<knowledge-runtime-url>",
            "message":    "BNT162b2 trial sponsors and relationships",
            "session_id": "abc-123",
            "skill_id":   "query_graph",
            "metadata":   {"domain": "pharma"}
        }
    ])
    research_result, knowledge_result = results
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Optional

import httpx

from .schemas import (
    AgentCard,
    Artifact,
    JSONRPCRequest,
    JSONRPCResponse,
    JSONRPCError,
    Message,
    TaskResult,
    TaskSendParams,
    TaskStatus,
    TextPart,
)

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TASK_TIMEOUT_SECONDS = 120.0   # sub-agents can take up to 2 min (Pinecone + LLM)
CARD_TIMEOUT_SECONDS = 10.0    # agent card fetch should be fast
MAX_RETRIES = 2                # retry failed requests once before giving up


class A2AClient:
    """
    Async A2A HTTP client.

    One instance lives on the Supervisor Agent for its entire lifetime.
    Thread-safe — asyncio coroutines share this instance.

    Attributes:
        auth_token:   Bearer token for authenticating with sub-agents.
                      Set from Secrets Manager at Supervisor cold start.
        _card_cache:  In-memory cache of Agent Cards keyed by base URL.
                      Populated by discover_agents() at cold start.
    """

    def __init__(self, auth_token: Optional[str] = None):
        self.auth_token = auth_token
        self._card_cache: dict[str, AgentCard] = {}

    # ── Headers ───────────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.auth_token:
            h["Authorization"] = f"Bearer {self.auth_token}"
        return h

    # ── Agent Card Discovery ──────────────────────────────────────────────────

    async def fetch_agent_card(self, base_url: str) -> AgentCard:
        """
        Fetch and cache the Agent Card from GET /.well-known/agent.json.

        Called by discover_agents() at cold start. Subsequent calls for the
        same URL return the cached card without hitting the network.

        Args:
            base_url: The root URL of the sub-agent AgentCore Runtime.
                      e.g. "https://abc123.bedrock-agentcore.us-east-1.amazonaws.com"

        Returns:
            AgentCard with name, description, url, capabilities, and skills.

        Raises:
            httpx.HTTPStatusError: if the sub-agent returns a non-2xx response.
            httpx.TimeoutException: if the request times out.
        """
        if base_url in self._card_cache:
            log.debug(f"[A2A] Card cache hit: {base_url}")
            return self._card_cache[base_url]

        url = f"{base_url.rstrip('/')}/.well-known/agent.json"
        log.info(f"[A2A] Fetching Agent Card: {url}")

        async with httpx.AsyncClient(timeout=CARD_TIMEOUT_SECONDS) as client:
            resp = await client.get(url, headers=self._headers())
            resp.raise_for_status()
            card = AgentCard(**resp.json())

        self._card_cache[base_url] = card
        log.info(
            f"[A2A] Discovered: {card.name}"
            f"  skills={[s.id for s in card.skills]}"
            f"  tags={[t for s in card.skills for t in s.tags]}"
        )
        return card

    async def discover_agents(
        self, agent_urls: dict[str, str]
    ) -> dict[str, AgentCard]:
        """
        Discover all sub-agents in parallel at Supervisor cold start.

        Fetches all Agent Cards concurrently so cold start is fast even
        with 5 sub-agents.

        Args:
            agent_urls: dict mapping agent name → base URL.
                        e.g. {"research": "https://...", "knowledge": "https://..."}

        Returns:
            dict mapping agent name → AgentCard.

        Example:
            cards = await client.discover_agents({
                "research":  os.environ["RESEARCH_AGENT_URL"],
                "knowledge": os.environ["KNOWLEDGE_AGENT_URL"],
                "hitl":      os.environ["HITL_AGENT_URL"],
                "safety":    os.environ["SAFETY_AGENT_URL"],
                "chart":     os.environ["CHART_AGENT_URL"],
            })
        """
        log.info(f"[A2A] Discovering {len(agent_urls)} sub-agents in parallel...")

        results = await asyncio.gather(
            *[self.fetch_agent_card(url) for url in agent_urls.values()],
            return_exceptions=True
        )

        cards = {}
        for name, result in zip(agent_urls.keys(), results):
            if isinstance(result, Exception):
                log.error(f"[A2A] Failed to discover {name}: {result}")
            else:
                cards[name] = result

        log.info(f"[A2A] Discovery complete — {len(cards)}/{len(agent_urls)} agents online")
        return cards

    # ── Send Single Task ──────────────────────────────────────────────────────

    async def send_task(
        self,
        agent_url:  str,
        message:    str,
        session_id: str,
        skill_id:   Optional[str] = None,
        metadata:   Optional[dict] = None,
    ) -> TaskResult:
        """
        Send a tasks/send JSON-RPC request to one sub-agent.

        Builds the full A2A wire format, POSTs to /a2a, parses the response.
        Retries once on network failure before returning a failed TaskResult.

        Args:
            agent_url:  Base URL of the target sub-agent.
            message:    The query or instruction as plain text.
            session_id: User's thread_id — ties this task to a conversation.
            skill_id:   Optional skill ID hint so the sub-agent knows which
                        capability to use. If None the agent auto-selects.
            metadata:   Additional routing context:
                          domain    — "pharma"
                          intent    — "specific" | "vague" | "chart"
                          run_id    — supervisor's trace run_id

        Returns:
            TaskResult with status.state == "completed", "input-required", or "failed".

        Wire format sent:
            POST <agent_url>/a2a
            {
              "jsonrpc": "2.0",
              "method":  "tasks/send",
              "id":      "<request-uuid>",
              "params": {
                "id":        "<task-uuid>",
                "sessionId": "<session_id>",
                "message": {
                  "role":  "user",
                  "parts": [{"type": "text", "text": "<message>"}]
                },
                "metadata": {"domain": "pharma", "skill_id": "...", ...}
              }
            }
        """
        task_id    = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        meta       = {**(metadata or {})}
        if skill_id:
            meta["skill_id"] = skill_id

        params = TaskSendParams(
            id        = task_id,
            sessionId = session_id,
            message   = Message(
                role  = "user",
                parts = [TextPart(text=message).model_dump()]
            ),
            metadata = meta,
        )

        rpc_request = JSONRPCRequest(
            id     = request_id,
            method = "tasks/send",
            params = params.model_dump(),
        )

        log.info(
            f"[A2A] → {agent_url}"
            f"  task_id={task_id[:8]}"
            f"  skill={skill_id}"
            f"  msg={message[:60]}..."
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=TASK_TIMEOUT_SECONDS) as client:
                    resp = await client.post(
                        f"{agent_url.rstrip('/')}/a2a",
                        json    = rpc_request.model_dump(),
                        headers = self._headers(),
                    )
                    resp.raise_for_status()

                rpc_resp = JSONRPCResponse(**resp.json())

                if rpc_resp.error:
                    log.error(
                        f"[A2A] ← Error from {agent_url}"
                        f"  task={task_id[:8]}"
                        f"  error={rpc_resp.error}"
                    )
                    return _failed_result(task_id, str(rpc_resp.error))

                result = TaskResult(**rpc_resp.result)
                log.info(
                    f"[A2A] ← {agent_url}"
                    f"  task={task_id[:8]}"
                    f"  state={result.status.state}"
                    f"  artifacts={len(result.artifacts)}"
                )
                return result

            except httpx.TimeoutException:
                log.warning(f"[A2A] Timeout on attempt {attempt}/{MAX_RETRIES}: {agent_url}")
                if attempt == MAX_RETRIES:
                    return _failed_result(task_id, f"Task timed out after {TASK_TIMEOUT_SECONDS}s")

            except httpx.HTTPStatusError as exc:
                log.error(f"[A2A] HTTP {exc.response.status_code} from {agent_url}: {exc}")
                return _failed_result(task_id, f"HTTP {exc.response.status_code}: {exc.response.text[:200]}")

            except Exception as exc:
                log.error(f"[A2A] Unexpected error calling {agent_url}: {exc}")
                return _failed_result(task_id, str(exc))

        # Should never reach here but satisfies type checker
        return _failed_result(task_id, "Max retries exceeded")

    # ── Send Parallel Tasks ───────────────────────────────────────────────────

    async def send_tasks_parallel(
        self, tasks: list[dict]
    ) -> list[TaskResult]:
        """
        Send multiple tasks simultaneously using asyncio.gather.

        Used by Supervisor for Research + Knowledge agents in parallel.
        A failure in one task does NOT cancel the others — each exception
        is caught and returned as a failed TaskResult so the Supervisor
        can still use results from whichever tasks succeeded.

        Args:
            tasks: list of task dicts, each with keys:
                     url        — sub-agent base URL (required)
                     message    — query text (required)
                     session_id — user's thread_id (required)
                     skill_id   — optional skill hint
                     metadata   — optional extra context

        Returns:
            list[TaskResult] in the same order as the input tasks list.

        Example:
            results = await client.send_tasks_parallel([
                {
                    "url":        research_url,
                    "message":    "BNT162b2 Phase 3 efficacy",
                    "session_id": session_id,
                    "skill_id":   "search_clinical_trials",
                    "metadata":   {"domain": "pharma", "intent": "specific"}
                },
                {
                    "url":        knowledge_url,
                    "message":    "BNT162b2 sponsor and trial relationships",
                    "session_id": session_id,
                    "skill_id":   "query_graph",
                    "metadata":   {"domain": "pharma", "intent": "specific"}
                }
            ])
            research_result, knowledge_result = results
        """
        if not tasks:
            return []

        log.info(f"[A2A] Launching {len(tasks)} tasks in parallel")

        coroutines = [
            self.send_task(
                agent_url  = t["url"],
                message    = t["message"],
                session_id = t["session_id"],
                skill_id   = t.get("skill_id"),
                metadata   = t.get("metadata", {}),
            )
            for t in tasks
        ]

        raw_results = await asyncio.gather(*coroutines, return_exceptions=True)

        final: list[TaskResult] = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                log.error(f"[A2A] Parallel task {i} raised: {r}")
                final.append(_failed_result(str(uuid.uuid4()), str(r)))
            else:
                final.append(r)

        completed = sum(1 for r in final if r.is_complete())
        log.info(f"[A2A] Parallel done — {completed}/{len(tasks)} completed")
        return final

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_cached_card(self, base_url: str) -> Optional[AgentCard]:
        """Return cached Agent Card for a URL, or None if not yet fetched."""
        return self._card_cache.get(base_url)

    def list_cached_agents(self) -> list[str]:
        """Return names of all discovered agents."""
        return [card.name for card in self._card_cache.values()]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _failed_result(task_id: str, error_message: str) -> TaskResult:
    """
    Build a failed TaskResult for error cases.
    Keeps Supervisor logic clean — it always gets a TaskResult back,
    never an exception, from send_task() and send_tasks_parallel().
    """
    return TaskResult(
        id       = task_id,
        status   = TaskStatus(state="failed"),
        metadata = {"error": error_message},
    )