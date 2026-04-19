"""
shared/a2a/server.py
====================
Base A2A server used by ALL sub-agents (Research, Knowledge, HITL, Safety, Chart).

Every sub-agent imports A2AServer, registers one task handler function,
and gets a fully compliant A2A endpoint with zero boilerplate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS PROVIDES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Every sub-agent gets these routes for free:

    GET  /.well-known/agent.json   → serves the AgentCard (discovery)
    POST /a2a                      → receives JSON-RPC 2.0, calls handler
    GET  /health                   → health check for ECS/ALB

The sub-agent only needs to implement ONE function:

    async def handle_task(task: Task) -> TaskResult:
        # your agent logic here — call LangGraph, tools, LLM etc.
        ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW SUB-AGENTS USE THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # agents/research/app.py
    from shared.a2a.server import A2AServer
    from shared.a2a.schemas import (
        AgentCard, AgentCapabilities, Skill,
        Task, TaskResult, TaskStatus, Artifact, TextPart
    )

    AGENT_CARD = AgentCard(
        name        = "Research Agent",
        description = "Semantic search over 5,772 clinical trial chunks.",
        url         = os.environ["RESEARCH_AGENT_URL"],
        skills      = [
            Skill(
                id          = "search_clinical_trials",
                name        = "Search Clinical Trials",
                description = "Semantic search over Pinecone. Returns top-k chunks with citations.",
                tags        = ["search", "rag", "pinecone", "clinical-trials"],
                examples    = ["What are Phase 3 results for BNT162b2?"]
            ),
            Skill(
                id          = "synthesise_evidence",
                name        = "Synthesise Evidence",
                description = "Combine multiple chunks into one cited answer via GPT-4o.",
                tags        = ["synthesis", "summarisation"],
                examples    = ["Summarise the efficacy data across all COVID vaccines"]
            ),
        ]
    )

    server = A2AServer(agent_card=AGENT_CARD)

    @server.task_handler
    async def handle_task(task: Task) -> TaskResult:
        query  = task.message.get_text()
        result = await run_research_graph(query, task.sessionId)
        return TaskResult(
            id       = task.id,
            status   = TaskStatus(state="completed"),
            artifacts = [Artifact(parts=[TextPart(text=result).model_dump()])]
        )

    app = server.build_app()
    # uvicorn agents.research.app:app --host 0.0.0.0 --port 8000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JSON-RPC 2.0 REQUEST / RESPONSE FLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Supervisor sends:
    POST /a2a
    {
      "jsonrpc": "2.0",
      "method":  "tasks/send",
      "id":      "req-uuid",
      "params": {
        "id":        "task-uuid",
        "sessionId": "thread-uuid",
        "message":   {"role": "user", "parts": [{"type": "text", "text": "..."}]},
        "metadata":  {"domain": "pharma", "skill_id": "search_clinical_trials"}
      }
    }

  This server:
    1. Validates method == "tasks/send"
    2. Deserialises params → Task
    3. Calls registered handle_task(task)
    4. Wraps TaskResult in JSONRPCResponse
    5. Returns JSON

  Sub-agent returns:
    {
      "jsonrpc": "2.0",
      "id":      "req-uuid",
      "result": {
        "id":        "task-uuid",
        "sessionId": "thread-uuid",
        "status":    {"state": "completed"},
        "artifacts": [{"parts": [{"type": "text", "text": "BNT162b2 showed 95%..."}]}]
      }
    }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ERROR HANDLING STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Exceptions in handle_task() are caught here and returned as:
  - JSON-RPC error response (for protocol-level errors)
  - TaskResult with state="failed" (for task execution errors)

The Supervisor's client.py always gets a valid TaskResult back —
it never receives a raw exception from a sub-agent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUPPORTED JSON-RPC METHODS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  tasks/send    → run the task (main method)
  tasks/get     → poll task status (stub — all tasks are synchronous)
  tasks/cancel  → cancel a running task (stub — not yet implemented)
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Awaitable, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    AgentCard,
    Artifact,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    Message,
    Task,
    TaskResult,
    TaskStatus,
    TextPart,
)

log = logging.getLogger(__name__)

# Type alias for the handler function every sub-agent must implement
TaskHandler = Callable[[Task], Awaitable[TaskResult]]


class A2AServer:
    """
    Base A2A server — used by ALL sub-agents.

    Wraps a FastAPI app with A2A routes. Sub-agents register their
    task handler via the @server.task_handler decorator, then call
    server.build_app() to get the FastAPI instance for uvicorn.

    Attributes:
        agent_card: The AgentCard this server publishes at /.well-known/agent.json.
        _handler:   The task handler function registered by the sub-agent.
    """

    def __init__(self, agent_card: AgentCard):
        self.agent_card = agent_card
        self._handler: Optional[TaskHandler] = None

    # ── Handler Registration ──────────────────────────────────────────────────

    def task_handler(self, fn: TaskHandler) -> TaskHandler:
        """
        Decorator — register the sub-agent's task handler function.

        The handler must be an async function with this signature:
            async def handle_task(task: Task) -> TaskResult

        It is called once per incoming tasks/send request.
        It should never raise — catch exceptions internally and return
        a TaskResult with state="failed" if something goes wrong.

        Example:
            @server.task_handler
            async def handle_task(task: Task) -> TaskResult:
                query = task.message.get_text()
                answer = await my_agent_logic(query)
                return TaskResult(
                    id        = task.id,
                    status    = TaskStatus(state="completed"),
                    artifacts = [Artifact(parts=[TextPart(text=answer).model_dump()])]
                )
        """
        self._handler = fn
        log.info(f"[A2A Server] Handler registered: {fn.__name__}")
        return fn

    # ── App Builder ───────────────────────────────────────────────────────────

    def build_app(self, existing_app: Optional[FastAPI] = None) -> FastAPI:
        """
        Build (or extend) a FastAPI app with all A2A routes.

        Args:
            existing_app: Optional existing FastAPI app to add routes to.
                          If None a new FastAPI app is created.
                          Pass an existing app if the sub-agent needs custom
                          routes in addition to the A2A endpoints.

        Returns:
            FastAPI app ready to be served by uvicorn.

        Routes added:
            GET  /.well-known/agent.json
            POST /a2a
            GET  /health
        """
        app = existing_app or FastAPI(
            title       = self.agent_card.name,
            description = self.agent_card.description,
            version     = self.agent_card.version,
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins  = ["*"],
            allow_methods  = ["GET", "POST", "OPTIONS"],
            allow_headers  = ["*"],
        )

        # ── GET /.well-known/agent.json ───────────────────────────────────────
        @app.get("/.well-known/agent.json", tags=["A2A"])
        async def agent_card_endpoint():
            """
            A2A Agent Card discovery endpoint.
            Called by Supervisor at cold start to discover this agent's skills.
            """
            return JSONResponse(content=self.agent_card.model_dump())

        # ── POST /a2a ─────────────────────────────────────────────────────────
        @app.post("/a2a", tags=["A2A"])
        async def handle_rpc(raw: dict):
            """
            A2A JSON-RPC 2.0 task endpoint.
            Called by Supervisor to send tasks to this sub-agent.

            Validates the request, deserialises the Task,
            calls the registered handler, wraps the result.
            """
            request_id = raw.get("id", str(uuid.uuid4()))

            # ── Validate handler is registered ────────────────────────────────
            if not self._handler:
                log.error("[A2A Server] No task handler registered")
                return _error_response(
                    request_id, -32603, "Internal error: no task handler registered"
                )

            # ── Validate JSON-RPC method ──────────────────────────────────────
            method = raw.get("method", "")

            if method == "tasks/get":
                # Stub — we don't support async polling yet
                return _error_response(
                    request_id, -32601, "tasks/get not implemented — all tasks are synchronous"
                )

            if method == "tasks/cancel":
                # Stub — cancellation not yet implemented
                return _error_response(
                    request_id, -32601, "tasks/cancel not implemented"
                )

            if method != "tasks/send":
                return _error_response(
                    request_id, -32601, f"Method not found: {method}"
                )

            # ── Deserialise params → Task ─────────────────────────────────────
            params = raw.get("params", {})
            if not params:
                return _error_response(
                    request_id, -32602, "Missing params in JSON-RPC request"
                )

            try:
                task = Task(
                    id        = params.get("id", str(uuid.uuid4())),
                    sessionId = params.get("sessionId", str(uuid.uuid4())),
                    message   = Message(**params["message"]),
                    metadata  = params.get("metadata", {}),
                )
            except Exception as exc:
                log.error(f"[A2A Server] Failed to deserialise Task: {exc}")
                return _error_response(
                    request_id, -32602, f"Invalid params: {exc}"
                )

            # ── Call handler ──────────────────────────────────────────────────
            log.info(
                f"[A2A Server] ← task_id={task.id[:8]}"
                f"  session={task.sessionId[:8]}"
                f"  skill={task.metadata.get('skill_id', 'auto')}"
                f"  msg={task.message.get_text()[:60]}..."
            )
            t0 = time.perf_counter()

            try:
                result: TaskResult = await self._handler(task)
            except Exception as exc:
                elapsed = round((time.perf_counter() - t0) * 1000, 2)
                log.error(
                    f"[A2A Server] Handler exception"
                    f"  task={task.id[:8]}"
                    f"  elapsed_ms={elapsed}"
                    f"  error={exc}"
                )
                result = TaskResult(
                    id       = task.id,
                    status   = TaskStatus(state="failed"),
                    metadata = {"error": str(exc), "elapsed_ms": elapsed},
                )

            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            log.info(
                f"[A2A Server] → task_id={task.id[:8]}"
                f"  state={result.status.state}"
                f"  artifacts={len(result.artifacts)}"
                f"  elapsed_ms={elapsed}"
            )

            # ── Wrap in JSON-RPC response ─────────────────────────────────────
            return JSONResponse(
                content=JSONRPCResponse(
                    id     = request_id,
                    result = result.model_dump(),
                ).model_dump()
            )

        # ── GET /health ───────────────────────────────────────────────────────
        @app.get("/health", tags=["System"])
        async def health():
            """
            Health check for ECS/ALB target group health checks.
            Returns agent name and skill count so ops can verify the right image is running.
            """
            return {
                "status":      "ok",
                "agent":       self.agent_card.name,
                "version":     self.agent_card.version,
                "skills":      [s.id for s in self.agent_card.skills],
                "handler_registered": self._handler is not None,
            }

        log.info(
            f"[A2A Server] Built app for: {self.agent_card.name}"
            f"  skills={[s.id for s in self.agent_card.skills]}"
        )
        return app


# ── Helpers ───────────────────────────────────────────────────────────────────

def _error_response(request_id: str, code: int, message: str) -> JSONResponse:
    """
    Build a JSON-RPC 2.0 error response.

    JSON-RPC error codes:
        -32700  Parse error
        -32600  Invalid request
        -32601  Method not found
        -32602  Invalid params
        -32603  Internal error
        -32000 to -32099  Server error (application-defined)
    """
    return JSONResponse(
        content=JSONRPCResponse(
            id    = request_id,
            error = JSONRPCError(code=code, message=message).model_dump(),
        ).model_dump()
    )


def make_completed_result(task_id: str, text: str, session_id: Optional[str] = None) -> TaskResult:
    """
    Convenience function — build a completed TaskResult with a single text artifact.

    Used by sub-agents that return a simple text answer:

        return make_completed_result(task.id, answer_text, task.sessionId)
    """
    return TaskResult(
        id        = task_id,
        sessionId = session_id,
        status    = TaskStatus(state="completed"),
        artifacts = [
            Artifact(parts=[TextPart(text=text).model_dump()])
        ],
    )


def make_failed_result(task_id: str, error: str, session_id: Optional[str] = None) -> TaskResult:
    """
    Convenience function — build a failed TaskResult with an error message.

    Used when a sub-agent catches an unrecoverable error:

        try:
            result = await pinecone_search(query)
        except Exception as e:
            return make_failed_result(task.id, str(e), task.sessionId)
    """
    return TaskResult(
        id        = task_id,
        sessionId = session_id,
        status    = TaskStatus(state="failed"),
        metadata  = {"error": error},
    )


def make_interrupt_result(
    task_id:      str,
    question:     str,
    options:      list[str],
    session_id:   str,
    allow_freetext: bool = True,
) -> TaskResult:
    """
    Convenience function — build an input-required TaskResult for HITL.

    Used by the HITL Agent when it needs user input:

        return make_interrupt_result(
            task_id    = task.id,
            question   = "Which cancer trial are you interested in?",
            options    = ["NCI-MATCH", "BASIC3", "SAFIR01", "NCI-MPACT", "iCAT"],
            session_id = task.sessionId,
        )

    The Supervisor detects input-required state and emits:
        {"type": "interrupt", "question": "...", "options": [...]}
    as an SSE event to the UI.
    """
    from .schemas import DataPart, HITLInterruptData

    interrupt_data = HITLInterruptData(
        question      = question,
        options       = options,
        session_id    = session_id,
        allow_freetext = allow_freetext,
    )

    return TaskResult(
        id        = task_id,
        sessionId = session_id,
        status    = TaskStatus(state="input-required"),
        artifacts = [
            Artifact(parts=[DataPart(data=interrupt_data.model_dump()).model_dump()])
        ],
    )