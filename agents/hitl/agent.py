"""
agents/hitl/agent.py
=====================
HITL Agent assembly.

FIX (Issue #1 — Postgres connection per invocation):
  BEFORE: build_hitl_agent() called psycopg.AsyncConnection.connect()
          and AsyncPostgresSaver() on EVERY request. Each HITL interaction
          opened a new Postgres connection that was never explicitly closed.
          Under load this would exhaust the Postgres connection pool.

  AFTER:  build_hitl_agent() accepts a pre-built checkpointer as a parameter.
          The checkpointer is built ONCE at cold start in app.py via
          build_hitl_cold_start() and reused across all requests.

          The pattern is identical to the Supervisor's build_supervisor_cold_start():
            cold start  → build connection + checkpointer once
            per request → pass checkpointer into build_hitl_agent()

WHY AsyncPostgresSaver IS STILL CRITICAL HERE:
  When interrupt fires, container A pauses and the connection closes.
  When user picks an option, invoke_agent_runtime() creates container B.
  Container B must read the paused state from Postgres by thread_id.
  Without Postgres, container B has empty state → graph can't resume.
  The single shared connection from cold start handles both containers
  because AgentCore runs one container per agent type, not per request.

DESIGN NOTE — HITL as sub-agent vs Supervisor-native tool:
  The Archive contains TWO HITL implementations:
    1. This file (hitl/agent.py) — HITL as a dedicated sub-agent with its
       own AgentCore Runtime. The Supervisor calls it via hitl_agent @tool.
       HumanInTheLoopMiddleware fires NodeInterrupt on ask_user_input.

    2. a2a_tools.py build_ask_user_tool() — HITL as a Supervisor-native
       tool that raises HITLInterrupt directly (no sub-agent, no extra runtime).

  The Supervisor-native approach (option 2) is simpler and cheaper — no
  extra AgentCore Runtime, no A2A round-trip. Use it when:
    - The Supervisor already has the context to formulate options
    - Cost and latency matter
    - You don't need the HITL agent to independently search for candidates

  The sub-agent approach (option 1, this file) is more powerful — use it
  when the HITL agent needs its own tools (e.g. search for real trial names
  before presenting options). It's the correct choice when options must come
  from the actual database, not from the Supervisor's current context.

  Current deployment: BOTH are wired. The Supervisor's ask_user tool fires
  for simple clarifications. The hitl_agent @tool fires for database-driven
  clarifications. The Supervisor LLM picks which to use based on its prompt.
"""

import logging
import os
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from core.aws import get_bedrock_prompt, init_postgres_url
from core.schema import AgentContext

log = logging.getLogger(__name__)

AGENT_NAME = os.environ.get("AGENT_NAME", "hitl-agent")


async def build_hitl_agent(tools: list[Any], checkpointer: Any) -> Any:
    """
    Assemble the HITL Agent.

    Args:
        tools:        [clarify___ask_user_input] — filtered from get_mcp_tools().
        checkpointer: AsyncPostgresSaver built once at cold start in app.py.
                      MUST be passed in — never built here.

    Returns:
        Compiled LangChain agent with HumanInTheLoopMiddleware.
    """
    system_prompt = get_bedrock_prompt(AGENT_NAME)
    log.info(f"[HITL Agent] Prompt loaded for: {AGENT_NAME}")

    # HumanInTheLoopMiddleware — fires NodeInterrupt when clarify tool is called.
    # interrupt_on must match the exact tool name from the MCP Gateway.
    middleware = [
        HumanInTheLoopMiddleware(
            interrupt_on={"tool-hitl___ask_user_input": True},
        ),
    ]

    agent = create_agent(
        model          = "gpt-4o",
        tools          = tools,
        system_prompt  = system_prompt,
        middleware     = middleware,
        checkpointer   = checkpointer,   # ← injected, not built here
        context_schema = AgentContext,
    )

    log.info(
        f"[HITL Agent] Built"
        f"  agent_name={AGENT_NAME}"
        f"  tools={[t.name for t in tools]}"
        f"  middleware=[HumanInTheLoopMiddleware]"
        f"  checkpointer=postgres (shared)"
    )
    return agent


async def build_hitl_cold_start() -> dict:
    """
    Build all heavy objects for HITL Agent at cold start.
    Called once in app.py _ensure_agent(). Returns a dict passed to
    build_hitl_agent() on every request.

    Returns:
        {"checkpointer": AsyncPostgresSaver}

    WHY a separate function (not just inlined in app.py):
      Mirrors the Supervisor pattern (build_supervisor_cold_start) so the
      two cold start paths are structurally identical and easy to compare.
      Also makes unit testing easier — you can mock this function without
      touching app.py.
    """
    import psycopg
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    log.info("[HITL] Cold start — building Postgres checkpointer")

    conn = await psycopg.AsyncConnection.connect(
        init_postgres_url(),
        autocommit=True,
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()   # idempotent — creates tables if missing

    log.info("[HITL] Cold start complete — Postgres checkpointer ready")
    return {"checkpointer": checkpointer}