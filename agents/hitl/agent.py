"""
agents/hitl/agent.py
=====================
HITL Agent assembly.

IDENTICAL pattern to research/agent.py with these differences:
  1. tools       : clarify___ask_user_input only
  2. middleware  : HumanInTheLoopMiddleware (interrupt_on clarify tool)
  3. AGENT_NAME  : "hitl-agent"
                   → SSM: /hitl-agent/prod/bedrock/prompt_id
  4. AsyncPostgresSaver — REQUIRED for interrupt/resume across containers
                          HITL Agent is the reason Postgres exists.

WHY HumanInTheLoopMiddleware HERE AND NOT ON SUPERVISOR?
  HumanInTheLoopMiddleware fires interrupt_on for the clarify tool.
  The clarify tool lives in the HITL Agent — so the middleware must
  live here too. The Supervisor doesn't see the clarify tool directly;
  it sees the hitl_agent @tool which wraps this entire agent.

  Flow:
    Supervisor calls hitl_agent @tool
      → invoke_agent_runtime() on HITL Agent
        → HITL Agent runs create_agent with HumanInTheLoopMiddleware
          → LLM calls clarify___ask_user_input
            → HumanInTheLoopMiddleware fires interrupt_on
              → LangGraph NodeInterrupt → state saved to Postgres
                → HITL Agent yields interrupt event
                  → Supervisor's _invoke_sub_agent raises HITLInterrupt
                    → Supervisor's handler yields interrupt SSE
                      → Platform API forwards to UI → HITL card shown

WHY AsyncPostgresSaver IS CRITICAL HERE:
  When interrupt fires, container A pauses and the connection closes.
  When user picks an option, invoke_agent_runtime() creates container B.
  Container B must read the paused state from Postgres by thread_id.
  Without Postgres, container B has empty state → graph can't resume.
"""

import logging
import os
from typing import Any

import psycopg
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from core.aws import get_bedrock_prompt, init_postgres_url
from core.schema import AgentContext

log = logging.getLogger(__name__)

AGENT_NAME = os.environ.get("AGENT_NAME", "hitl-agent")


async def build_hitl_agent(tools: list[Any]) -> Any:
    """
    Assemble the HITL Agent.

    Args:
        tools: [clarify___ask_user_input] — filtered from get_mcp_tools().

    Returns:
        Compiled LangChain agent with HumanInTheLoopMiddleware.
    """
    system_prompt = get_bedrock_prompt(AGENT_NAME)
    log.info(f"[HITL Agent] Prompt loaded for: {AGENT_NAME}")

    # AsyncPostgresSaver — REQUIRED for interrupt/resume
    conn = await psycopg.AsyncConnection.connect(
        init_postgres_url(),
        autocommit=True,
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()
    log.info("[HITL Agent] Postgres checkpointer ready")

    # HumanInTheLoopMiddleware — fires NodeInterrupt when clarify tool is called
    # interrupt_on must match the exact tool name from the MCP Gateway
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
        checkpointer   = checkpointer,
        context_schema = AgentContext,
    )

    log.info(
        f"[HITL Agent] Built"
        f"  agent_name={AGENT_NAME}"
        f"  tools={[t.name for t in tools]}"
        f"  middleware=[HumanInTheLoopMiddleware]"
        f"  checkpointer=postgres"
    )
    return agent