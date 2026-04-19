"""
agents/research/agent.py
=========================
Research Agent assembly.

IDENTICAL pattern to vs-agentcore-platform-aws/agent/agent.py with
these differences:
  1. No middleware  — sub-agents are lean, Supervisor owns middleware
  2. No PineconeStore / SemanticCache — no episodic memory, no cache
  3. No safety_llm  — no output guardrail here
  4. system_prompt  — loaded via get_bedrock_prompt(AGENT_NAME)
                      AGENT_NAME="research-agent" set by Terraform
                      reads /research-agent/prod/bedrock/prompt_id from SSM

WHY AsyncPostgresSaver (not MemorySaver)?
  AgentCore creates a FRESH container for each invoke_agent_runtime() call.
  MemorySaver lives in-process — state is lost when the container is recycled.
  AsyncPostgresSaver persists state to RDS by thread_id so:
    - Multi-turn conversations work across container invocations
    - Research Agent knows what it retrieved earlier in the same session
    - After a HITL resume, Supervisor calls Research again — it reads
      the thread history from Postgres and has full context
  Same RDS instance as single agent — each sub-agent is just another
  connection pool on the same database.
"""

import logging
import os
from typing import Any

import psycopg
from langchain.agents import create_agent
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from core.aws import get_bedrock_prompt, init_postgres_url
from core.schema import AgentContext

log = logging.getLogger(__name__)

AGENT_NAME = os.environ.get("AGENT_NAME", "research-agent")


async def build_research_agent(tools: list[Any]) -> Any:
    """
    Assemble the Research Agent.

    Args:
        tools: Filtered list from get_mcp_tools() — search + summariser only.
               Passed in from app.py after cold start tool discovery.

    Returns:
        Compiled LangChain agent ready for astream_events().
    """
    # Fetch versioned system prompt from Bedrock Prompt Management.
    # AGENT_NAME drives which prompt is loaded — same call as single agent.
    system_prompt = get_bedrock_prompt(AGENT_NAME)
    log.info(f"[Research Agent] Prompt loaded for: {AGENT_NAME}")

    # AsyncPostgresSaver — same as single agent.
    # Persists thread state across fresh container invocations.
    conn = await psycopg.AsyncConnection.connect(
        init_postgres_url(),
        autocommit=True,
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()
    log.info("[Research Agent] Postgres checkpointer ready")

    # create_agent — same call as single agent, no middleware
    agent = create_agent(
        model          = "gpt-4o",
        tools          = tools,
        system_prompt  = system_prompt,
        checkpointer   = checkpointer,
        context_schema = AgentContext,
    )

    log.info(
        f"[Research Agent] Built"
        f"  agent_name={AGENT_NAME}"
        f"  tools={[t.name for t in tools]}"
        f"  checkpointer=postgres"
        f"  middleware=none"
    )
    return agent