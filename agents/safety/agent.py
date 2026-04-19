"""
agents/safety/agent.py
=======================
Safety Agent assembly.

IDENTICAL pattern to research/agent.py with these differences:
  1. tools       : [] — no MCP tools, pure LLM judge
  2. model       : "gpt-4o-mini" — same as OutputGuardrailMiddleware
                   in the single agent. Sufficient for binary YES/NO
                   faithfulness and consistency judgement. 5x cheaper
                   than gpt-4o for this simple evaluation task.
  3. AGENT_NAME  : "safety-agent"
                   → SSM: /safety-agent/prod/bedrock/prompt_id
  4. AsyncPostgresSaver — reads prior answers from thread history
                          for consistency checking across turns

WHY gpt-4o-mini AND NOT gpt-4o?
  Safety evaluation is a binary task:
    "Does this answer contradict the evidence? YES or NO."
  GPT-4o-mini scores reliably on this at ~500ms per check.
  Using gpt-4o would double the cost of every query for no
  measurable quality gain on a yes/no classification task.
  Same reasoning as OutputGuardrailMiddleware in single agent.

WHY NO TOOLS?
  Faithfulness and consistency are pure reasoning tasks.
  The Safety Agent receives the answer + evidence in the message.
  No external lookups needed — everything is in the prompt.

SYSTEM PROMPT (in Bedrock Prompt Management as "safety-agent"):
  Evaluate the following answer for:
  1. FAITHFULNESS — does any claim contradict the provided evidence?
  2. CONSISTENCY  — does the answer contradict prior answers in this conversation?

  Respond with exactly one of:
    PASSED
    BLOCKED: <brief reason>
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

AGENT_NAME = os.environ.get("AGENT_NAME", "safety-agent")


async def build_safety_agent() -> Any:
    """
    Assemble the Safety Agent.

    No tools — pure LLM evaluation using gpt-4o-mini.
    AsyncPostgresSaver for consistency checking against thread history.

    Returns:
        Compiled LangChain agent ready for astream_events().
    """
    system_prompt = get_bedrock_prompt(AGENT_NAME)
    log.info(f"[Safety Agent] Prompt loaded for: {AGENT_NAME}")

    conn = await psycopg.AsyncConnection.connect(
        init_postgres_url(),
        autocommit=True,
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()
    log.info("[Safety Agent] Postgres checkpointer ready")

    # gpt-4o-mini — sufficient for binary YES/NO evaluation
    # No tools — evaluation is pure reasoning from the message content
    agent = create_agent(
        model          = "gpt-4o-mini",
        tools          = [],
        system_prompt  = system_prompt,
        checkpointer   = checkpointer,
        context_schema = AgentContext,
    )

    log.info(
        f"[Safety Agent] Built"
        f"  agent_name={AGENT_NAME}"
        f"  model=gpt-4o-mini"
        f"  tools=none"
        f"  checkpointer=postgres"
    )
    return agent