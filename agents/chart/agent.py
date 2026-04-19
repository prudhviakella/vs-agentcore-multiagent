"""
agents/chart/agent.py
======================
Chart Agent assembly.

IDENTICAL pattern to research/agent.py with these differences:
  1. tools       : search_tool + summariser_tool + tool-chart___chart_tool
  2. AGENT_NAME  : "chart-agent"
                   → SSM: /chart-agent/prod/bedrock/prompt_id
  3. AsyncPostgresSaver (same as all agents)

RESPONSIBILITY:
  The Chart Agent is called when the Supervisor detects numerical or
  comparative queries that would benefit from visualisation.

TOOL SEQUENCE (in system prompt):
  Step 1 → search_tool       : retrieve numerical evidence from Pinecone
  Step 2 → summariser_tool   : extract and structure the numbers
  Step 3 → tool-chart___chart_tool: generate Chart.js config from the numbers

SYSTEM PROMPT (in Bedrock as "chart-agent") instructs:
  - Always search first to get the raw data
  - Extract specific numbers (percentages, counts, dates)
  - Choose the right chart type:
      bar       → comparisons between trials/drugs
      line      → trends over time
      pie       → proportional distributions (e.g. Phase 1/2/3/4 split)
      doughnut  → same as pie with better readability
  - Call tool-chart___chart_tool with the structured data
  - Return a text summary alongside the chart

tool-chart___chart_tool (MCP Gateway → chart_lambda):
  Input:  {"data": {...}, "chart_type": "bar", "title": "..."}
  Output: {"chart": {...Chart.js config...}, "chart_type": "bar"}
  The chart config is yielded as a special event in app.py:
    {"type": "chart", "config": {...}, "chart_type": "bar"}
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

AGENT_NAME = os.environ.get("AGENT_NAME", "chart-agent")


async def build_chart_agent(tools: list[Any]) -> Any:
    """
    Assemble the Chart Agent.

    Args:
        tools: Filtered list — search + summariser + chart tool.

    Returns:
        Compiled LangChain agent ready for astream_events().
    """
    system_prompt = get_bedrock_prompt(AGENT_NAME)
    log.info(f"[Chart Agent] Prompt loaded for: {AGENT_NAME}")

    conn = await psycopg.AsyncConnection.connect(
        init_postgres_url(),
        autocommit=True,
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()
    log.info("[Chart Agent] Postgres checkpointer ready")

    agent = create_agent(
        model          = "gpt-4o",
        tools          = tools,
        system_prompt  = system_prompt,
        checkpointer   = checkpointer,
        context_schema = AgentContext,
    )

    log.info(
        f"[Chart Agent] Built"
        f"  agent_name={AGENT_NAME}"
        f"  tools={[t.name for t in tools]}"
        f"  checkpointer=postgres"
        f"  middleware=none"
    )
    return agent