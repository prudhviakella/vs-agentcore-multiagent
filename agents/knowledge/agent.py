"""
agents/knowledge/agent.py
==========================
Knowledge Agent assembly.

IDENTICAL pattern to research/agent.py with these differences:
  1. tools       : graph_tool + summariser_tool
  2. AGENT_NAME  : "knowledge-agent"
                   → SSM: /knowledge-agent/prod/bedrock/prompt_id
  3. No middleware (same as all sub-agents)
  4. AsyncPostgresSaver (same as all agents)

RESPONSIBILITY:
  The Knowledge Agent writes and executes Cypher queries against Neo4j.
  It answers structural questions about the graph:
    - Trial discovery   : "What trials exist for colorectal cancer?"
    - Relationship lookup: "What drugs does NCT04368728 use?"
    - Sponsor queries   : "Who sponsors the NCI-MATCH study?"
    - Geographic queries: "Which trials are conducted in Germany?"

  graph_tool returns raw Neo4j rows.
  summariser_tool formats them into readable prose before returning.

GRAPH SCHEMA AVAILABLE TO THIS AGENT (via system prompt):
  (Trial)-[:TARGETS]-------> (Disease)
  (Trial)-[:USES]----------> (Drug)
  (Trial)-[:SPONSORED_BY]--> (Sponsor)
  (Trial)-[:CONDUCTED_IN]--> (Country)
  (Trial)-[:LOCATED_AT]----> (Site)
  (Trial)-[:MEASURES]------> (Outcome)
  (Trial)-[:INCLUDES]------> (PatientPopulation)
  (Trial)-[:ASSOCIATED_WITH]>(MeSHTerm)
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

AGENT_NAME = os.environ.get("AGENT_NAME", "knowledge-agent")


async def build_knowledge_agent(tools: list[Any]) -> Any:
    """
    Assemble the Knowledge Agent.

    Args:
        tools: Filtered list from get_mcp_tools() — graph + summariser only.

    Returns:
        Compiled LangChain agent ready for astream_events().
    """
    system_prompt = get_bedrock_prompt(AGENT_NAME)
    log.info(f"[Knowledge Agent] Prompt loaded for: {AGENT_NAME}")

    conn = await psycopg.AsyncConnection.connect(
        init_postgres_url(),
        autocommit=True,
    )
    checkpointer = AsyncPostgresSaver(conn)
    await checkpointer.setup()
    log.info("[Knowledge Agent] Postgres checkpointer ready")

    agent = create_agent(
        model          = "gpt-4o",
        tools          = tools,
        system_prompt  = system_prompt,
        checkpointer   = checkpointer,
        context_schema = AgentContext,
    )

    log.info(
        f"[Knowledge Agent] Built"
        f"  agent_name={AGENT_NAME}"
        f"  tools={[t.name for t in tools]}"
        f"  checkpointer=postgres"
        f"  middleware=none"
    )
    return agent