"""
core/mcp_client.py — MCP Gateway Tool Client
=============================================
Moved to core/ so ALL agents share it without duplication.

ONLY CHANGE from vs-agentcore-platform-aws/agent/tools/mcp_client.py:
  SSM_PREFIX default: /vs-agentcore/prod  →  /vs-agentcore-multiagent/prod

Everything else is identical — AwsSigV4, MultiServerMCPClient,
get_mcp_tools(), transport, signing flow, tool naming.

USAGE IN EACH AGENT'S agent.py:
  from core.mcp_client import get_mcp_tools

  all_tools = await get_mcp_tools()

  # Research Agent — search + synthesise
  tools = [t for t in all_tools if t.name in {
      "tool-search___search_tool",
      "tool-summariser___summariser_tool",
  }]

  # Knowledge Agent — graph + synthesise
  tools = [t for t in all_tools if t.name in {
      "tool-graph___graph_tool",
      "tool-summariser___summariser_tool",
  }]

  # HITL Agent — clarify only
  tools = [t for t in all_tools if t.name in {
      "clarify___ask_user_input",
  }]

  # Chart Agent — search + synthesise + chart
  tools = [t for t in all_tools if t.name in {
      "tool-search___search_tool",
      "tool-summariser___summariser_tool",
      "chart___chart_tool",
  }]

  # Safety Agent — no MCP tools (pure LLM judge)
  tools = []

MCP GATEWAY SSM PATH:
  /vs-agentcore-multiagent/prod/mcp/gateway_url
  Same gateway serves all 6 agents — routing is by tool name.

TOOLS RETURNED (5 in multi-agent, was 4 in single agent):
  tool-search___search_tool         — Pinecone semantic search
  tool-graph___graph_tool           — Neo4j Cypher
  clarify___ask_user_input          — HITL interrupt
  tool-summariser___summariser_tool — GPT-4o synthesis
  chart___chart_tool                — Chart.js generation (NEW)

FULL CALL PATH (unchanged):
  create_agent() tool call
    → StructuredTool.invoke()
      → httpx POST signed by AwsSigV4
        → Bedrock MCP Gateway
          → Lambda handler
            → ToolMessage back into agent

For full docs on SigV4 + MCP protocol:
  vs-agentcore-platform-aws/agent/tools/mcp_client.py
"""

import logging
import os

import boto3
import httpx
from botocore.auth import SigV4Auth as BotoSigV4Auth
from botocore.awsrequest import AWSRequest
from langchain_mcp_adapters.client import MultiServerMCPClient

log = logging.getLogger(__name__)

REGION     = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")  # ← updated


class AwsSigV4(httpx.Auth):
    """
    Signs every MCP Gateway HTTP request with AWS SigV4.
    Identical to single agent — see vs-agentcore-platform-aws for full docs.
    """

    def auth_flow(self, request: httpx.Request):
        creds = boto3.Session().get_credentials().get_frozen_credentials()

        aws_req = AWSRequest(
            method  = request.method,
            url     = str(request.url),
            data    = request.content,
            headers = dict(request.headers),
        )

        BotoSigV4Auth(creds, "bedrock-agentcore", REGION).add_auth(aws_req)

        for key, value in aws_req.headers.items():
            request.headers[key] = value

        yield request


async def get_mcp_tools() -> list:
    """
    Connect to the Bedrock MCP Gateway and return ALL tool definitions
    as LangChain StructuredTool objects.

    Each agent.py filters this list to only the tools it needs.
    Gateway URL read from SSM:
      /vs-agentcore-multiagent/prod/mcp/gateway_url
    """
    ssm         = boto3.client("ssm", region_name=REGION)
    gateway_url = ssm.get_parameter(
        Name=f"{SSM_PREFIX}/mcp/gateway_url"
    )["Parameter"]["Value"]

    log.info(f"[MCP] Connecting to gateway: {gateway_url}")

    client = MultiServerMCPClient({
        "vs-agentcore-multiagent-tools": {
            "transport": "streamable_http",
            "url":       gateway_url,
            "auth":      AwsSigV4(),
        }
    })

    tools = await client.get_tools()
    log.info(f"[MCP] Tools discovered: {[t.name for t in tools]}")
    return tools
