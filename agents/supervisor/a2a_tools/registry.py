"""
a2a_tools/registry.py
======================
Agent registry and runtime ARN loading from AWS SSM.

WHAT IS THE AGENT REGISTRY?
-----------------------------
The registry is a JSON list stored in SSM Parameter Store:
  [
    {"name": "knowledge", "description": "Neo4j graph queries...", "port": 8002},
    {"name": "research",  "description": "Pinecone semantic search...", "port": 8001},
    {"name": "chart",     "description": "Chart.js generation...", "port": 8005},
  ]

The supervisor reads this at cold start to know which sub-agents exist
and what they do. The descriptions become part of the tool schemas that
gpt-5.5 reads to decide which agent to call for each query.

NOTE: safety agent is intentionally excluded from the registry.
OutputGuardrailMiddleware handles output safety via Bedrock Guardrails.

WHAT IS A RUNTIME ARN?
-----------------------
Every AgentCore runtime gets an ARN (Amazon Resource Name) when deployed:
  arn:aws:bedrock-agentcore:us-east-1:107282186797:runtime/vs_agentcore_ma_knowledge-Xxxx

The supervisor uses this ARN to invoke the sub-agent via boto3:
  client.invoke_agent_runtime(agentRuntimeArn=arn, ...)

ARNs are stored in SSM by deploy.py when each agent is deployed:
  /vs-agentcore-multiagent/prod/agents/knowledge/runtime_arn

WHY lru_cache FOR REGISTRY BUT NOT FOR ARNs?
---------------------------------------------
Registry (lru_cache): The list of agents and their descriptions is stable —
it only changes when you add a new agent type, which requires a code change
and container restart anyway. Caching avoids one SSM call per request.

ARNs (no cache): ARNs change every time an agent is redeployed (AgentCore
generates a new suffix). If we cached ARNs, a redeployed sub-agent would
be unreachable until the supervisor also restarted. Fetching fresh on every
request (~5-10ms per agent) is the correct trade-off during development
when agents are redeployed frequently.
"""
import json
import logging
import os
from functools import lru_cache

import boto3

log        = logging.getLogger(__name__)
REGION     = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")


@lru_cache(maxsize=1)
def get_agent_registry() -> list:
    """
    Load the agent registry from SSM. Cached for the container lifetime.

    The registry tells the supervisor:
      - Which sub-agents exist (name)
      - What each one does (description → becomes tool schema for gpt-5.5)
      - What port it runs on locally (port, used for local testing only)

    lru_cache(maxsize=1) means this SSM call happens ONCE per container.
    All subsequent calls return the cached list instantly.

    To force a refresh: restart the container (or call get_agent_registry.cache_clear())
    """
    ssm = boto3.client("ssm", region_name=REGION)
    try:
        value    = ssm.get_parameter(Name=f"{SSM_PREFIX}/agents/registry")["Parameter"]["Value"]
        registry = json.loads(value)
        log.info(f"[Registry] Loaded {len(registry)} agents: {[a['name'] for a in registry]}")
        return registry
    except ssm.exceptions.ParameterNotFound:
        raise RuntimeError(
            f"Agent registry not found at {SSM_PREFIX}/agents/registry. "
            "Run: python deploy.py registry"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load agent registry from SSM: {exc}")


def get_runtime_arns() -> dict:
    """
    Fetch current runtime ARNs for all registered agents from SSM.

    NOT cached — ARNs change on every agent redeploy.

    Returns
    -------
    dict mapping agent name → runtime ARN
    e.g. {"knowledge": "arn:aws:bedrock-agentcore:...:runtime/vs_...-Xxxx"}

    Missing ARNs are logged as warnings (not errors) so one broken agent
    doesn't prevent the others from working.
    """
    ssm      = boto3.client("ssm", region_name=REGION)
    registry = get_agent_registry()
    arns     = {}

    for agent in registry:
        name = agent["name"]
        try:
            arn = ssm.get_parameter(
                Name=f"{SSM_PREFIX}/agents/{name}/runtime_arn"
            )["Parameter"]["Value"]
            arns[name] = arn
            log.info(f"[Registry] ARN loaded for {name}")
        except Exception as exc:
            # Warning not error — supervisor can still serve requests using
            # other agents. The missing agent's tool will raise RuntimeError
            # when called, which LangGraph handles as a tool error.
            log.warning(f"[Registry] No ARN for {name}: {exc}")

    return arns