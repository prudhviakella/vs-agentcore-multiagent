"""
supervisor/cold_start.py
=========================
Cold start initialisation — runs ONCE per container lifetime.

WHAT IS A COLD START?
----------------------
When AWS spins up a new container for the supervisor agent, the first
request triggers the cold start. This is where we:
  1. Load secrets from AWS Secrets Manager (OpenAI key, LangSmith key)
  2. Build the expensive shared objects (Postgres connection, Pinecone
     index, semantic cache, LangGraph compiled graph)

After cold start, all subsequent requests reuse these objects at zero cost.
Cold start takes ~3-5 seconds. Warm requests take ~10ms for this step.

WHY A GLOBAL VARIABLE?
-----------------------
_cold_start_objects is a module-level global (None until first request).
This is the correct pattern for AgentCore because:
  - AgentCore runs handler() as a coroutine in a single Python process
  - asyncio is single-threaded — only one coroutine runs at a time
  - No race condition is possible, so no lock is needed
  - A singleton class would add complexity with no benefit here

WHY NOT BUILD EVERYTHING IN handler()?
---------------------------------------
If we built a Postgres connection on every request:
  - Connection setup:  ~500ms
  - Pinecone init:     ~300ms
  - Graph compilation: ~200ms
  - Total wasted:      ~1s per request, every request
With cold start this cost is paid ONCE for the container lifetime.
"""
import json
import logging
import os

import boto3

from agents.supervisor.agent        import build_supervisor_cold_start
from agents.supervisor.logging_setup import setup_cloudwatch

log = logging.getLogger(__name__)

# None = not yet initialised. Set to the dict returned by build_supervisor_cold_start()
# on the first request. All subsequent requests check this and return immediately.
_cold_start_objects = None


def _load_langsmith_from_ssm() -> None:
    """
    Load LangSmith API key from Secrets Manager and write to os.environ.

    WHY SECRETS MANAGER, NOT ENVIRONMENT VARIABLES?
    ------------------------------------------------
    API keys must never be baked into container images or ECS task definitions
    — they would appear in CloudTrail logs and AWS console in plaintext.
    Secrets Manager encrypts them at rest and in transit.

    We write to os.environ so LangChain's built-in tracing picks up
    LANGSMITH_API_KEY automatically — no code changes needed in LangChain.

    WHY CALLED AT COLD START, NOT MODULE IMPORT?
    ---------------------------------------------
    At module import time, the container may not have AWS credentials yet.
    AgentCore injects the IAM role credentials just before handler() is called.
    Cold start runs inside handler(), so credentials are always available.

    SHORT-CIRCUITS if LANGSMITH_API_KEY is already in the environment
    (useful for local development where you export keys manually).
    """
    if os.environ.get("LANGSMITH_API_KEY"):
        return  # already set — skip Secrets Manager call
    try:
        prefix = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
        sm     = boto3.client("secretsmanager", region_name="us-east-1")
        # Secret is stored as JSON: {"api_key": "lsv2_...", "project": "...", "tracing": "true"}
        secret = json.loads(sm.get_secret_value(SecretId=f"{prefix}/langsmith")["SecretString"])
        os.environ["LANGSMITH_API_KEY"] = secret.get("api_key", "")
        os.environ["LANGSMITH_PROJECT"]  = secret.get("project", "langchain-agent-experiments")
        os.environ["LANGSMITH_TRACING"]  = secret.get("tracing", "true")
        log.info("[ColdStart] LangSmith credentials loaded")
    except Exception as e:
        # Non-fatal — LangSmith is observability, not core functionality.
        # Platform still works without tracing.
        log.warning(f"[ColdStart] LangSmith not available — tracing disabled: {e}")


async def ensure_cold_start() -> dict:
    """
    Return shared objects, building them on first call.

    RETURN VALUE
    ------------
    A dict passed as **kwargs into build_supervisor_agent() on every request:
      {
        "checkpointer": AsyncPostgresSaver,  — HITL pause/resume + history
        "store":        PineconeStore,       — episodic memory read/write
        "cache":        SemanticCache,       — avoids duplicate sub-agent calls
      }

    WHAT build_supervisor_cold_start() BUILDS
    ------------------------------------------
    See agents/supervisor/agent.py for full details. In summary:
      - OpenAI embedding model (text-embedding-3-small)
      - Pinecone index connection (used by both store and cache)
      - PineconeStore for episodic memory
      - SemanticCache (similarity threshold 0.97 — high for clinical precision)
      - AsyncPostgresSaver — LangGraph checkpoint backend for HITL
      - Runs checkpointer.setup() to create LangGraph tables if missing
    """
    global _cold_start_objects
    if _cold_start_objects is not None:
        return _cold_start_objects  # warm request — return immediately

    log.info("[ColdStart] First request — building shared objects...")

    # Phase 2 logging: attach CloudWatch handler now that IAM credentials
    # have been injected by AgentCore. Called here (not at import time)
    # because boto3.client('logs') needs credentials to succeed.
    setup_cloudwatch()

    # Load OpenAI key from Secrets Manager if not already set in env.
    # Must be set BEFORE build_supervisor_cold_start() because the LLM
    # client is initialised during graph compilation.
    if not os.environ.get("OPENAI_API_KEY"):
        prefix = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")
        sm     = boto3.client("secretsmanager", region_name="us-east-1")
        secret = json.loads(sm.get_secret_value(SecretId=f"{prefix}/openai")["SecretString"])
        os.environ["OPENAI_API_KEY"] = secret.get("api_key", "")
        log.info("[ColdStart] OpenAI key loaded")

    # Load LangSmith BEFORE building the graph so LangChain picks up the
    # LANGSMITH_API_KEY env var during graph compilation and wires up tracing.
    _load_langsmith_from_ssm()

    # Build all expensive shared objects (Postgres, Pinecone, cache, graph).
    # This is the slow part (~3-5s). Everything after this is fast.
    _cold_start_objects = await build_supervisor_cold_start()
    log.info("[ColdStart] Complete — all shared objects ready")

    return _cold_start_objects