"""
a2a_tools/__init__.py
======================
Public API for the a2a_tools package.

PACKAGE STRUCTURE
-----------------
  hitl.py      — HITLInterrupt exception + ask_user_input tool
  registry.py  — SSM agent registry + runtime ARN loading
  parser.py    — SSE event parser (one parser for all sub-agents)
  invoke.py    — async httpx invocation + SigV4 signing
  tools.py     — StructuredTool factory (what app.py and agent.py use)

WHAT TO IMPORT FROM WHERE
--------------------------
From outside this package, you only need:
  from agents.supervisor.a2a_tools import HITLInterrupt      — for exception handling
  from agents.supervisor.a2a_tools import build_a2a_tools    — for agent assembly

Everything else (invoke, parse, registry) is internal to this package.
"""
from agents.supervisor.a2a_tools.hitl     import HITLInterrupt, build_ask_user_tool
from agents.supervisor.a2a_tools.tools    import build_a2a_tools
from agents.supervisor.a2a_tools.registry import get_agent_registry, get_runtime_arns
from agents.supervisor.a2a_tools.invoke   import invoke_sub_agent

__all__ = [
    # Used by app.py and streaming.py for exception handling
    "HITLInterrupt",
    # Used by agent.py to assemble the supervisor's tool list
    "build_a2a_tools",
    # Used by build_ask_user_tool (internal) — exported for completeness
    "build_ask_user_tool",
    # Useful for debugging or health checks
    "get_agent_registry",
    "get_runtime_arns",
    "invoke_sub_agent",
]