"""
agents/safety/app.py
===================
Safety Agent — faithfulness + consistency guardrail evaluation via GPT-4o-mini

AgentCore Runtime entrypoint.
Exposes A2A routes via shared.a2a.server.A2AServer.

TODO: implement
  1. Define AGENT_CARD with skills
  2. Register task handler
  3. Run LangGraph graph from graph.py
"""
