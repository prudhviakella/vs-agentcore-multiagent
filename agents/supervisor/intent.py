"""
agents/supervisor/intent.py
============================
Intent classifier — decides which sub-agents to call.

Intent types:
  specific   → Research Agent + Knowledge Agent (parallel)
  vague      → Research Agent (candidates) → HITL Agent → resume
  chart      → Research Agent → Chart Agent
  list       → Knowledge Agent only
  safety     → always appended after any answer

TODO: implement
  - classify(query: str) -> Intent
  - get_agents_for_intent(intent: Intent) -> list[str]
"""
