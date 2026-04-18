"""
shared/a2a/server.py
====================
Base A2A FastAPI server — every sub-agent imports and uses this.

Exposes:
  GET  /.well-known/agent.json  — Agent Card discovery
  POST /a2a                      — JSON-RPC 2.0 task handling
  GET  /health                   — Health check

TODO: implement
"""
