# VS AgentCore — Multi-Agent Platform (A2A)

> **Vidya Sankalp · Applied GenAI Engineering**
> Multi-agent clinical trial research platform built on AWS Bedrock AgentCore
> using the A2A (Agent-to-Agent) protocol.

## Architecture

Supervisor + 5 specialist sub-agents communicating via A2A JSON-RPC 2.0.

```
User → Platform API → Supervisor Agent
                          ├── A2A → Research Agent   (Pinecone search + synthesis)
                          ├── A2A → Knowledge Agent  (Neo4j graph queries)
                          ├── A2A → HITL Agent       (clarification + interrupt)
                          ├── A2A → Safety Agent     (guardrail evaluation)
                          └── A2A → Chart Agent      (Chart.js generation)
```

## Project Structure

```
vs-agentcore-multiagent/
├── agents/
│   ├── supervisor/     # Intent classifier + A2A router + middleware
│   ├── research/       # Pinecone semantic search + evidence synthesis
│   ├── knowledge/      # Neo4j graph traversal + trial discovery
│   ├── hitl/           # Clarification card + NodeInterrupt + resume
│   ├── safety/         # Faithfulness + consistency guardrail evaluation
│   └── chart/          # Chart.js generation from numerical answers
├── shared/
│   ├── a2a/
│   │   ├── schemas.py  # A2A protocol Pydantic models (AgentCard, Task, Skill...)
│   │   ├── client.py   # A2A HTTP client (used by Supervisor)
│   │   └── server.py   # A2A FastAPI base server (used by all sub-agents)
│   ├── middleware/     # SemanticCache, EpisodicMemory, Tracer, Guardrail...
│   └── config.py       # SSM + Secrets Manager loader
├── infra/
│   ├── main.tf         # 6 AgentCore Runtimes + A2A Gateway
│   └── variables.tf
├── scripts/
│   └── deploy.sh       # Build + push + deploy all agents
└── tests/
    ├── test_a2a_schemas.py
    ├── test_supervisor.py
    ├── test_a2a_flow.py
    └── test_chart_agent.py
```

## Quick Start

```bash
# 1. Clone
git clone <repo-url>
cd vs-agentcore-multiagent

# 2. Configure
cp .env.example .env.prod
# Fill in .env.prod

# 3. Deploy infra
./scripts/deploy.sh infra

# 4. Deploy all agents (sub-agents first, supervisor last)
./scripts/deploy.sh all
```

## A2A Protocol

Each sub-agent publishes an Agent Card at:
```
GET https://<agent-url>/.well-known/agent.json
```

The Supervisor discovers all agents at cold start by reading their Agent Cards,
then routes tasks based on Skills and intent classification.

See `shared/a2a/schemas.py` for full A2A type definitions.

## Related

- `vs-agentcore-platform-aws` — single-agent platform (predecessor to this repo)
