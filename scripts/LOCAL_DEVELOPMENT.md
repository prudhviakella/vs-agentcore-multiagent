# VS AgentCore Multi-Agent — Local Development Guide

Run all 5 agents + Supervisor + Platform locally for development and testing.
No Docker required. All agents connect to real AWS services (Pinecone, Neo4j, RDS, Bedrock).

Both `local.py` and `deploy.py` live in `scripts/` — always run them from there.

---

## Prerequisites

- **Python 3.11+** — macOS system Python is 3.8 and will not work (see setup below)
- **AWS credentials** configured (`aws configure`)
- **boto3** installed (`python3.11 -m pip install boto3`)

---

## One-time Setup

### 1. Verify Python version

```bash
python3.11 --version   # must print Python 3.11.x or higher
```

If `python3.11` is not found:

```bash
brew install python@3.11
echo 'export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
python3.11 --version   # verify
```

> **Why 3.11?** `langchain>=0.3`, `langgraph`, and `bedrock-agentcore` all require Python ≥3.9.
> `langgraph-checkpoint-postgres` requires ≥3.10. The macOS system `python3` is 3.8 and
> pip will silently install nothing useful.

### 2. Install dependencies

```bash
cd ~/PycharmProjects/vs-agentcore-multiagent
python3.11 -m pip install -r requirements_local.txt
```

### 3. Configure environment

```bash
cp .env.prod.template .env.prod
# Edit .env.prod with your actual values
```

Required values in `.env.prod`:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=<your-account-id>
export SSM_PREFIX=/vs-agentcore-multiagent/prod
export AGENT_ENV=prod
export OPENAI_API_KEY=sk-...
export PINECONE_API_KEY=pcsk_...
export PINECONE_INDEX_NAME=clinical-agent
export NEO4J_URI=neo4j+s://...
export NEO4J_USER=...
export NEO4J_PASSWORD=...
export PLATFORM_API_KEY=vs-...
export POSTGRES_URL=postgresql://...
```

> **Windows users:** No `source` command needed — `local.py` parses `.env.prod` directly.

### 4. Deploy required AWS resources (first time only)

These are lightweight deploy steps — no Docker or Terraform needed.
Run from `scripts/`:

```bash
cd scripts

python3.11 deploy.py prompts      # upload agent prompts to Bedrock Prompt Management
python3.11 deploy.py guardrails   # create Bedrock Guardrail (safety filters)
python3.11 deploy.py gateway      # create/update MCP Gateway and targets
python3.11 deploy.py secrets      # write SSM config params (trace table, Pinecone index)
python3.11 deploy.py registry     # write agent registry to SSM
```

### 5. Verify everything is ready

```bash
python3.11 local.py preflight
```

All items should show ✅ before proceeding. The preflight checks:

| Check | What it verifies |
|---|---|
| Secrets Manager | openai, pinecone, neo4j, postgres, platform_api_key |
| Bedrock Prompts | prompt_id + prompt_version in SSM for all 5 agents |
| MCP Gateway | gateway URL in SSM |
| Agent Registry | agent list in SSM (supervisor reads this at cold start) |
| Bedrock Guardrail | guardrail_id + guardrail_version in SSM |
| Misc SSM params | trace table name, Pinecone index name |
| Connectivity | Postgres TCP, Pinecone API key present |

---

## Running Locally

All commands run from `scripts/`:

```bash
cd ~/PycharmProjects/vs-agentcore-multiagent/scripts

# Start all agents + platform API
python3.11 local.py start

# Agents only — no platform (call agents directly)
python3.11 local.py start --no-platform

# Skip preflight check (if you know resources exist)
python3.11 local.py start --skip-preflight

# Custom env file
python3.11 local.py start --env ../.env.staging

# Check which services are running
python3.11 local.py status
```

`local.py start` replaces the old 7-terminal-tab process:
1. Parses `.env.prod` cross-platform (no `source` needed)
2. Runs pre-flight against all required AWS resources
3. Starts research, knowledge, safety, chart agents in parallel
4. Waits for all 4 sub-agents to be healthy before starting supervisor
5. Starts platform API last
6. Streams all agent logs to one terminal with coloured `[agent_name]` prefixes
7. Shuts everything down cleanly on Ctrl+C

---

## Port Reference

| Service | Port | Start command |
|---|---|---|
| Platform API | 8080 | `local.py start` |
| Supervisor | 8000 | `local.py start` |
| Research | 8001 | `local.py start` |
| Knowledge | 8002 | `local.py start` |
| Safety | 8003 | `local.py start` |
| Chart | 8005 | `local.py start` |

---

## Testing

All commands run from `scripts/`:

### Via local.py (recommended)

```bash
python3.11 local.py test           # run all test queries
python3.11 local.py test research  # mRNA-1273 Phase 3 results
python3.11 local.py test knowledge # cancer trials from graph
python3.11 local.py test chart     # COVID vaccine efficacy chart
python3.11 local.py test hitl      # vague query → HITL → resume
python3.11 local.py test direct    # bypass platform, call agents directly
```

### Via curl (Mac/Linux)

Run from the project root (not `scripts/`):

```bash
source .env.prod

# Research query
curl -s -X POST http://localhost:8080/api/v1/clinical-trial/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $PLATFORM_API_KEY" \
  -d '{"message":"What are Phase 3 results for mRNA-1273?","thread_id":"test-001","domain":"pharma"}'

# HITL query — triggers clarification card
curl -s -X POST http://localhost:8080/api/v1/clinical-trial/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $PLATFORM_API_KEY" \
  -d '{"message":"Show me cancer trials","thread_id":"test-hitl-001","domain":"pharma"}'

# Resume after HITL
curl -s -X POST http://localhost:8080/api/v1/clinical-trial/resume \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $PLATFORM_API_KEY" \
  -d '{"thread_id":"test-hitl-001","user_answer":"NCT02788279","domain":"pharma"}'

# Chart query
curl -s -X POST http://localhost:8080/api/v1/clinical-trial/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $PLATFORM_API_KEY" \
  -d '{"message":"Compare efficacy across COVID-19 vaccine trials and show me a chart","thread_id":"test-chart-001","domain":"pharma"}'
```

### Directly to sub-agents (bypass platform and supervisor)

```bash
# Research Agent
curl -s -X POST http://localhost:8001/invocations \
  -H "Content-Type: application/json" \
  -d '{"message":"What are Phase 3 results for mRNA-1273?","session_id":"r-001","domain":"pharma"}'

# Knowledge Agent
curl -s -X POST http://localhost:8002/invocations \
  -H "Content-Type: application/json" \
  -d '{"message":"What trials exist for colorectal cancer?","session_id":"k-001","domain":"pharma"}'

# Chart Agent
curl -s -X POST http://localhost:8005/invocations \
  -H "Content-Type: application/json" \
  -d '{"message":"Compare efficacy: mRNA-1273 94.1%, BNT162b2 95%","session_id":"c-001","domain":"pharma"}'
```

---

## Observability Dashboard

Open in browser after starting:

```
http://localhost:8080/observability
```

**Filter pills** (top-right) — Supervisor shows all end-to-end traces; Research / Knowledge / Chart show traces that called that specific sub-agent.

**Trace detail panel** (click any row):
- Agent waterfall — per-agent latency bars
- Full question + answer, token counts, cost per turn
- Tool calls with query and response
- HITL question, options presented, user selection
- Guardrail verdict (PASSED / BLOCKED + reason)

**Inspect from terminal:**

```bash
cd scripts

python3.11 ../scripts/inspect_traces.py                           # last 50 traces
python3.11 ../scripts/inspect_traces.py --limit 5                 # last 5
python3.11 ../scripts/inspect_traces.py --run-id fresh-010        # specific trace
python3.11 ../scripts/inspect_traces.py --agent supervisor-agent  # filter by agent
```

---

## SSE Event Reference

All agents stream Server-Sent Events. Event types:

| Event type | Description |
|---|---|
| `tool_start` | MCP tool call started — `name` = `tool-search___search_tool` etc. |
| `tool_end` | MCP tool call completed |
| `token` | Streaming LLM token — `content` field |
| `chart` | Chart.js config — `config` + `chart_type` fields |
| `interrupt` | HITL pause — `question` + `options` fields |
| `span` | Sub-agent observability span (internal, not shown in UI) |
| `error` | Error — `message` field |
| `done` | Request complete — `latency_ms` field |

---

## Data Accuracy Note

The Pinecone knowledge base contains **mRNA-1273 (Moderna COVE trial)** data.
Queries about BNT162b2 (Pfizer) will return mRNA-1273 results with a note.

Reliable test queries:
```
"What are Phase 3 results for mRNA-1273?"
"Show me cancer trials"
"Compare efficacy across COVID-19 vaccine trials and show me a chart"
"Which trials target breast cancer?"
```

---

## Reset / Clean Up

```bash
cd scripts

# Clear Postgres HITL checkpoints + DynamoDB traces
python3.11 local.py clean
```

Always use a **new `thread_id`** for each fresh test run to avoid replaying stale checkpoints.

---

## Adding a New Agent

1. Create the agent under `agents/{name}/`
2. Deploy its AgentCore Runtime: `python3.11 deploy.py agents`
3. Add an entry to `step_registry()` in `deploy.py`
4. Push the registry: `python3.11 deploy.py registry`
5. Restart Supervisor — it reads the registry from SSM at cold start and builds tools dynamically. No code changes required.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ERROR: No matching distribution found for langchain>=0.3.0` | Running Python 3.8 (macOS system default) | `brew install python@3.11` then use `python3.11` |
| `prompts/ directory not found` | Running `deploy.py` from project root instead of `scripts/` | `cd scripts` then re-run |
| `{"detail": "Invalid API key"}` | `PLATFORM_API_KEY` not loaded | Check `.env.prod` — `local.py` loads it automatically |
| `ParameterNotFound` | `AGENT_ENV` not set | Add `export AGENT_ENV=prod` to `.env.prod` |
| Pre-flight fails on prompts / guardrail / gateway | AWS resources not deployed yet | Run `python3.11 deploy.py prompts` etc. (see Step 4) |
| `Address already in use` | Port already taken by a previous run | `python3.11 local.py status` to see what's running |
| Port busy (Mac/Linux) | Stale process | `kill -9 $(lsof -ti :8000)` |
| Port busy (Windows) | Stale process | `netstat -ano \| findstr :8000` → `taskkill /PID <pid> /F` |
| `SSL SYSCALL error: Operation timed out` | Postgres idle timeout (>60 min) | Restart agents — `Ctrl+C` then `python3.11 local.py start` |
| `No module named 'pinecone'` | Dependencies not installed | `python3.11 -m pip install -r requirements_local.txt` |
| Agent dies immediately after launch | Bad env var or missing secret | Check coloured log output in the terminal for the specific error |
| Supervisor cold start fails | Sub-agents not yet healthy | `local.py` waits automatically — if timeout, check sub-agent logs |