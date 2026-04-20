# VS AgentCore Multi-Agent — Local Development Guide

Run all 5 agents + Supervisor + Platform locally for development and testing.
No Docker required. All agents connect to real AWS services (Pinecone, Neo4j, RDS, Bedrock).

---

## Prerequisites

- Python 3.11
- AWS credentials configured (`~/.aws/credentials`)
- All secrets deployed to AWS (`./scripts/deploy.sh secrets`)
- All prompts deployed to Bedrock (`./scripts/deploy.sh prompts`)
- Agent registry written to SSM (`./scripts/deploy.sh registry`)

---

## One-time Setup

### 1. Install dependencies

```bash
cd ~/PycharmProjects/vs-agentcore-multiagent
python3.11 -m pip install -r requirements-local.txt
python3.11 -m pip install fastapi uvicorn httpx python-multipart  # platform deps
```

### 2. Configure environment

```bash
cp .env.prod.template .env.prod
# Edit .env.prod with your actual keys
```

Required values in `.env.prod`:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=<your-account-id>
export SSM_PREFIX=/vs-agentcore-multiagent/prod
export AGENT_ENV=prod
export LOCAL_MODE=false          # overridden per-agent below
export OPENAI_API_KEY=sk-...
export PINECONE_API_KEY=pcsk_...
export PINECONE_INDEX_NAME=clinical-agent
export NEO4J_URI=neo4j+s://...
export NEO4J_USER=...
export NEO4J_PASSWORD=...
export PLATFORM_API_KEY=vs-...
export POSTGRES_URL=postgresql://...
```

### 3. Write agent registry to SSM (first time only)

```bash
source .env.prod
./scripts/deploy.sh registry
```

---

## Running Locally

Open **7 terminal tabs** — one per agent + one for Platform.

### Terminal 1 — Research Agent (port 8001)

```bash
cd ~/PycharmProjects/vs-agentcore-multiagent
source .env.prod && export AGENT_NAME=research-agent && python3.11 -m agents.research.app
```

### Terminal 2 — Knowledge Agent (port 8002)

```bash
cd ~/PycharmProjects/vs-agentcore-multiagent
source .env.prod && export AGENT_NAME=knowledge-agent && python3.11 -m agents.knowledge.app
```

### Terminal 3 — Safety Agent (port 8003)

```bash
cd ~/PycharmProjects/vs-agentcore-multiagent
source .env.prod && export AGENT_NAME=safety-agent && python3.11 -m agents.safety.app
```

### Terminal 4 — Chart Agent (port 8005)

```bash
cd ~/PycharmProjects/vs-agentcore-multiagent
source .env.prod && export AGENT_NAME=chart-agent && python3.11 -m agents.chart.app
```

### Terminal 5 — Supervisor Agent (port 8000)

```bash
cd ~/PycharmProjects/vs-agentcore-multiagent
source .env.prod && export AGENT_NAME=supervisor-agent && export LOCAL_MODE=true && python3.11 -m agents.supervisor.app
```

> `LOCAL_MODE=true` makes the Supervisor call sub-agents via HTTP on localhost
> instead of `invoke_agent_runtime()`. Only set this for the Supervisor.

### Terminal 6 — Platform API (port 8080)

```bash
cd ~/PycharmProjects/vs-agentcore-multiagent/platform
source ../.env.prod && LOCAL_MODE=true python3.11 -m uvicorn main:app --port 8080 --reload
```

---

## Port Reference

| Service | Port | Module |
|---|---|---|
| Platform API | 8080 | `platform/main.py` |
| Supervisor | 8000 | `agents.supervisor.app` |
| Research | 8001 | `agents.research.app` |
| Knowledge | 8002 | `agents.knowledge.app` |
| Safety | 8003 | `agents.safety.app` |
| Chart | 8005 | `agents.chart.app` |

---

## Testing

### Via Platform API (recommended)

```bash
source .env.prod

# Research query
curl -s -X POST http://localhost:8080/api/v1/clinical-trial/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $PLATFORM_API_KEY" \
  -d '{"message":"What are Phase 3 results for mRNA-1273?","thread_id":"test-001","domain":"pharma"}'

# HITL query
curl -s -X POST http://localhost:8080/api/v1/clinical-trial/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $PLATFORM_API_KEY" \
  -d '{"message":"Show me cancer trials","thread_id":"test-hitl-001","domain":"pharma"}'

# Resume HITL
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

### Directly to Supervisor (bypass platform)

```bash
curl -s -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"message":"What are Phase 3 results for mRNA-1273?","thread_id":"direct-001","domain":"pharma"}'
```

### Directly to sub-agents

```bash
# Research Agent
curl -s -X POST http://localhost:8001/invocations \
  -H "Content-Type: application/json" \
  -d '{"message":"What are Phase 3 results for mRNA-1273?","session_id":"r-001","domain":"pharma"}'

# Knowledge Agent
curl -s -X POST http://localhost:8002/invocations \
  -H "Content-Type: application/json" \
  -d '{"message":"What trials exist for colorectal cancer?","session_id":"k-001","domain":"pharma"}'

# Safety Agent
curl -s -X POST http://localhost:8003/invocations \
  -H "Content-Type: application/json" \
  -d '{"message":"mRNA-1273 showed 94.1% efficacy in 30,000 participants. Source: NCT04470427","session_id":"s-001","domain":"pharma"}'

# Chart Agent
curl -s -X POST http://localhost:8005/invocations \
  -H "Content-Type: application/json" \
  -d '{"message":"Compare efficacy: mRNA-1273 94.1%, BNT162b2 95%","session_id":"c-001","domain":"pharma"}'
```

---

## Observability Dashboard

Open in browser after starting Platform:

```
http://localhost:8080/observability
```

**Agent filter pills** (top-right):
- **Supervisor** — all end-to-end traces with full pipeline view
- **Research / Knowledge / Safety / Chart** — supervisor traces filtered to show only those that called that sub-agent

**Trace detail panel** (click any row) shows:
- Agent waterfall — distributed trace with per-agent latency bars
- Full Q&A, token counts, cost per turn
- Tool calls with args and results
- HITL questions and user selections
- Safety verdict

**Inspect traces from terminal:**

```bash
python3.11 scripts/inspect_traces.py                           # all traces (last 50)
python3.11 scripts/inspect_traces.py --limit 5                 # last 5 only
python3.11 scripts/inspect_traces.py --run-id fresh-010        # one specific run
python3.11 scripts/inspect_traces.py --agent supervisor-agent  # filter by agent
python3.11 scripts/inspect_traces.py --agent supervisor-agent --limit 3
```

`inspect_traces.py` prints a full terminal report for each trace including:

- **Identity** — run_id, agent_name, session_id, domain, prompt version
- **Question / Answer** — full text, answer length
- **Tokens + Cost** — input/output totals, per-turn breakdown (T1, T2, T3...)
- **Sub-agents called** — routing chain e.g. `research_agent → safety_agent`
- **Tool calls** — each call with type (🤖 sub-agent or 🔧 MCP), full query, full response, response length
- **Agent waterfall** — ASCII bar chart of per-agent latency and percentage of total time
- **HITL** — question asked, options presented, user selection
- **Guardrails** — PASSED/BLOCKED verdict, faithfulness and consistency scores
- **LLM timings** — per-turn latency and token counts
- **Episodic memory** — hits injected, whether stored
- **Errors** — full error messages per tool

Example output:

```
────────────────────────────────────────────────────────────────────────────────
  RUN ID      : fresh-010
  AGENT       : supervisor-agent
  ELAPSED     : 42.17s
  LLM TURNS   : 3

  QUESTION    : What are the Phase 3 results for mRNA-1273?
  ANSWER      : The Phase 3 clinical trial for mRNA-1273 (NCT04470427)...
  ANSWER LEN  : 1064 chars

  TOKENS      : 3612 in + 933 out = 4545 total
  COST        : $0.015300
  PER TURN    :  T1: 1067+24  T2: 1415+320  T3: 1767+68

  SUB-AGENTS  : research_agent → safety_agent

  TOOL CALLS (2):
    [1] research_agent  🤖 sub-agent
         query    : Phase 3 results for mRNA-1273
         response : The Phase 3 clinical trial for mRNA-1273...
         resp_len : 1064 chars

    [2] safety_agent  🤖 sub-agent
         query    : The Phase 3 clinical trial...
         response : PASSED
         resp_len : 6 chars

  AGENT SPANS (distributed trace):
    ✅ research      ████████████████████  28.73s (68%)
    ✅ safety        ░░░░░░░░░░░░░░░░░░░░  1.88s (4%)
    ─────────────────────────────────────────────────

  LLM TIMINGS:
    Turn 1: 1.46s  1067in+24out
    Turn 2: 4.17s  1415in+371out
    Turn 3: 1.62s  1849in+50out
```

---

## SSE Event Reference

All agents stream Server-Sent Events (SSE). Event types:

| Event | Description |
|---|---|
| `{"type": "tool_start", "name": "tool-search___search_tool"}` | MCP tool call started |
| `{"type": "tool_end", "name": "tool-search___search_tool"}` | MCP tool call completed |
| `{"type": "token", "content": "The Phase..."}` | Streaming LLM token |
| `{"type": "chart", "config": {...}, "chart_type": "bar"}` | Chart.js config for UI rendering |
| `{"type": "interrupt", "question": "...", "options": [...]}` | HITL pause — show options card to user |
| `{"type": "span", "data": {...}}` | Sub-agent observability span (internal) |
| `{"type": "error", "message": "..."}` | Error occurred |
| `{"type": "done", "latency_ms": 45000, "answer": "..."}` | Request complete |

---

## Data Accuracy Note

The Pinecone knowledge base contains **mRNA-1273 (Moderna)** clinical trial data.
Queries about **BNT162b2 (Pfizer)** will return mRNA-1273 data with a note that it is a different vaccine.

Use these queries for reliable testing:
```bash
"What are Phase 3 results for mRNA-1273?"
"Show me cancer trials"
"Compare efficacy across COVID-19 vaccine trials and show me a chart"
```

---

## Reset / Clean Up

### Clear stale Postgres checkpoints

```bash
source .env.prod
python3.11 -c "
import psycopg, os
conn = psycopg.connect(os.environ['POSTGRES_URL'])
cur  = conn.cursor()
cur.execute('DELETE FROM checkpoints')
cur.execute('DELETE FROM checkpoint_writes')
conn.commit(); conn.close()
print('Checkpoints cleared')
"
```

### Clear DynamoDB traces

```bash
source .env.prod
python3.11 -c "
import boto3
ddb   = boto3.resource('dynamodb', region_name='us-east-1')
table = ddb.Table('vs-agentcore-ma-traces')
items = table.scan()['Items']
with table.batch_writer() as b:
    for i in items: b.delete_item(Key={'run_id': i['run_id']})
print(f'Cleared {len(items)} records')
"
```

Always use a **new thread_id** for each fresh test to avoid stale checkpoints.

---

## Adding a New Agent

1. Build and deploy the new AgentCore Runtime
2. Add entry to `scripts/deploy.sh` `step_registry()` function
3. Run: `./scripts/deploy.sh registry`
4. Restart Supervisor — it picks up the new agent automatically from SSM registry

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `{"detail": "Invalid API key"}` | `PLATFORM_API_KEY` not exported | `source .env.prod` in the curl terminal |
| `{"detail": [{"msg": "Field required"...}]}` | `$PLATFORM_API_KEY` empty | `source .env.prod` then re-run |
| `ParameterNotFound` | `AGENT_ENV` not set | `export AGENT_ENV=prod` |
| `Address already in use` | Port taken | `kill -9 $(lsof -ti :8000)` |
| `SSL SYSCALL error: Operation timed out` | Postgres idle timeout | Restart agents (idle >60min) |
| `server closed the connection unexpectedly` | Postgres dropped | Restart Supervisor |
| `No module named 'pinecone'` | Missing deps | `pip install -r requirements-local.txt` |
| `zsh: parse error near '\n'` | Multiline paste in zsh | Run commands one at a time |