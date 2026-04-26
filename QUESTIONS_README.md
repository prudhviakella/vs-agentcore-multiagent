# VS AgentCore — Test Query Reference

This document covers every query category the platform handles, with example questions, expected behaviour, and what to look for in the UI.

---

## How the Supervisor Routes Queries

The supervisor classifies every incoming query into one of four cases before deciding which agents to call.

**Case A — Aggregate / general question**
No specific trial required. Supervisor calls knowledge_agent and/or research_agent in parallel, then optionally chart_agent if a visualisation was requested. No HITL interrupt.

**Case B — Direct / trial-specific question**
The user names a specific trial (NCT ID or drug name). Supervisor calls research_agent directly. No HITL interrupt.

**Case C — Ambiguous trial-specific**
The question is clearly about a single trial's data (efficacy, dosage, endpoints, AEs) but no trial is named. Supervisor triggers HITL, asks "Which trial are you asking about?" and presents options from the knowledge graph.

**Case D — Ambiguous intent**
The question is vague with no clear data need. Supervisor triggers HITL, asks for clarification on what the user actually wants.

---

## Case B — Direct Questions

These should answer immediately with no HITL interrupt. The research_agent searches Pinecone (5,772 chunks from protocol PDFs).

```
What are the Phase 3 results for mRNA-1273?
What are the primary endpoints of NCT04470427?
What was the dosage protocol in NCT04283461?
What adverse events were reported in NCT04405076?
How many participants were enrolled in NCT04652245?
What is the vaccine efficacy of mRNA-1273 against symptomatic COVID-19?
What statistical method was used in the COVE trial?
What were the secondary endpoints of NCT04470427?
```

**What to expect:** Reasoning card shows plan → tool steps (Searching knowledge base, Synthesising results) → answer bubble with trial-specific data.

---

## Case A — Aggregate Questions

These answer without HITL. The knowledge_agent queries Neo4j for structured trial metadata.

```
Which cancer trials are in the knowledge base?
List all Phase 3 trials
Which trials are currently active?
How many trials target colorectal cancer?
What disease areas are covered in the knowledge base?
Which sponsors have the most trials?
List all trials in Phase 1 and Phase 2
Which trials involve immunotherapy?
```

**What to expect:** Reasoning card → Querying knowledge graph → answer with a structured list.

---

## Case A — Chart Questions

Same as aggregate but supervisor also calls chart_agent. Triggers the Chart.js canvas in the UI.

```
Compare enrollment numbers across cancer trials as a bar chart
Show me trial phases distribution as a pie chart
Show me cancer trials by phase as a chart
Compare trial statuses across all studies as a chart
Show me enrollment by disease area as a chart
Show me adverse event counts across clinical trial phases as a chart
Compare efficacy across COVID-19 vaccine trials and show me a chart
Show me enrollment numbers over time for cancer trials as a line chart
```

**What to expect:** Reasoning card → Querying knowledge graph → Chart ready → chart renders inline → answer bubble with description.

---

## Case C — Ambiguous Trial-Specific (HITL)

These trigger the HITL interrupt because the question clearly needs a specific trial's data but no trial is named.

```
What were the efficacy results?
What was the dosage protocol?
What were the primary endpoints?
What adverse events were reported?
How long was the follow-up period?
What were the inclusion criteria?
What was the sample size?
What was the control arm?
What were the safety findings?
What was the recommended Phase 2 dose?
```

**What to expect:** Reasoning card → Clarification needed card → numbered list of trials → user picks one → resume call fires → answer about the selected trial.

---

## Case D — Ambiguous Intent (HITL)

These trigger HITL because the intent itself is unclear.

```
Tell me about the trials
How did it perform?
Is it safe?
What were the results?
Can you help me understand the data?
Tell me more
What do you know?
Give me a summary
```

**What to expect:** Reasoning card → Clarification needed card → intent options (e.g. safety profile, efficacy data, trial overview) → user picks one → answer.

---

## HITL Resume Flow

After selecting an option from the HITL card, the platform sends a `/resume` request. The resume response has **no `<thinking>` block** — it streams a direct answer.

**What to expect:** Selected option appears as a user bubble → answer bubble streams directly with no reasoning card.

---

## Queries That Will Fail Gracefully

These queries ask for data that is not in the knowledge base. The agents will try and return a "not found" response.

```
Show me adverse event rates across clinical trial phases as a chart
What are the Phase 1 results for NCT03374254?
Compare survival rates across oncology trials
What were the biomarker outcomes?
```

**What to expect:** Multiple reasoning cards (replanning), tool steps, then a "Note: data could not be verified" answer. No crash.

---

## Data Coverage Reference

| Source | Contents |
|---|---|
| Pinecone (`clinical-trials-index`) | 5,772 chunks from mRNA-1273 protocol PDFs (NCT04283461, NCT04405076, NCT04470427) |
| Neo4j | 20 trials — metadata only (NCT ID, title, phase, status, sponsor, disease area) |
| Semantic cache | Pinecone namespace `cache_pharma` — hit after first query, TTL 3600s |

**Trials with full protocol data (Pinecone):**
- NCT04283461 — mRNA-1273 Phase 1 safety/immunogenicity
- NCT04405076 — mRNA-1273 dose confirmation, adults 18+
- NCT04470427 — mRNA-1273 efficacy (COVE trial)

**Trials with metadata only (Neo4j):**
All 20 trials in the registry including the 3 above plus cancer, MS, diabetes, and other trials.

---

## Clearing State Between Test Runs

```bash
# Clear Postgres HITL checkpoints
python3.11 scripts/local.py clean

# Clear Pinecone semantic cache
python3.11 - << 'EOF'
import os; from dotenv import load_dotenv; load_dotenv("scripts/.env.prod")
from pinecone import Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pc.Index("clinical-agent").delete(delete_all=True, namespace="cache_pharma")
print("cache cleared")
EOF
```

Or use the **New Chat** button in the UI (top right) to start a fresh session without clearing cache.

---

## Supervisor Prompt Version

Current prompt: `GLMV90RUMZ` v10 (stored in SSM)

The prompt instructs the supervisor to:
- Wrap all reasoning in `<thinking>...</thinking>` tags
- Use `ask_user_input` tool for ambiguous queries before calling any data agent
- Call `search_tool` before generating options in `ask_user_input`
- Route visualisation requests through `chart_agent`
