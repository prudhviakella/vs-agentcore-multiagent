"""
traces.py — Full DynamoDB trace inspector for vs-agentcore-multiagent

Usage:
  python3.11 scripts/traces.py                    # all traces
  python3.11 scripts/traces.py --limit 5          # last 5 traces
  python3.11 scripts/traces.py --run-id clean-001 # specific run
  python3.11 scripts/traces.py --agent research-agent  # filter by agent
"""
import sys, boto3, json
from decimal import Decimal

# ── Args ──────────────────────────────────────────────────────────────────────
args       = sys.argv[1:]
limit      = int(args[args.index("--limit")   + 1]) if "--limit"  in args else 50
run_id_arg = args[args.index("--run-id") + 1]       if "--run-id" in args else None
agent_arg  = args[args.index("--agent")  + 1]       if "--agent"  in args else None

def fix(obj):
    if isinstance(obj, Decimal): return float(obj)
    if isinstance(obj, list):    return [fix(i) for i in obj]
    if isinstance(obj, dict):    return {k: fix(v) for k, v in obj.items()}
    return obj

import os
region    = os.environ.get("AWS_REGION", "us-east-1")
ddb       = boto3.resource("dynamodb", region_name=region)
ssm       = boto3.client("ssm", region_name=region)
ssm_pfx   = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")

try:
    table_name = ssm.get_parameter(
        Name=f"{ssm_pfx}/dynamodb/trace_table_name"
    )["Parameter"]["Value"]
except Exception:
    table_name = "vs-agentcore-ma-traces"

table = ddb.Table(table_name)

# Scan with optional filter
from boto3.dynamodb.conditions import Attr

if run_id_arg:
    resp  = table.get_item(Key={"run_id": run_id_arg})
    items = [resp["Item"]] if "Item" in resp else []
else:
    scan_kwargs = {}
    if agent_arg:
        scan_kwargs["FilterExpression"] = Attr("agent_name").eq(agent_arg)
    items, resp = [], table.scan(**scan_kwargs)
    items.extend(resp.get("Items", []))
    while "LastEvaluatedKey" in resp:
        resp = table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"], **scan_kwargs)
        items.extend(resp.get("Items", []))

items = [fix(i) for i in items]
items.sort(key=lambda x: x.get("ts", 0), reverse=True)
items = items[:limit]

print(f"\n{'═'*80}")
print(f"  VS AgentCore Multi-Agent — Trace Inspector")
print(f"  Total records: {len(items)}")
print(f"{'═'*80}\n")

for item in items:
    print(f"\n{'─'*80}")
    print(f"  RUN ID      : {item.get('run_id')}")
    print(f"  AGENT       : {item.get('agent_name', 'unknown')}")
    print(f"  SESSION     : {item.get('session_id')}")
    print(f"  DOMAIN      : {item.get('domain')}")
    print(f"  PROMPT V    : {item.get('prompt_version')}")
    print(f"  ELAPSED     : {item.get('elapsed_ms', 0)/1000:.2f}s")
    print(f"  LLM TURNS   : {item.get('llm_turns')}")
    print(f"  CACHE HIT   : {item.get('cache_hit')}")
    print(f"  HAS ERRORS  : {item.get('has_errors')}")
    print(f"{'─'*80}")

    # Question / Answer
    q = item.get("question", "")
    a = item.get("answer", "")
    print(f"\n  QUESTION    : {q[:100]}{'...' if len(q)>100 else ''}")
    print(f"  ANSWER      : {a[:150]}{'...' if len(a)>150 else ''}")
    print(f"  ANSWER LEN  : {item.get('answer_length')} chars")

    # Tokens + Cost
    print(f"\n  TOKENS      : {item.get('input_tokens',0)} in + {item.get('output_tokens',0)} out = {item.get('total_tokens',0)} total")
    print(f"  COST        : ${item.get('token_cost_usd', 0):.6f}")
    ipt = item.get("input_tokens_per_turn", [])
    opt = item.get("output_tokens_per_turn", [])
    if ipt:
        print(f"  PER TURN    :", end="")
        for i, (inp, out) in enumerate(zip(ipt, opt)):
            print(f"  T{i+1}: {inp}+{out}", end="")
        print()

    # Sub-agents called (Supervisor trace)
    sub = item.get("sub_agents_called", [])
    if sub:
        print(f"\n  SUB-AGENTS  : {' → '.join(sub)}")

    # MCP tools (sub-agent trace)
    mcp = item.get("mcp_tools_called", [])
    if mcp:
        print(f"\n  MCP TOOLS   : {' → '.join(t.split('___')[-1] for t in mcp)}")

    # Tool details — full
    details = item.get("tool_details", [])
    if details:
        print(f"\n  TOOL CALLS ({len(details)}):")
        for i, td in enumerate(details):
            name     = td.get("name", "")
            is_sub   = td.get("is_sub_agent", False)
            is_err   = td.get("is_error", False)
            status   = "❌ ERROR" if is_err else ("🤖 sub-agent" if is_sub else "🔧 mcp")
            print(f"\n    [{i+1}] {name}  {status}")

            if is_sub:
                q2 = td.get("query", "")
                r2 = td.get("response", "")
                rl = td.get("response_length", 0)
                print(f"         query    : {q2[:80]}{'...' if len(q2)>80 else ''}")
                print(f"         response : {r2[:120]}{'...' if len(r2)>120 else ''}")
                print(f"         resp_len : {rl} chars")
            else:
                if td.get("search_query"):
                    print(f"         search   : {td['search_query'][:80]}")
                elif td.get("cypher"):
                    print(f"         cypher   : {td['cypher'][:80]}")
                elif td.get("question"):
                    print(f"         question : {td['question'][:80]}")
                    print(f"         options  : {td.get('options', [])}")
                else:
                    print(f"         args     : {td.get('args','')[:80]}")
                result = td.get("result", "")
                print(f"         result   : {result[:120]}{'...' if len(result)>120 else ''}")

    # HITL
    if item.get("hitl_fired"):
        print(f"\n  HITL:")
        print(f"    question : {item.get('hitl_question','')}")
        print(f"    options  : {item.get('hitl_options', [])}")
        if item.get("hitl_user_answer"):
            print(f"    selected : {item.get('hitl_user_answer')}")

    # Guardrails
    gp = item.get("guardrail_passed")
    if gp is not None:
        gb = item.get("guardrail_blocked", False)
        fs = item.get("faithfulness_score")
        cs = item.get("consistency_score")
        print(f"\n  GUARDRAIL   : {'✅ PASSED' if gp else '❌ BLOCKED'}")
        if fs is not None: print(f"    faithfulness : {fs:.2f}")
        if cs is not None: print(f"    consistency  : {cs:.2f}")

    # Agent spans (distributed trace)
    spans = item.get("agent_spans", [])
    if spans:
        total_ms = item.get("elapsed_ms", 1)
        print(f"\n  AGENT SPANS (distributed trace):")
        total_span_cost = 0
        for sp in spans:
            agent    = sp.get("agent", "?")
            ms       = sp.get("elapsed_ms", 0)
            pct      = round(ms / total_ms * 100) if total_ms else 0
            tokens   = sp.get("tokens", 0)
            cost     = sp.get("cost_usd", 0)
            tools_sp = sp.get("tools", [])
            status   = "✅" if sp.get("status") == "ok" else "❌"
            bar      = "█" * (pct // 5) + "░" * (20 - pct // 5)
            total_span_cost += cost
            print(f"    {status} {agent:<12}  {bar}  {ms/1000:.2f}s ({pct}%)")
            if tokens: print(f"               tokens={tokens}  cost=${cost:.5f}")
            if tools_sp:
                print(f"               tools: {', '.join(t.split('___')[-1] for t in tools_sp)}")
        print(f"    ─────────────────────────────────────────────────")
        print(f"    Total sub-agent cost: ${total_span_cost:.5f}")

    # Episodic memory
    eh = item.get("episodic_hits", 0)
    es = item.get("episodic_stored", False)
    if eh or es:
        print(f"\n  EPISODIC    : hits={eh}  stored={es}")

    # LLM timings
    timings = item.get("llm_timings", [])
    if timings:
        print(f"\n  LLM TIMINGS:")
        for t in timings:
            print(f"    Turn {t.get('turn')}: {t.get('elapsed_ms',0)/1000:.2f}s  "
                  f"{t.get('input_tokens',0)}in+{t.get('output_tokens',0)}out")

    # Errors
    if item.get("errors"):
        print(f"\n  ERRORS:")
        for e in item.get("errors", []):
            if isinstance(e, dict):
                print(f"    {e.get('tool','?')}: {e.get('message','')[:100]}")
            else:
                print(f"    {str(e)[:100]}")

print(f"\n{'═'*80}\n")
