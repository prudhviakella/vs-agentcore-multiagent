#!/usr/bin/env python3
"""
harness.py — VS AgentCore Multi-Agent Test Harness
====================================================
Runs golden queries against the live platform and validates:
  1. RAG pipeline metrics (Stage 1-4 from DynamoDB)
  2. Answer content (expected facts, keywords, NCT citations)
  3. Negative queries (confidence gate must fire)
  4. Graph queries (knowledge agent returns structured data)
  5. Cross-agent routing (supervisor calls correct agents)
  6. HITL (vague queries trigger clarification)
  7. Guardrail (blocked queries return block message)

Usage:
    cd ~/PycharmProjects/vs-agentcore-multiagent/scripts
    source .env.prod

    python3.11 harness.py                         # all 49 queries
    python3.11 harness.py --category rag_high     # specific category
    python3.11 harness.py --id R01 R02 R03        # specific query IDs
    python3.11 harness.py --quick                 # first 5 RAG high + 3 graph
    python3.11 harness.py --output results.json   # save results to JSON
    python3.11 harness.py --report                # show last 10 DynamoDB traces

Categories: rag_high, rag_medium, rag_negative, rag_comparative,
            graph, cross_agent, hitl, guardrail

Exit codes:  0 = all pass  |  1 = failures exist
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

import boto3

# ── Config ────────────────────────────────────────────────────────────────

ALB_URL    = os.environ.get(
    "PLATFORM_URL",
    "http://vs-agentcore-ma-alb-550721339.us-east-1.elb.amazonaws.com"
)
API_KEY    = os.environ.get("PLATFORM_API_KEY", "")
REGION     = os.environ.get("AWS_REGION", "us-east-1")
TABLE_NAME = "vs-agentcore-ma-traces"

DATASET_PATHS = [
    Path(__file__).resolve().parent.parent / "tests" / "golden_dataset.json",
    Path(__file__).resolve().parent / "golden_dataset.json",
]

# ── RAG Thresholds ────────────────────────────────────────────────────────

RAG_THRESHOLDS = {
    "rag_chunks_stage1":       (">=", 50,    "Pinecone recall ≥50"),
    "rag_chunks_stage2":       (">=", 5,     "Cohere rerank kept ≥5"),
    "rag_rerank_top_score":    (">=", 0.15,  "Top score ≥0.15"),
    "rag_threshold_triggered": ("==", "False","Gate not fired"),
    "rag_chunks_compressed":   (">=", 1,     "Stage 3 output ≥1"),
    "rag_citation_coverage":   (">=", 0.3,   "Citations ≥30%"),
}

NEGATIVE_THRESHOLDS = {
    "rag_threshold_triggered": ("==", "True", "Gate fired (expected)"),
}

# ── Console helpers ───────────────────────────────────────────────────────

def ok(msg):    print(f"  ✅ {msg}")
def fail(msg):  print(f"  ❌ {msg}")
def warn(msg):  print(f"  ⚠️  {msg}")
def info(msg):  print(f"     {msg}")
def sep():      print(f"  {'─'*58}")

# ── Dataset loading ───────────────────────────────────────────────────────

def load_dataset() -> dict:
    for path in DATASET_PATHS:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    print("❌ golden_dataset.json not found. Copy it to tests/ or scripts/")
    sys.exit(1)

def flatten_queries(dataset: dict, categories=None, ids=None) -> list:
    """Flatten all query categories into a single list."""
    # Dataset key -> short category name used in query objects
    cat_keys = {
        "rag_high_confidence":  "rag_high",
        "rag_medium_confidence":"rag_medium",
        "rag_negative":         "rag_negative",
        "rag_comparative":      "rag_comparative",
        "graph_queries":        "graph",
        "cross_agent_queries":  "cross_agent",
        "hitl_queries":         "hitl",
        "guardrail_queries":    "guardrail",
    }

    all_queries = []
    for dataset_key, short_cat in cat_keys.items():
        for q in dataset.get(dataset_key, []):
            # Ensure category field is set correctly
            q.setdefault("category", short_cat)
            all_queries.append(q)

    if categories:
        all_queries = [q for q in all_queries if q.get("category") in categories]

    if ids:
        all_queries = [q for q in all_queries if q.get("id") in ids]

    return all_queries

# ── Platform call ─────────────────────────────────────────────────────────

def call_platform(query: str, thread_id: str, timeout: int = 180) -> dict:
    url     = f"{ALB_URL}/api/v1/clinical-trial/chat"
    payload = json.dumps({
        "message":   query,
        "thread_id": thread_id,
        "domain":    "pharma",
    }).encode()

    req = Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "X-API-Key":    API_KEY,
    })

    answer  = ""
    run_id  = ""
    latency = 0.0
    error   = None

    try:
        with urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                etype = event.get("type", "")
                if etype == "token":
                    answer += event.get("content", "")
                elif etype == "done":
                    run_id  = event.get("run_id", "")
                    latency = event.get("latency_ms", 0.0)
    except URLError as e:
        error = str(e)

    return {
        "run_id":     run_id,
        "answer":     answer,
        "latency_ms": latency,
        "error":      error,
    }

# ── DynamoDB trace fetch ──────────────────────────────────────────────────

def fetch_trace(run_id: str) -> dict:
    if not run_id:
        return {}
    try:
        ddb  = boto3.resource("dynamodb", region_name=REGION)
        item = ddb.Table(TABLE_NAME).get_item(Key={"run_id": run_id}).get("Item", {})
        return dict(item)
    except Exception as e:
        warn(f"DynamoDB fetch error: {e}")
        return {}

# ── Scoring ───────────────────────────────────────────────────────────────

def score_rag(q_def: dict, answer: str, trace: dict) -> dict:
    """Score RAG pipeline metrics and answer content."""
    checks = {}
    is_negative = q_def.get("should_gate_fire", False)
    thresholds  = NEGATIVE_THRESHOLDS if is_negative else RAG_THRESHOLDS

    # RAG metric checks
    for key, (op, threshold, label) in thresholds.items():
        raw = trace.get(key)
        if raw is None:
            checks[key] = {"passed": False, "value": "missing", "label": label}
            continue
        try:
            val = float(raw) if str(raw) not in ("True", "False") else raw
        except (ValueError, TypeError):
            val = raw
        if op == ">=":
            passed = float(val) >= threshold
        elif op == "==":
            passed = str(val) == str(threshold)
        else:
            passed = False
        checks[key] = {"passed": passed, "value": raw, "threshold": threshold, "label": label}

    # Minimum rerank score check (per-query)
    min_score = q_def.get("min_rerank_score", 0.15)
    raw_score = trace.get("rag_rerank_top_score")
    if raw_score and not is_negative:
        score_val = float(raw_score)
        checks["min_rerank_score"] = {
            "passed":    score_val >= min_score,
            "value":     raw_score,
            "threshold": min_score,
            "label":     f"Top score ≥{min_score}",
        }

    # Answer content checks
    answer_lower = answer.lower()

    # Expected keywords
    expected_kw = q_def.get("expected_answer_contains", [])
    kw_hits = [kw for kw in expected_kw if kw.lower() in answer_lower]
    kw_score = len(kw_hits) / max(len(expected_kw), 1) if expected_kw else 1.0
    checks["keyword_coverage"] = {
        "passed":    kw_score >= 0.5,
        "value":     f"{kw_score:.2f}  hits={kw_hits}",
        "threshold": 0.5,
        "label":     "Keyword coverage ≥50%",
    }

    # Expected facts (more specific than keywords)
    expected_facts = q_def.get("expected_facts", [])
    if expected_facts and not is_negative:
        fact_hits = [f for f in expected_facts if f.lower() in answer_lower]
        fact_score = len(fact_hits) / max(len(expected_facts), 1)
        checks["fact_coverage"] = {
            "passed":    fact_score >= 0.4,
            "value":     f"{fact_score:.2f}  hits={fact_hits}",
            "threshold": 0.4,
            "label":     "Fact coverage ≥40%",
        }

    # NCT citation check
    expected_nct = q_def.get("expected_nct")
    if expected_nct:
        checks["nct_citation"] = {
            "passed": expected_nct in answer,
            "value":  "found" if expected_nct in answer else "missing",
            "label":  f"NCT citation ({expected_nct})",
        }

    # Expected chunk IDs retrieved
    expected_chunk_ids = q_def.get("expected_chunk_ids", [])
    if expected_chunk_ids:
        retrieved_ids = trace.get("rag_top_chunk_ids", [])
        chunk_hits = [cid for cid in expected_chunk_ids if cid in retrieved_ids]
        checks["chunk_retrieval"] = {
            "passed": len(chunk_hits) > 0,
            "value":  f"{len(chunk_hits)}/{len(expected_chunk_ids)} expected chunks retrieved",
            "label":  "Expected chunk retrieved",
        }

    passed = sum(1 for r in checks.values() if r.get("passed", False))
    total  = len(checks)
    return {"checks": checks, "passed": passed, "total": total,
            "score": passed / total if total else 0}


def score_graph(q_def: dict, answer: str, trace: dict) -> dict:
    """Score graph/knowledge agent response."""
    checks = {}
    answer_lower = answer.lower()

    expected_kw = q_def.get("expected_answer_contains", [])
    kw_hits = [kw for kw in expected_kw if kw.lower() in answer_lower]
    kw_score = len(kw_hits) / max(len(expected_kw), 1) if expected_kw else 1.0
    checks["keyword_coverage"] = {
        "passed":    kw_score >= 0.5,
        "value":     f"{kw_score:.2f}  hits={kw_hits}",
        "label":     "Keyword coverage ≥50%",
    }

    # Graph agent should never return INSUFFICIENT_CONTEXT
    checks["no_insufficient"] = {
        "passed": "INSUFFICIENT_CONTEXT" not in answer,
        "value":  "ok" if "INSUFFICIENT_CONTEXT" not in answer else "INSUFFICIENT_CONTEXT returned",
        "label":  "No INSUFFICIENT_CONTEXT",
    }

    # Answer not empty
    checks["has_answer"] = {
        "passed": len(answer.strip()) > 20,
        "value":  f"{len(answer)} chars",
        "label":  "Answer not empty",
    }

    expected_nct = q_def.get("expected_nct")
    if expected_nct:
        checks["nct_mention"] = {
            "passed": expected_nct in answer,
            "value":  "found" if expected_nct in answer else "missing",
            "label":  f"NCT mentioned ({expected_nct})",
        }

    passed = sum(1 for r in checks.values() if r.get("passed", False))
    total  = len(checks)
    return {"checks": checks, "passed": passed, "total": total,
            "score": passed / total if total else 0}


def score_guardrail(q_def: dict, answer: str, trace: dict) -> dict:
    """Score guardrail — answer should contain block message."""
    block_phrases = [
        "could not be processed", "prohibited content", "outside the scope",
        "blocked", "cannot", "not able to", "clinical trial research platform",
        "qualified healthcare professional"
    ]
    answer_lower = answer.lower()
    blocked = any(p in answer_lower for p in block_phrases)

    checks = {
        "guardrail_fired": {
            "passed": blocked,
            "value":  "blocked" if blocked else "NOT blocked — guardrail missed",
            "label":  "Guardrail fired",
        }
    }
    return {"checks": checks, "passed": 1 if blocked else 0, "total": 1,
            "score": 1.0 if blocked else 0.0}


def score_hitl(q_def: dict, answer: str, trace: dict) -> dict:
    """Score HITL — answer should ask for clarification."""
    clarify_phrases = [
        "clarify", "which trial", "which nct", "could you specify",
        "please provide", "more specific", "which study", "did you mean"
    ]
    answer_lower = answer.lower()
    clarified = any(p in answer_lower for p in clarify_phrases)

    # Also check if tools contains ask_user_input
    tools = trace.get("tools", [])
    tools_list = tools if isinstance(tools, list) else []
    hitl_called = any("clarify" in str(t).lower() or "ask_user" in str(t).lower()
                      for t in tools_list)

    checks = {
        "clarification_requested": {
            "passed": clarified or hitl_called,
            "value":  "clarification in answer" if clarified else
                      ("HITL tool called" if hitl_called else "No clarification"),
            "label":  "Clarification requested",
        }
    }
    return {"checks": checks, "passed": 1 if (clarified or hitl_called) else 0,
            "total": 1, "score": 1.0 if (clarified or hitl_called) else 0.0}


def score_query(q_def: dict, answer: str, trace: dict) -> dict:
    category = q_def.get("category", "")
    if category in ("rag_high", "rag_medium", "rag_negative", "rag_comparative"):
        return score_rag(q_def, answer, trace)
    elif category == "graph":
        return score_graph(q_def, answer, trace)
    elif category == "cross_agent":
        return score_rag(q_def, answer, trace)  # reuse RAG scoring + keyword check
    elif category == "guardrail":
        return score_guardrail(q_def, answer, trace)
    elif category == "hitl":
        return score_hitl(q_def, answer, trace)
    return {"checks": {}, "passed": 0, "total": 0, "score": 0}

# ── Print result ──────────────────────────────────────────────────────────

def print_result(q_def: dict, response: dict, trace: dict, evaluation: dict):
    score_pct = int(evaluation["score"] * 100)
    status    = "✅ PASS" if evaluation["score"] >= 0.7 else "❌ FAIL"
    category  = q_def.get("category", "?").upper()

    print(f"\n{'─'*60}")
    print(f"  [{category}] {q_def['id']} — {status} ({score_pct}%)")
    print(f"  Query: {q_def['query'][:80]}...")
    print(f"  Latency: {response.get('latency_ms', 0)/1000:.1f}s  "
          f"run_id: {response.get('run_id', 'N/A')[:8]}...")

    if response.get("error"):
        fail(f"Request error: {response['error']}")
        return

    print(f"\n  Checks:")
    for key, r in evaluation["checks"].items():
        icon = "✅" if r.get("passed") else "❌"
        print(f"    {icon} {r.get('label', key)}: {r.get('value', '?')}")

    # RAG pipeline summary
    rag_fields = {k: v for k, v in trace.items() if k.startswith("rag_")}
    if rag_fields:
        print(f"\n  RAG Pipeline:")
        for key in ["rag_chunks_stage1", "rag_chunks_stage2", "rag_rerank_top_score",
                    "rag_chunks_compressed", "rag_citation_coverage", "rag_threshold_triggered"]:
            if key in rag_fields:
                print(f"    {key}: {rag_fields[key]}")

    # Answer preview
    answer = response.get("answer", "")
    if answer:
        print(f"\n  Answer: {answer[:250]}{'...' if len(answer) > 250 else ''}")

# ── Main harness run ──────────────────────────────────────────────────────

def run_harness(queries: list, output_file=None) -> int:
    if not API_KEY:
        print("❌ PLATFORM_API_KEY not set. Run: source .env.prod")
        sys.exit(1)

    print("=" * 60)
    print(f"  VS AgentCore — Test Harness v3.0")
    print(f"  Queries: {len(queries)}  Platform: {ALB_URL}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Category summary
    cats = {}
    for q in queries:
        c = q.get("category", "?")
        cats[c] = cats.get(c, 0) + 1
    for cat, count in sorted(cats.items()):
        info(f"{cat}: {count} queries")

    all_results = []
    passed_total = 0
    category_stats = {}

    for i, q_def in enumerate(queries):
        print(f"\n▶ [{i+1}/{len(queries)}] Running {q_def['id']}...")
        thread_id = str(uuid.uuid4())

        response = call_platform(q_def["query"], thread_id)

        if response.get("error"):
            warn(f"Request failed: {response['error']}")
            all_results.append({"query_id": q_def["id"], "error": response["error"]})
            continue

        # Wait for DynamoDB write
        time.sleep(2)
        trace = fetch_trace(response["run_id"])

        evaluation = score_query(q_def, response.get("answer", ""), trace)
        print_result(q_def, response, trace, evaluation)

        passed = evaluation["score"] >= 0.7
        if passed:
            passed_total += 1

        cat = q_def.get("category", "?")
        if cat not in category_stats:
            category_stats[cat] = {"passed": 0, "total": 0}
        category_stats[cat]["total"] += 1
        if passed:
            category_stats[cat]["passed"] += 1

        all_results.append({
            "query_id":   q_def["id"],
            "category":   q_def.get("category"),
            "query":      q_def["query"],
            "run_id":     response.get("run_id"),
            "latency_ms": response.get("latency_ms"),
            "score":      evaluation["score"],
            "passed":     passed,
            "checks":     evaluation["checks"],
            "rag_metrics": {k: v for k, v in trace.items() if k.startswith("rag_")},
            "answer":     response.get("answer", "")[:500],
        })

        if i < len(queries) - 1:
            time.sleep(2)

    # Summary
    total = len(queries)
    print(f"\n{'='*60}")
    print(f"  HARNESS SUMMARY")
    print(f"  Total: {total}  Passed: {passed_total}  Failed: {total - passed_total}")
    print(f"  Overall: {int(passed_total/total*100)}%")
    print(f"\n  By Category:")
    for cat, stats in sorted(category_stats.items()):
        pct = int(stats["passed"] / stats["total"] * 100)
        icon = "✅" if pct >= 70 else "❌"
        print(f"    {icon} {cat:<20} {stats['passed']}/{stats['total']} ({pct}%)")
    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    if output_file:
        with open(output_file, "w") as f:
            json.dump({
                "run_at":       datetime.now().isoformat(),
                "platform_url": ALB_URL,
                "passed":       passed_total,
                "total":        total,
                "category_stats": category_stats,
                "results":      all_results,
            }, f, indent=2, default=str)
        print(f"\n  Results saved: {output_file}")

    return 0 if passed_total == total else 1


def show_report():
    print("=" * 60)
    print("  Recent DynamoDB Traces (last 10 with RAG data)")
    print("=" * 60)
    try:
        ddb   = boto3.resource("dynamodb", region_name=REGION)
        items = ddb.Table(TABLE_NAME).scan(
            FilterExpression="attribute_exists(rag_chunks_stage1)",
            Limit=20,
        ).get("Items", [])
        items.sort(key=lambda x: float(x.get("ts", 0)), reverse=True)
        for item in items[:10]:
            rid  = item.get("run_id", "")[:8]
            s1   = item.get("rag_chunks_stage1", "?")
            s2   = item.get("rag_chunks_stage2", "?")
            sc   = item.get("rag_rerank_top_score", "?")
            gate = item.get("rag_threshold_triggered", "?")
            comp = item.get("rag_chunks_compressed", "?")
            cit  = item.get("rag_citation_coverage", "?")
            print(f"  {rid}  s1={s1}  s2={s2}  score={sc}  "
                  f"gate={gate}  compressed={comp}  citation={cit}")
    except Exception as e:
        print(f"  Error: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VS AgentCore Test Harness v3.0")
    parser.add_argument("--category", nargs="+",
                        choices=["rag_high","rag_medium","rag_negative","rag_comparative",
                                 "graph","cross_agent","hitl","guardrail"],
                        help="Run specific categories")
    parser.add_argument("--id",     nargs="+", help="Run specific query IDs e.g. R01 G03")
    parser.add_argument("--quick",  action="store_true",
                        help="Quick smoke test: 3 rag_high + 2 graph + 1 negative")
    parser.add_argument("--report", action="store_true",
                        help="Show recent DynamoDB trace report")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    if args.report:
        show_report()
        sys.exit(0)

    dataset = load_dataset()

    if args.quick:
        queries = flatten_queries(dataset, categories=["rag_high"])[:3] + \
                  flatten_queries(dataset, categories=["graph"])[:2] + \
                  flatten_queries(dataset, categories=["rag_negative"])[:1]
    else:
        queries = flatten_queries(dataset,
                                  categories=args.category,
                                  ids=args.id)

    if not queries:
        print("❌ No queries matched. Check --category or --id filters.")
        sys.exit(1)

    sys.exit(run_harness(queries, output_file=args.output))
