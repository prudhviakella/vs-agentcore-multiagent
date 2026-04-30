"""
platform/learning_pipeline.py
==============================
Continuous learning pipeline for VS AgentCore Multi-Agent Platform.

Triggered on demand via POST /api/v1/clinical-trial/learning/run.

Steps:
  1. Triage        — fetch traces, separate positive/negative feedback
  2. Diagnose      — cluster negative traces by failure pattern
  3. Prompt improve — GPT-4o rewrites supervisor prompt based on failures
  4. Self-test      — verify improved prompt with 3 probe questions
  5. Auto-deploy    — Bedrock + SSM update (only if self-test passes)
  6. RAG gap detect — identify topics the knowledge base can't answer
  7. Fine-tune gen  — export positive traces as OpenAI JSONL
"""

import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import boto3
from openai import AsyncOpenAI

log = logging.getLogger(__name__)

REGION     = os.environ.get("AWS_REGION", "us-east-1")
SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")

# Minimum negative feedback traces before attempting prompt improvement.
# Below this threshold there isn't enough signal — skip and log.
MIN_NEGATIVE_TRACES = 3

# Self-test probe questions — must return a valid clinical trial answer
SELF_TEST_PROBES = [
    "Which cancer trials are in the knowledge base?",
    "What are the primary endpoints of NCT04470427?",
    "Which trials are in Phase 3?",
]


# ── Data models ────────────────────────────────────────────────────────────

@dataclass
class FailurePattern:
    reason:   str
    count:    int
    examples: list[dict] = field(default_factory=list)
    summary:  str = ""


@dataclass
class PromptImprovementResult:
    status:          str   # deployed | skipped | self_test_failed | error
    reason:          str   = ""
    old_version:     str   = ""
    new_version:     str   = ""
    changes_summary: str   = ""
    patterns:        list  = field(default_factory=list)
    self_test_pass:  bool  = False


@dataclass
class RagGap:
    topic:     str
    frequency: int
    questions: list[str] = field(default_factory=list)


@dataclass
class LearningReport:
    run_id:            str
    timestamp:         str
    traces_analyzed:   int
    negative_count:    int
    positive_count:    int
    prompt_result:     PromptImprovementResult = None
    rag_gaps:          list[RagGap]            = field(default_factory=list)
    finetune_examples: int                     = 0
    finetune_jsonl:    str                     = ""
    error:             str                     = ""

    def to_dict(self) -> dict:
        def _ser(o):
            if hasattr(o, "__dataclass_fields__"):
                return {k: _ser(v) for k, v in o.__dict__.items()}
            if isinstance(o, list):
                return [_ser(i) for i in o]
            return o

        return _ser(self)


# ── Helpers ────────────────────────────────────────────────────────────────

def _flatten(item: dict) -> dict:
    """Flatten DynamoDB typed values to plain Python."""
    out = {}
    for k, v in item.items():
        if isinstance(v, dict):
            if "S" in v:   out[k] = v["S"]
            elif "N" in v: out[k] = float(v["N"])
            elif "BOOL" in v: out[k] = v["BOOL"]
            elif "L" in v: out[k] = [_flatten(i) if isinstance(i, dict) else i for i in v["L"]]
            elif "M" in v: out[k] = _flatten(v["M"])
            else: out[k] = v
        elif isinstance(v, Decimal):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def _get_ssm(name: str) -> str:
    ssm = boto3.client("ssm", region_name=REGION)
    return ssm.get_parameter(Name=name)["Parameter"]["Value"]


def _get_openai_key() -> str:
    import json as _json
    sm = boto3.client("secretsmanager", region_name=REGION)
    secret = sm.get_secret_value(SecretId=f"{SSM_PREFIX}/openai")["SecretString"]
    return _json.loads(secret)["api_key"]


# ── Main pipeline ──────────────────────────────────────────────────────────

class ContinuousLearningPipeline:

    def __init__(self):
        self.region     = REGION
        self.ssm_prefix = SSM_PREFIX
        self._openai: Optional[AsyncOpenAI] = None

    # ── OpenAI client ─────────────────────────────────────────────────────

    async def _oai(self) -> AsyncOpenAI:
        if not self._openai:
            self._openai = AsyncOpenAI(api_key=_get_openai_key())
        return self._openai

    # ── Step 1: Fetch traces from DynamoDB ────────────────────────────────

    def _fetch_traces(self) -> list[dict]:
        try:
            table_name = _get_ssm(f"{SSM_PREFIX}/dynamodb/trace_table_name")
            dynamo     = boto3.resource("dynamodb", region_name=REGION)
            table      = dynamo.Table(table_name)
            items, resp = [], table.scan()
            items.extend(resp.get("Items", []))
            while "LastEvaluatedKey" in resp:
                resp = table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
                items.extend(resp.get("Items", []))
            # Convert Decimal → float
            return [_flatten(i) if any(isinstance(v, Decimal) for v in i.values()) else i
                    for i in items]
        except Exception as exc:
            log.error(f"[LEARN] fetch_traces failed: {exc}")
            return []

    # ── Step 2: Triage ────────────────────────────────────────────────────

    def _triage(self, traces: list[dict]) -> tuple[list, list]:
        """Split traces into negative-feedback and positive-feedback groups."""
        negative = [t for t in traces if t.get("feedback_rating") == "negative"
                    and t.get("question")]
        positive = [t for t in traces if t.get("feedback_rating") == "positive"
                    and t.get("question") and t.get("answer")]
        return negative, positive

    # ── Step 3: Diagnose failure patterns ─────────────────────────────────

    def _diagnose(self, negative: list[dict]) -> list[FailurePattern]:
        """Group negative traces by feedback_reason and extract examples."""
        groups: dict[str, list] = {}
        for t in negative:
            reason = t.get("feedback_reason", "Other") or "Other"
            groups.setdefault(reason, []).append(t)

        patterns = []
        for reason, traces in sorted(groups.items(), key=lambda x: -len(x[1])):
            examples = []
            for t in traces[:3]:  # max 3 examples per pattern
                tool_summary = []
                for td in (t.get("tool_details") or [])[:2]:
                    if isinstance(td, dict):
                        tool_summary.append({
                            "agent":    td.get("name", ""),
                            "query":    str(td.get("query", ""))[:200],
                            "response": str(td.get("response", ""))[:300],
                        })
                examples.append({
                    "question":     t.get("question", "")[:300],
                    "answer":       t.get("answer", "")[:300],
                    "tool_details": tool_summary,
                    "comment":      t.get("feedback_comment", ""),
                })
            patterns.append(FailurePattern(
                reason=reason, count=len(traces), examples=examples
            ))
        return patterns

    # ── Step 4: Improve supervisor prompt with GPT-4o ─────────────────────

    async def _improve_prompt(self, patterns: list[FailurePattern]) -> tuple[str, str]:
        """
        Returns (improved_prompt_text, changes_summary).
        Uses the current supervisor prompt from Bedrock as the base.
        """
        oai = await self._oai()

        # Fetch current prompt from Bedrock
        bedrock     = boto3.client("bedrock-agent", region_name=REGION)
        prompt_id   = _get_ssm("/supervisor-agent/prod/bedrock/prompt_id")
        resp        = bedrock.get_prompt(promptIdentifier=prompt_id)
        variants    = resp.get("variants", [])
        current_txt = ""
        if variants:
            cfg = variants[0].get("templateConfiguration", {})
            current_txt = cfg.get("text", {}).get("text", "")

        # Build failure pattern summary for GPT-4o
        pattern_block = ""
        for p in patterns:
            pattern_block += f"\n### Reason: {p.reason} ({p.count} traces)\n"
            for ex in p.examples[:2]:
                pattern_block += f"Question: {ex['question']}\n"
                pattern_block += f"Answer given: {ex['answer']}\n"
                if ex.get("comment"):
                    pattern_block += f"User comment: {ex['comment']}\n"
                for td in ex.get("tool_details", []):
                    pattern_block += f"  Agent {td['agent']} returned: {td['response'][:200]}\n"
                pattern_block += "---\n"

        system_msg = (
            "You are an expert prompt engineer specialising in multi-agent clinical trial "
            "research platforms. Your task is to improve an agent's system prompt based on "
            "observed failure patterns from real user feedback.\n\n"
            "Rules:\n"
            "- Return ONLY the improved prompt text. No preamble, no explanation.\n"
            "- Preserve all existing structure, cases, and agent routing rules.\n"
            "- Make targeted, minimal edits to address the failure patterns.\n"
            "- Do not invent new agents or tools.\n"
            "- Keep the prompt under 4000 words."
        )

        user_msg = (
            f"CURRENT SUPERVISOR PROMPT:\n```\n{current_txt}\n```\n\n"
            f"FAILURE PATTERNS FROM USER FEEDBACK:\n{pattern_block}\n\n"
            "Produce an improved version of the supervisor prompt that addresses these failures. "
            "Also append a brief XML comment at the end: "
            "<!-- CHANGES: one sentence summary of what was changed -->"
        )

        completion = await oai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=4096,
            temperature=0.2,
        )

        improved = completion.choices[0].message.content.strip()

        # Extract changes summary from XML comment
        m = re.search(r"<!--\s*CHANGES:\s*(.+?)\s*-->", improved)
        changes_summary = m.group(1) if m else "Prompt improved based on feedback patterns."
        # Strip the comment from the actual prompt
        improved = re.sub(r"<!--\s*CHANGES:.+?-->", "", improved).strip()

        return improved, changes_summary

    # ── Step 5: Self-test ─────────────────────────────────────────────────

    async def _self_test(self, new_prompt_text: str) -> bool:
        """
        Temporarily apply the new prompt to a draft (not a versioned release)
        and call GPT-4o directly to verify it produces sensible reasoning.
        We test against the prompt logic without invoking the full AgentCore runtime
        (which would be too slow and expensive for a pipeline test).
        """
        oai = await self._oai()

        # Test: does the improved prompt correctly classify clinical queries?
        test_cases = [
            ("What are the Phase 3 results for mRNA-1273?", "Case B"),
            ("What is the capital of France?",              "Case E"),
            ("Compare enrollment across cancer trials",     "Case A"),
        ]

        passed = 0
        for question, expected_case in test_cases:
            try:
                resp = await oai.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": new_prompt_text},
                        {"role": "user",   "content":
                            f"Classify this query (respond with just 'Case X' where X is A/B/C/D/E): {question}"},
                    ],
                    max_tokens=20,
                    temperature=0,
                )
                result = resp.choices[0].message.content.strip()
                if expected_case.lower() in result.lower():
                    passed += 1
                    log.info(f"[LEARN] Self-test PASS: {question[:40]} → {result}")
                else:
                    log.warning(f"[LEARN] Self-test FAIL: {question[:40]} → {result} (expected {expected_case})")
            except Exception as exc:
                log.error(f"[LEARN] Self-test error: {exc}")

        # Pass if at least 2/3 correct
        return passed >= 2

    # ── Step 6: Deploy to Bedrock ─────────────────────────────────────────

    def _deploy_prompt(self, improved_text: str) -> tuple[str, str]:
        """
        Update supervisor prompt in Bedrock draft + create a new version.
        Returns (old_version, new_version).
        """
        bedrock    = boto3.client("bedrock-agent", region_name=REGION)
        ssm        = boto3.client("ssm",          region_name=REGION)
        prompt_id  = _get_ssm("/supervisor-agent/prod/bedrock/prompt_id")
        old_ver    = _get_ssm("/supervisor-agent/prod/bedrock/prompt_version")

        bedrock.update_prompt(
            promptIdentifier = prompt_id,
            name             = "vs-agentcore-ma-supervisor",
            description      = "Supervisor Agent system prompt — auto-updated by learning pipeline",
            variants=[{
                "name":         "default",
                "modelId":      "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "templateType": "TEXT",
                "templateConfiguration": {"text": {"text": improved_text, "inputVariables": []}},
                "inferenceConfiguration": {"text": {"temperature": 0.0, "maxTokens": 4096}},
            }],
            defaultVariant="default",
        )

        v_resp  = bedrock.create_prompt_version(
            promptIdentifier=prompt_id,
            description=f"Auto-deployed by learning pipeline — {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        )
        new_ver = str(v_resp["version"])
        ssm.put_parameter(
            Name="/supervisor-agent/prod/bedrock/prompt_version",
            Value=new_ver, Type="String", Overwrite=True
        )
        log.info(f"[LEARN] Deployed prompt v{old_ver} → v{new_ver}")
        return old_ver, new_ver

    # ── Step 7: RAG gap detection ─────────────────────────────────────────

    async def _detect_rag_gaps(self, negative: list[dict], all_traces: list[dict]) -> list[RagGap]:
        """
        Identify topics the knowledge base consistently fails to answer.
        Looks for:
          - feedback_reason = "Missing information"
          - research_agent responses containing "not found" / "no results"
          - questions with no Pinecone hits (tool_count = 0 after research call)
        Then clusters similar questions with GPT-4o.
        """
        gap_questions = []

        # From explicit "Missing information" feedback
        for t in negative:
            if t.get("feedback_reason") == "Missing information":
                gap_questions.append(t.get("question", ""))

        # From traces where research agent said "no results"
        no_result_phrases = ["no results", "not found", "could not find",
                             "no relevant", "no information", "unable to retrieve"]
        for t in all_traces:
            for td in (t.get("tool_details") or []):
                if isinstance(td, dict) and td.get("name") == "research_agent":
                    resp = str(td.get("response", "")).lower()
                    if any(p in resp for p in no_result_phrases):
                        q = t.get("question", "")
                        if q and q not in gap_questions:
                            gap_questions.append(q)

        if not gap_questions:
            return []

        # Cluster with GPT-4o
        oai = await self._oai()
        cluster_prompt = (
            "Cluster these clinical trial questions by topic area where the knowledge base "
            "lacked information. Return JSON object with key 'clusters', each having "
            "'topic' and 'questions' keys. Topics should be 2-5 words. "
            "Group similar questions together."
        )
        try:
            resp = await oai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": cluster_prompt},
                    {"role": "user",   "content": "\n".join(f"- {q}" for q in gap_questions[:20])},
                ],
                max_tokens=800,
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw     = resp.choices[0].message.content
            data    = json.loads(raw)
            clusters = data if isinstance(data, list) else data.get("clusters", data.get("topics", []))
            return [
                RagGap(
                    topic=c.get("topic", "Unknown"),
                    frequency=len(c.get("questions", [])),
                    questions=c.get("questions", [])[:5],
                )
                for c in clusters
            ]
        except Exception as exc:
            log.error(f"[LEARN] RAG gap clustering failed: {exc}")
            # Return ungrouped
            return [RagGap(topic="Uncategorized gaps", frequency=len(gap_questions),
                           questions=gap_questions[:10])]

    # ── Step 8: Fine-tuning dataset ───────────────────────────────────────

    def _generate_finetune_dataset(self, positive: list[dict]) -> tuple[int, str]:
        """
        Convert positive feedback traces into OpenAI fine-tuning JSONL.
        Format: {"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
        Returns (count, jsonl_string).
        """
        lines = []
        for t in positive:
            q = t.get("question", "").strip()
            a = t.get("answer",   "").strip()
            if not q or not a:
                continue
            # Clean the answer (remove thinking blocks, internal notes)
            a = re.sub(r"<thinking>.*?</thinking>", "", a, flags=re.DOTALL).strip()
            a = re.sub(r"EPISODIC:\s*(YES|NO)[\d.\s]*", "", a, flags=re.IGNORECASE).strip()
            if len(a) < 50:
                continue
            lines.append(json.dumps({
                "messages": [
                    {"role": "system",    "content": "You are a clinical trial research assistant."},
                    {"role": "user",      "content": q},
                    {"role": "assistant", "content": a},
                ]
            }))

        return len(lines), "\n".join(lines)

    # ── Main entrypoint ───────────────────────────────────────────────────

    async def run(self) -> LearningReport:
        run_id    = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        log.info(f"[LEARN] Pipeline started  run_id={run_id[:8]}")

        report = LearningReport(
            run_id=run_id, timestamp=timestamp,
            traces_analyzed=0, negative_count=0, positive_count=0,
        )

        try:
            # 1. Fetch
            traces = self._fetch_traces()
            report.traces_analyzed = len(traces)
            log.info(f"[LEARN] Fetched {len(traces)} traces")

            # 2. Triage
            negative, positive = self._triage(traces)
            report.negative_count = len(negative)
            report.positive_count = len(positive)
            log.info(f"[LEARN] Negative={len(negative)}  Positive={len(positive)}")

            # 3. Prompt improvement
            if len(negative) >= MIN_NEGATIVE_TRACES:
                patterns = self._diagnose(negative)
                log.info(f"[LEARN] Diagnosed {len(patterns)} failure patterns")

                improved_text, changes_summary = await self._improve_prompt(patterns)
                log.info("[LEARN] Prompt improvement generated")

                # 4. Self-test
                self_test_ok = await self._self_test(improved_text)
                log.info(f"[LEARN] Self-test: {'PASS' if self_test_ok else 'FAIL'}")

                if self_test_ok:
                    # 5. Deploy
                    old_ver, new_ver = self._deploy_prompt(improved_text)
                    report.prompt_result = PromptImprovementResult(
                        status="deployed",
                        reason=f"{len(negative)} negative traces analyzed",
                        old_version=old_ver,
                        new_version=new_ver,
                        changes_summary=changes_summary,
                        patterns=[{"reason": p.reason, "count": p.count} for p in patterns],
                        self_test_pass=True,
                    )
                else:
                    report.prompt_result = PromptImprovementResult(
                        status="self_test_failed",
                        reason="Improved prompt failed 2/3 self-test probes — not deployed",
                        patterns=[{"reason": p.reason, "count": p.count} for p in patterns],
                        self_test_pass=False,
                    )
            else:
                report.prompt_result = PromptImprovementResult(
                    status="skipped",
                    reason=f"Only {len(negative)} negative traces (need {MIN_NEGATIVE_TRACES}+)",
                )

            # 6. RAG gaps
            report.rag_gaps = await self._detect_rag_gaps(negative, traces)
            log.info(f"[LEARN] RAG gaps detected: {len(report.rag_gaps)}")

            # 7. Fine-tuning dataset
            count, jsonl = self._generate_finetune_dataset(positive)
            report.finetune_examples = count
            report.finetune_jsonl    = jsonl
            log.info(f"[LEARN] Fine-tune dataset: {count} examples")

        except Exception as exc:
            log.exception(f"[LEARN] Pipeline error: {exc}")
            report.error = str(exc)

        log.info(f"[LEARN] Pipeline complete  run_id={run_id[:8]}")
        return report