"""
output_guardrail.py — OutputGuardrailMiddleware
=================================================
Two-layer output safety check before the answer reaches the user.

Layer 1 — regex  (<1ms, no API call) : catches obvious medical directives
Layer 2 — Bedrock Guardrail (OUTPUT) : denied topic + content filters + contextual grounding

FAIL-OPEN: If Bedrock is unreachable, Layer 2 logs a warning and passes through.
"""

import logging
import os
from functools import lru_cache
from typing import Any

import boto3
from langchain.agents.middleware import AgentState, hook_config
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from core.guardrails import check_medical_action_output
from core.middleware.base import BaseAgentMiddleware
from core.middleware.tracer import TracerMiddleware

log = logging.getLogger(__name__)

_FALLBACK_MARKER = "did not meet safety and accuracy standards"
_REGION          = os.environ.get("AWS_REGION", "us-east-1")
_SSM_PREFIX      = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")


@lru_cache(maxsize=1)
def _guardrail_config() -> tuple[str, str]:
    try:
        ssm  = boto3.client("ssm", region_name=_REGION)
        gid  = ssm.get_parameter(Name=f"{_SSM_PREFIX}/bedrock/guardrail_id")["Parameter"]["Value"]
        gver = ssm.get_parameter(Name=f"{_SSM_PREFIX}/bedrock/guardrail_version")["Parameter"]["Value"]
        log.info(f"[OUTPUT_GUARD] Guardrail loaded  id={gid}  version={gver}")
        return gid, gver
    except Exception as exc:
        log.warning(f"[OUTPUT_GUARD] Could not load guardrail config ({exc}) — failing open")
        return "", ""


@lru_cache(maxsize=1)
def _bedrock():
    return boto3.client("bedrock-runtime", region_name=_REGION)


class OutputGuardrailMiddleware(BaseAgentMiddleware):

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        run_id   = self._get_run_id(runtime)
        messages = state.get("messages", [])
        if not messages:
            return None

        # Skip: HITL interrupt pending
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                if any("ask_user_input" in tc.get("name", "")
                       for tc in (getattr(msg, "tool_calls", None) or [])):
                    log.info("[OUTPUT_GUARD] Skipping — HITL interrupt pending")
                    return None
                break

        # Find latest text answer
        answer = next(
            (str(msg.content) for msg in reversed(messages)
             if isinstance(msg, AIMessage) and msg.content),
            ""
        )
        if not answer or _FALLBACK_MARKER in answer:
            return None

        # Layer 1: regex — obvious medical directives (<1ms)
        ok, reason = check_medical_action_output(answer)
        if not ok:
            log.warning(f"[OUTPUT_GUARD] Layer 1 blocked  reason='{reason}'")
            TracerMiddleware.update_trace(run_id, {
                "guardrail_passed": False, "guardrail_blocked": True, "guardrail_reason": reason
            })
            return self._blocked(state, f"Layer 1: {reason}")

        # Layer 2: Bedrock Guardrail (OUTPUT)
        # Extract original user query — required by Bedrock contextual grounding.
        # Without the "query" qualifier Bedrock raises ValidationException:
        #   "does not contain the query. Grounding source, query and content
        #    to guard are required for contextual grounding policy evaluation."
        user_query = next(
            (str(msg.content) for msg in messages
             if getattr(msg, "type", None) == "human" and msg.content),
            ""
        )
        context_chunks = self._extract_tool_results(messages)
        passed, reason = self._apply_guardrail(answer, context_chunks, user_query)
        if not passed:
            log.warning(f"[OUTPUT_GUARD] Layer 2 blocked  reason='{reason}'")
            TracerMiddleware.update_trace(run_id, {
                "guardrail_passed": False, "guardrail_blocked": True, "guardrail_reason": reason
            })
            return self._blocked(state, f"Layer 2: {reason}")

        TracerMiddleware.update_trace(run_id, {
            "guardrail_passed": True, "guardrail_blocked": False, "guardrail_reason": None
        })
        log.info(f"[OUTPUT_GUARD] Passed  answer='{answer[:60]}'")
        return None

    # ── Bedrock call ──────────────────────────────────────────────────────

    def _apply_guardrail(
        self, answer: str, context_chunks: list[str], query: str = ""
    ) -> tuple[bool, str]:
        guardrail_id, guardrail_version = _guardrail_config()
        if not guardrail_id:
            return True, ""

        try:
            # Contextual grounding is intentionally NOT used at supervisor level.
            #
            # Bedrock contextual grounding checks word-level overlap between the answer
            # and the grounding source. It works well for single-agent RAG where the
            # answer directly quotes retrieved chunks (scores 0.85-0.95).
            #
            # In this multi-agent architecture, the supervisor synthesises across
            # sub-agent responses that are themselves already summaries of raw tool
            # results. Two layers of paraphrasing mean the supervisor answer will always
            # score < 0.4 against any grounding source — blocking legitimate responses.
            #
            # Grounded retrieval is the sub-agents' responsibility. The supervisor's
            # output guardrail only needs to check content filters and denied topics.
            # Both run when plain text is sent with no qualifiers (verified against live API).
            bedrock_content = [{"text": {"text": answer}}]

            resp = _bedrock().apply_guardrail(
                guardrailIdentifier = guardrail_id,
                guardrailVersion    = guardrail_version,
                source              = "OUTPUT",
                content             = bedrock_content,
            )

            if resp.get("action") == "GUARDRAIL_INTERVENED":
                # ANONYMIZED = Bedrock redacted PII but allowed the response — not a hard block.
                # BLOCKED = a topic/content policy hard-blocked the response.
                # Only fail on BLOCKED. Ignore ANONYMIZED interventions.
                reason = self._block_reason(resp.get("assessments", []))
                if reason:
                    return False, reason
                # All interventions were ANONYMIZED — pass through
            return True, ""

        except Exception as exc:
            log.warning(f"[OUTPUT_GUARD] Bedrock call failed ({exc}) — failing open")
            return True, ""

    @staticmethod
    def _block_reason(assessments: list) -> str:
        for a in assessments:
            # Log full assessment keys so we can see every policy type Bedrock returns
            log.debug(f"[OUTPUT_GUARD] Assessment keys: {list(a.keys())}")
            for t in a.get("topicPolicy", {}).get("topics", []):
                if t.get("action") == "BLOCKED":
                    return f"denied_topic:{t.get('name', 'unknown')}"
            for f in a.get("contentPolicy", {}).get("filters", []):
                if f.get("action") == "BLOCKED":
                    return f"content_filter:{f.get('type', 'unknown')}"
            for g in a.get("contextualGroundingPolicy", {}).get("filters", []):
                if g.get("action") == "BLOCKED":
                    return f"grounding:{g.get('type', 'unknown')}"
            # Word policy — Bedrock may return this for sensitive terms
            for w in a.get("wordPolicy", {}).get("customWords", []):
                if w.get("action") == "BLOCKED":
                    return f"word_policy:custom:{w.get('match', 'unknown')}"
            for w in a.get("wordPolicy", {}).get("managedWordLists", []):
                if w.get("action") == "BLOCKED":
                    return f"word_policy:managed:{w.get('type', 'unknown')}"
            # Sensitive information policy
            for s in a.get("sensitiveInformationPolicy", {}).get("piiEntities", []):
                if s.get("action") == "BLOCKED":
                    return f"pii:{s.get('type', 'unknown')}"
            for s in a.get("sensitiveInformationPolicy", {}).get("regexes", []):
                if s.get("action") == "BLOCKED":
                    return f"regex:{s.get('name', 'unknown')}"
        # Nothing was BLOCKED (may have been ANONYMIZED only — that is not a hard block)
        log.debug(f"[OUTPUT_GUARD] No BLOCKED action found in assessments: {assessments}")
        return None

    # ── Context extraction ────────────────────────────────────────────────

    def _extract_tool_results(self, messages: list) -> list[str]:
        """
        Collect tool result messages as grounding sources for Bedrock.
        Only includes search and graph results — skips HITL and summariser.
        """
        chunks = []
        for msg in messages:
            if getattr(msg, "type", None) != "tool":
                continue
            name = str(getattr(msg, "name", ""))
            if "ask_user_input" in name or "summariser" in name or "chart" in name:
                continue
            raw = getattr(msg, "content", "")
            text = self._to_text(raw)
            if text:
                chunks.append(text[:800])
        return chunks[:6]

    @staticmethod
    def _to_text(raw: Any) -> str:
        """Flatten tool result content to plain text."""
        if not raw:
            return ""
        if isinstance(raw, list):
            # MCP returns list of {"type": "text", "text": "..."}
            return "\n".join(
                item.get("text", "") for item in raw
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
        return str(raw).strip()

    # ── Fallback ──────────────────────────────────────────────────────────

    def _blocked(self, state: AgentState, reason: str) -> dict[str, Any]:
        log.error(f"[OUTPUT_GUARD] Hard block  reason='{reason}'")
        messages = list(state.get("messages", []))
        messages.append(AIMessage(content=(
            "I was unable to provide a verified answer for your question. "
            "The response did not meet safety and accuracy standards. "
            "Please consult a qualified professional or rephrase your question."
        )))
        return {"messages": messages, "_cache_is_fallback": True, "jump_to": "end"}