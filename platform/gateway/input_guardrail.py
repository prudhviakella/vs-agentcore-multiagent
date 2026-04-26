"""
input_guardrail.py — Bedrock Guardrail input check for the platform gateway.

Called by platform/main.py on every /chat request BEFORE the message
reaches any agent. Evaluates user input with source="INPUT" against the
configured Bedrock Guardrail.

What this catches (configured in deploy.sh step_guardrails):
  PROMPT_ATTACK   — jailbreaks and prompt injection attempts
  OffTopicQuery   — questions outside the clinical research domain
  HATE/VIOLENCE/SEXUAL/MISCONDUCT — harmful content in user messages
  PII             — patient identifiers in queries (anonymised, not hard-blocked)

What this does NOT catch (handled elsewhere):
  Output safety   — OutputGuardrailMiddleware calls apply_guardrail(source="OUTPUT")
  Toxic patterns  — Bedrock HATE/VIOLENCE/MISCONDUCT content filters (HIGH)

FAIL-OPEN DESIGN:
  If Bedrock is unreachable or the guardrail ID is not configured, the function
  returns (False, "") — the request passes through. A guardrail service outage
  should not block legitimate research queries. Log the failure so ops can alert.

READING GUARDRAIL CONFIG:
  guardrail_id and guardrail_version are read from SSM once at cold start
  via functools.lru_cache. A redeploy of the guardrail (new version in SSM)
  requires a container restart to pick up the new version — acceptable since
  guardrail config changes are infrequent and always paired with a deploy.
"""

import logging
import os
from functools import lru_cache

import boto3

log = logging.getLogger(__name__)

_REGION     = os.environ.get("AWS_REGION", "us-east-1")
_SSM_PREFIX = os.environ.get("SSM_PREFIX", "/vs-agentcore-multiagent/prod")


@lru_cache(maxsize=1)
def _guardrail_config() -> tuple[str, str]:
    """
    Read guardrail_id and guardrail_version from SSM once at cold start.
    Returns ("", "") if not configured — causes check_input_guardrail to fail open.
    """
    try:
        ssm = boto3.client("ssm", region_name=_REGION)
        gid = ssm.get_parameter(
            Name=f"{_SSM_PREFIX}/bedrock/guardrail_id"
        )["Parameter"]["Value"]
        gver = ssm.get_parameter(
            Name=f"{_SSM_PREFIX}/bedrock/guardrail_version"
        )["Parameter"]["Value"]
        log.info(f"[INPUT_GUARDRAIL] Loaded config  guardrail_id={gid}  version={gver}")
        return gid, gver
    except Exception as exc:
        log.warning(f"[INPUT_GUARDRAIL] Could not read SSM config ({exc}) — failing open")
        return "", ""


@lru_cache(maxsize=1)
def _bedrock_client():
    return boto3.client("bedrock-runtime", region_name=_REGION)


def check_input_guardrail(text: str) -> tuple[bool, str]:
    """
    Evaluate user input against the Bedrock Guardrail (source="INPUT").

    Args:
        text: Raw user message from the HTTP request body.

    Returns:
        (False, "")            — clean, request can proceed.
        (True,  block_reason)  — guardrail intervened, return 400 to caller.

    Fails open: any exception returns (False, "") so Bedrock unavailability
    does not block legitimate queries.
    """
    if not text or not text.strip():
        return False, ""

    guardrail_id, guardrail_version = _guardrail_config()
    if not guardrail_id:
        log.debug("[INPUT_GUARDRAIL] No guardrail configured — skipping")
        return False, ""

    try:
        response = _bedrock_client().apply_guardrail(
            guardrailIdentifier = guardrail_id,
            guardrailVersion    = guardrail_version,
            source              = "INPUT",
            content             = [{"text": {"text": text}}],
        )

        if response.get("action") == "GUARDRAIL_INTERVENED":
            # Extract which policy triggered for logging
            reason = _extract_block_reason(response.get("assessments", []))
            # Use configured blocked message as the user-facing response
            user_message = (
                response.get("outputs", [{}])[0].get("text")
                or "Your request could not be processed — it matches a prohibited content policy."
            )
            log.warning(f"[INPUT_GUARDRAIL] BLOCKED  reason={reason}")
            return True, user_message

        return False, ""

    except Exception as exc:
        log.error(f"[INPUT_GUARDRAIL] Bedrock call failed ({exc}) — failing open")
        return False, ""


def _extract_block_reason(assessments: list) -> str:
    """
    Pull the triggering policy name from the assessments array for logging.
    Does not expose to the user — user sees the configured blocked message.
    """
    for assessment in assessments:
        # Denied topic
        topics = assessment.get("topicPolicy", {}).get("topics", [])
        for topic in topics:
            if topic.get("action") == "BLOCKED":
                return f"denied_topic:{topic.get('name', 'unknown')}"

        # Content filter
        filters = assessment.get("contentPolicy", {}).get("filters", [])
        for f in filters:
            if f.get("action") == "BLOCKED":
                return f"content_filter:{f.get('type', 'unknown')}"

        # PII
        pii = assessment.get("sensitiveInformationPolicy", {}).get("piiEntities", [])
        for p in pii:
            if p.get("action") == "BLOCKED":
                return f"pii:{p.get('type', 'unknown')}"

    return "unknown"