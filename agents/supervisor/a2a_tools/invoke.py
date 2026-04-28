"""
a2a_tools/invoke.py
====================
Async sub-agent invocation via AWS Bedrock AgentCore using httpx.

WHY httpx INSTEAD OF boto3?
-----------------------------
boto3 is synchronous. Calling it in an async function blocks the event loop
for 15-30s — freezing all concurrent SSE streams.

httpx gives us true async streaming:
  async for line in response.aiter_lines():  # non-blocking, yields each line
      ...

No threads, no queues, no sentinel patterns needed.

HOW AWS REQUEST SIGNING WORKS (SigV4)
--------------------------------------
AWS requires every request to be signed. boto3 does this automatically.
With httpx we do it manually using botocore's signing utilities:

  1. Get current credentials from the standard AWS chain
     (env vars → ~/.aws/credentials → ECS IAM role)
  2. Build an AWSRequest with URL + headers + body
  3. SigV4Auth computes a HMAC-SHA256 signature and adds:
       Authorization: AWS4-HMAC-SHA256 Credential=... SignedHeaders=... Signature=...
       X-Amz-Date: 20240428T123456Z
  4. httpx sends the signed request

CREDENTIAL REFRESH
-------------------
get_frozen_credentials() calls self._refresh() internally before returning.
This means credentials are always fresh — botocore handles ECS IAM role
rotation automatically. Safe to call per request.

THE AGENTCORE HTTP API (discovered from boto3 service model)
--------------------------------------------------------------
  Method : POST
  URL    : https://bedrock-agentcore.{region}.amazonaws.com
              /runtimes/{percent_encoded_arn}/invocations
  Headers:
    Content-Type                                  : application/json
    X-Amzn-Bedrock-AgentCore-Runtime-Session-Id   : {session_id}
    Authorization                                 : (added by SigV4Auth)
    X-Amz-Date                                    : (added by SigV4Auth)

ARN ENCODING
-------------
The runtime ARN contains colons and slashes which must be percent-encoded
in the URL path. Verified: quote(arn, safe='') produces identical output
to boto3's internal encoding. Example:
  arn:aws:bedrock-agentcore:us-east-1:123:runtime/abc
  → arn%3Aaws%3Abedrock-agentcore%3Aus-east-1%3A123%3Aruntime%2Fabc
"""
import asyncio
import json
import logging
import os
from urllib.parse import quote

import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import get_session

from agents.supervisor.a2a_tools.parser   import parse_sse_stream
from agents.supervisor.a2a_tools.registry import get_runtime_arns

log    = logging.getLogger(__name__)
REGION = os.environ.get("AWS_REGION", "us-east-1")

# Verified from boto3 service model: client.meta.endpoint_url
_BASE_URL = f"https://bedrock-agentcore.{REGION}.amazonaws.com"


def _build_signed_headers(url: str, headers: dict, body: bytes) -> dict:
    """
    Sign an HTTP request with AWS SigV4 and return headers ready for httpx.

    Uses botocore's SigV4Auth directly — the same signing logic boto3 uses
    internally, just called explicitly since we're bypassing boto3's HTTP layer.

    WHY get_session() CALLED EACH TIME?
    ------------------------------------
    get_session().get_credentials().get_frozen_credentials() triggers
    botocore's automatic credential refresh. On ECS, IAM role credentials
    rotate every hour. Calling this per request ensures we always sign
    with valid credentials — never a stale snapshot.

    Parameters
    ----------
    url     : str   — full request URL with percent-encoded ARN
    headers : dict  — Content-Type + session ID headers
    body    : bytes — request body (included in signature)

    Returns
    -------
    dict — headers with Authorization and X-Amz-Date added by SigV4Auth
    """
    credentials = get_session().get_credentials().get_frozen_credentials()
    aws_request = AWSRequest(method="POST", url=url, data=body, headers=headers)
    SigV4Auth(credentials, "bedrock-agentcore", REGION).add_auth(aws_request)
    return dict(aws_request.headers)


async def invoke_sub_agent(
    agent_name:  str,
    payload:     dict,
    token_queue: asyncio.Queue,
) -> str:
    """
    Invoke a sub-agent AgentCore runtime and stream its response asynchronously.

    FLOW
    ----
    1. Get runtime ARN from SSM
    2. Build and sign the HTTP request (SigV4)
    3. Open async streaming connection with httpx
    4. aiter_lines() yields each SSE line as it arrives — non-blocking
    5. parse_sse_stream() processes lines → routes events → builds answer
    6. Return complete answer as the tool result

    Parameters
    ----------
    agent_name  : str            — "knowledge", "research", or "chart"
    payload     : dict           — {"message": query, "session_id": ..., "domain": ...}
    token_queue : asyncio.Queue  — chart events go here → UI renders them

    Returns
    -------
    str — complete sub-agent answer.
    This becomes the ToolMessage content that gpt-5.5 reads in the next
    ReAct turn to decide whether to call another tool or answer the user.
    """
    arns = get_runtime_arns()
    if agent_name not in arns:
        raise RuntimeError(
            f"No runtime ARN for '{agent_name}'. "
            f"Available: {list(arns.keys())}. "
            f"Run: python deploy.py agents --agent {agent_name}"
        )

    runtime_arn = arns[agent_name]
    session_id  = payload.get("session_id", f"{agent_name}-session")
    body        = json.dumps(payload).encode("utf-8")

    # Percent-encode the ARN for use in the URL path.
    # Verified: quote(arn, safe='') matches boto3's internal encoding exactly.
    # Colons → %3A, slashes → %2F (e.g. arn:aws:...runtime/abc → arn%3Aaws%3A...runtime%2Fabc)
    url = f"{_BASE_URL}/runtimes/{quote(runtime_arn, safe='')}/invocations"

    # Required headers from AgentCore service model (discovered via boto3 introspection)
    headers = {
        "Content-Type": "application/json",
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
    }

    # Add AWS SigV4 Authorization and X-Amz-Date headers
    signed_headers = _build_signed_headers(url, headers, body)

    log.info(f"[A2A] Invoking {agent_name}  session={session_id[-8:]}")

    # Stream the response with httpx.
    # timeout=300: sub-agents can take 90-120s — must be generous.
    # aiter_lines() yields each complete SSE line as it arrives from the network.
    # This is non-blocking — the event loop serves other requests between lines.
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream(
            "POST", url,
            headers = signed_headers,
            content = body,
        ) as response:

            # Non-200 means authentication failure, wrong ARN, or AgentCore error
            if response.status_code != 200:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="replace")[:300]
                if response.status_code == 403:
                    # 403 almost always means SigV4 signing failure or wrong IAM permissions
                    raise RuntimeError(
                        f"AgentCore 403 Forbidden for '{agent_name}'. "
                        f"Check IAM role has bedrock-agentcore:InvokeAgentRuntime permission. "
                        f"Detail: {error_text}"
                    )
                raise RuntimeError(
                    f"AgentCore {response.status_code} for '{agent_name}': {error_text}"
                )

            # Parse the SSE stream line by line.
            # aiter_lines() is an async generator — yields one line per await,
            # allowing the event loop to process other work between lines.
            answer = await parse_sse_stream(
                agent_name,
                response.aiter_lines(),
                token_queue,
            )

    log.info(f"[A2A] {agent_name} complete  answer_len={len(answer)}")
    return answer