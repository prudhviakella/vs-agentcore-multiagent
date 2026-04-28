"""
supervisor/logging_setup.py
============================
CloudWatch logging — two-phase setup.

WHY TWO PHASES?
---------------
Phase 1 — setup_logging():
  Called at module import (top of app.py). Sets up stdout logging only.
  We cannot set up CloudWatch here because AgentCore injects IAM role
  credentials AFTER the module is imported. boto3.client('logs') called
  at import time has no credentials → silently fails.

Phase 2 — setup_cloudwatch():
  Called inside ensure_cold_start() AFTER the first request arrives.
  By then AgentCore has injected credentials. CloudWatch setup succeeds.

This two-phase approach is the correct pattern for AgentCore runtimes.
"""
import logging
import os
import socket

log = logging.getLogger(__name__)


def setup_logging() -> None:
    """
    Phase 1 — stdout only. Safe to call at import time.
    Called once from app.py before the first request.
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def setup_cloudwatch() -> None:
    """
    Phase 2 — add CloudWatch handler. Called from ensure_cold_start().

    By the time this runs, AgentCore has injected the IAM role credentials
    so boto3.client('logs') succeeds.

    LOG GROUP
    ---------
    AgentCore creates log groups automatically:
      /aws/bedrock-agentcore/runtimes/{runtime_id}-DEFAULT

    AGENT_RUNTIME_ID is set by AgentCore in the container environment.
    We also check the actual log group exists before attaching the handler.

    LOG STREAM
    ----------
    Named "{hostname}-{pid}" so each container instance has its own stream.
    Predictable and unique — easy to find in the CloudWatch console.
    Auto-generated Watchtower names are UUIDs that are hard to correlate.
    """
    import boto3
    import watchtower

    root = logging.getLogger()

    # Skip if already attached — ensure_cold_start() may be called multiple times
    if any(isinstance(h, watchtower.CloudWatchLogHandler) for h in root.handlers):
        return

    try:
        rt_id      = os.environ.get("AGENT_RUNTIME_ID", "")
        aws_region = os.environ.get("AWS_REGION", "us-east-1")

        if not rt_id:
            log.warning("[Logging] AGENT_RUNTIME_ID not set — CloudWatch disabled")
            return

        log_group  = f"/aws/bedrock-agentcore/runtimes/{rt_id}-DEFAULT"

        # Predictable stream name: hostname + pid
        # e.g. "ip-10-0-1-55.ec2.internal-7"
        hostname   = socket.gethostname()
        pid        = os.getpid()
        log_stream = f"{hostname}-{pid}"

        logs_client = boto3.client("logs", region_name=aws_region)

        # Verify the log group exists — if not, create it.
        # create_log_group=False means Watchtower silently fails when the group
        # is missing. We check explicitly so the error is visible.
        try:
            groups = logs_client.describe_log_groups(
                logGroupNamePrefix=log_group
            )["logGroups"]
            group_exists = any(g["logGroupName"] == log_group for g in groups)
            if not group_exists:
                logs_client.create_log_group(logGroupName=log_group)
                log.info(f"[Logging] Created log group: {log_group}")
        except Exception as e:
            log.warning(f"[Logging] Could not verify/create log group: {e}")

        handler = watchtower.CloudWatchLogHandler(
            log_group_name    = log_group,
            log_stream_name   = log_stream,
            boto3_client      = logs_client,
            create_log_group  = True,   # safety net if describe_log_groups missed it
            create_log_stream = True,
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        root.addHandler(handler)

        log.info(
            f"[Logging] CloudWatch attached  "
            f"group={log_group}  stream={log_stream}"
        )

    except Exception as e:
        # Non-fatal — platform works without CloudWatch
        # The error is visible in stdout (ECS task logs)
        log.warning(f"[Logging] CloudWatch setup failed — stdout only: {e}")