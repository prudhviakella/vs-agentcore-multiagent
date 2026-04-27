#!/usr/bin/env python3.11
"""
local.py — VS AgentCore Multi-Agent Local Development
======================================================
Cross-platform replacement for the 7-terminal-tab manual startup.
Works on Windows, Mac, and Linux.

Usage:
    python local.py preflight                 # check all AWS resources before starting
    python local.py start                     # preflight + start all agents + platform
    python local.py start --no-platform       # agents only
    python local.py start --skip-preflight    # skip preflight (if you know resources exist)
    python local.py start --env FILE          # use custom env file
    python local.py start --postgres-url URL  # override POSTGRES_URL from .env.prod
    python local.py status                    # health-check all local ports
    python local.py test                      # run all test queries
    python local.py test research             # research / knowledge / chart / hitl / direct
    python local.py clean                     # clear Postgres checkpoints + DynamoDB traces

Requirements:
    pip install boto3 requests
    pip install -r requirements-local.txt
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # local.py lives in scripts/ — root is one level up

# ── Resolve Python executable ─────────────────────────────────────────────
# Always use python3.11. If the user ran `python local.py` using system python
# 3.8, warn and try to find python3.11 on PATH before launching subprocesses.

def _find_python() -> str:
    """Return path to Python 3.11+. Exits with clear message if not found."""
    if sys.version_info >= (3, 11):
        return sys.executable
    for candidate in ["python3.11", "python3.12", "python3.13"]:
        found = shutil.which(candidate)
        if found:
            return found
    print()
    print("  \u274c Python 3.11+ required but not found.")
    print()
    print("  You are running: " + sys.executable + " (" + sys.version.split()[0] + ")")
    print()
    print("  Fix:")
    print("    brew install python@3.11")
    print("    echo 'export PATH=/opt/homebrew/opt/python@3.11/bin:$PATH' >> ~/.zshrc")
    print("    source ~/.zshrc")
    print("    python3.11 local.py start")
    print()
    sys.exit(1)

PYTHON = _find_python()

# ── Agent / service definitions ───────────────────────────────────────────

AGENTS = [
    {"name": "research",   "module": "agents.research.app",   "port": 8001, "extra_env": {"AGENT_PORT": "8001"}},
    {"name": "knowledge",  "module": "agents.knowledge.app",  "port": 8002, "extra_env": {"AGENT_PORT": "8002"}},
    {"name": "chart",      "module": "agents.chart.app",      "port": 8005, "extra_env": {"AGENT_PORT": "8005"}},
    # Supervisor last — needs sub-agents healthy first
    {"name": "supervisor", "module": "agents.supervisor.app", "port": 8000,
     "extra_env": {"LOCAL_MODE": "true", "AGENT_PORT": "8000"}},
]

PLATFORM = {
    "name":      "platform",
    "cmd":       [PYTHON, "-m", "uvicorn", "main:app", "--port", "8080", "--reload"],
    "cwd":       ROOT / "platform",
    "port":      8080,
    "extra_env": {"LOCAL_MODE": "true"},
}

UI = {
    "name":      "ui",
    "cmd":       [PYTHON, "-m", "uvicorn", "app:app", "--port", "8501", "--reload"],
    "cwd":       ROOT / "ui",
    "port":      8501,
    "extra_env": {"AGENT_API_URL": "http://localhost:8080"},
}

SSM_PREFIX  = "/vs-agentcore-multiagent/prod"
AGENT_NAMES = ["research", "knowledge", "safety", "chart", "supervisor"]

# ── ANSI colours ──────────────────────────────────────────────────────────

COL = {
    "research":   "\033[94m",
    "knowledge":  "\033[92m",
    "safety":     "\033[93m",
    "chart":      "\033[96m",
    "supervisor": "\033[95m",
    "platform":   "\033[91m",
}
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"

def col(name, text):  return COL.get(name, "") + text + RESET
def ok(msg):          print("  \u2705 " + msg)
def fail(msg):        print("  \u274c " + msg)
def info(msg):        print("  " + msg)
def warn(msg):        print("  \u26a0\ufe0f  " + msg)
def hdr(msg):         print("\n" + BOLD + msg + RESET)
def dim(msg):         return DIM + msg + RESET
def sep():            print("  " + "\u2550" * 52)

# ── .env.prod parser ──────────────────────────────────────────────────────

def load_env(path=None):
    """
    Parse .env.prod cross-platform — no `source` needed on Windows.
    Handles `export KEY=val` and `KEY=val` formats.
    """
    candidates = [path, ROOT / ".env.prod"]
    env_file   = next((Path(p) for p in candidates if p and Path(p).exists()), None)
    if not env_file:
        warn(".env.prod not found — using current environment only")
        return {}

    env = {}
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip('"').strip("'")

    ok("Loaded " + env_file.name + " (" + str(len(env)) + " vars)")
    return env

# ── Health check ──────────────────────────────────────────────────────────

def port_open(port, timeout=1.0):
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False

def wait_for_port(name, port, timeout=60):
    print("  Waiting for " + name + " (:" + str(port) + ")", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open(port):
            elapsed = int(time.time() - (deadline - timeout))
            print(" \u2705  (" + str(elapsed) + "s)")
            return True
        print(".", end="", flush=True)
        time.sleep(1)
    print(" \u274c  timeout")
    return False

# ── Pre-flight check ──────────────────────────────────────────────────────

def cmd_preflight():
    """
    Verify every AWS resource the agents need before spawning any process.

    Resources checked:
      Secrets Manager  — openai, pinecone, neo4j, postgres keys
      SSM Prompts      — prompt_id + prompt_version per agent
      SSM Gateway      — mcp/gateway_url
      SSM Registry     — agents/registry
      SSM Guardrail    — bedrock/guardrail_id + guardrail_version
      SSM Misc         — dynamodb/trace_table_name, pinecone index name
      Connectivity     — Postgres TCP, Pinecone HTTP (optional, warn only)

    Returns:
        (ok: bool, issues: list[str])
    """
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    hdr("Pre-flight: checking AWS resources")

    issues   = []
    warnings = []

    try:
        session = boto3.session.Session()
        region  = os.environ.get("AWS_REGION", "us-east-1")
        sm      = session.client("secretsmanager", region_name=region)
        ssm     = session.client("ssm",            region_name=region)
    except NoCredentialsError:
        fail("AWS credentials not configured — run: aws configure")
        return False, ["aws-credentials"]

    # ── Secrets Manager ───────────────────────────────────────────────────
    print("\n  " + BOLD + "Secrets Manager" + RESET)
    secrets_needed = {
        "openai":           "OPENAI_API_KEY for LLM calls",
        "pinecone":         "Pinecone API key for vector search",
        "neo4j":            "Neo4j credentials for knowledge graph",
        "postgres":         "RDS Postgres for HITL checkpoints",
        "platform_api_key": "Platform API authentication",
    }
    for secret, desc in secrets_needed.items():
        full_name = SSM_PREFIX + "/" + secret
        try:
            sm.get_secret_value(SecretId=full_name)
            ok(full_name + dim("  # " + desc))
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("ResourceNotFoundException", "SecretNotFoundException"):
                fail(full_name + "  \u2192  missing")
                issues.append(("secrets/" + secret, "python deploy.py secrets"))
            else:
                warn(full_name + "  access error: " + str(e))

    # ── SSM: Bedrock Prompts ──────────────────────────────────────────────
    print("\n  " + BOLD + "Bedrock Prompts (SSM)" + RESET)
    prompt_params = []
    for agent in AGENT_NAMES:
        prompt_params.append("/" + agent + "-agent/prod/bedrock/prompt_id")
        prompt_params.append("/" + agent + "-agent/prod/bedrock/prompt_version")

    found_prompts = set()
    for i in range(0, len(prompt_params), 10):
        batch  = prompt_params[i:i+10]
        result = ssm.get_parameters(Names=batch, WithDecryption=False)
        for p in result.get("Parameters", []):
            found_prompts.add(p["Name"])

    missing_agents = []
    for agent in AGENT_NAMES:
        pid  = "/" + agent + "-agent/prod/bedrock/prompt_id"
        pver = "/" + agent + "-agent/prod/bedrock/prompt_version"
        if pid in found_prompts and pver in found_prompts:
            ok(agent + "-agent  prompt \u2713")
        else:
            fail(agent + "-agent  prompt missing")
            missing_agents.append(agent)

    if missing_agents:
        issues.append(("prompts/" + ",".join(missing_agents), "python deploy.py prompts"))

    # ── SSM: Gateway URL ──────────────────────────────────────────────────
    print("\n  " + BOLD + "MCP Gateway (SSM)" + RESET)
    gw_key = SSM_PREFIX + "/mcp/gateway_url"
    try:
        gw_val = ssm.get_parameter(Name=gw_key)["Parameter"]["Value"]
        ok(gw_key + "  = " + gw_val[:60] + ("..." if len(gw_val) > 60 else ""))
    except ClientError:
        fail(gw_key + "  \u2192  missing")
        issues.append(("mcp/gateway_url", "python deploy.py gateway"))

    # ── SSM: Agent Registry ───────────────────────────────────────────────
    reg_key = SSM_PREFIX + "/agents/registry"
    try:
        reg_val = ssm.get_parameter(Name=reg_key)["Parameter"]["Value"]
        agents_in_registry = [a["name"] for a in json.loads(reg_val)]
        ok(reg_key + "  agents=" + str(agents_in_registry))
    except ClientError:
        fail(reg_key + "  \u2192  missing")
        issues.append(("agents/registry", "python deploy.py registry"))
    except Exception as e:
        warn(reg_key + "  parse error: " + str(e))

    # ── SSM: Bedrock Guardrail ────────────────────────────────────────────
    print("\n  " + BOLD + "Bedrock Guardrail (SSM)" + RESET)
    guard_keys = [SSM_PREFIX + "/bedrock/guardrail_id",
                  SSM_PREFIX + "/bedrock/guardrail_version"]
    result = ssm.get_parameters(Names=guard_keys, WithDecryption=False)
    found_guards = {p["Name"]: p["Value"] for p in result.get("Parameters", [])}

    if len(found_guards) == 2:
        gid  = found_guards[SSM_PREFIX + "/bedrock/guardrail_id"]
        gver = found_guards[SSM_PREFIX + "/bedrock/guardrail_version"]
        ok("guardrail_id=" + gid + "  version=" + gver)
    else:
        fail("guardrail_id or guardrail_version missing from SSM")
        issues.append(("bedrock/guardrail", "python deploy.py guardrails"))

    # ── SSM: Misc params ──────────────────────────────────────────────────
    print("\n  " + BOLD + "Misc SSM Params" + RESET)
    misc_keys = [
        SSM_PREFIX + "/dynamodb/trace_table_name",
        SSM_PREFIX + "/pinecone/clinical_trials_index",
    ]
    result     = ssm.get_parameters(Names=misc_keys, WithDecryption=False)
    found_misc = {p["Name"]: p["Value"] for p in result.get("Parameters", [])}
    missing_misc = [k for k in misc_keys if k not in found_misc]

    for k, v in found_misc.items():
        ok(k.split("/")[-1] + " = " + v)
    for k in missing_misc:
        fail(k + "  \u2192  missing")
        issues.append((k, "python deploy.py secrets"))

    # ── Connectivity checks (warn only) ───────────────────────────────────
    print("\n  " + BOLD + "Connectivity (warn-only)" + RESET)

    postgres_url = os.environ.get("POSTGRES_URL", "")
    if postgres_url:
        import socket
        from urllib.parse import urlparse
        try:
            pg = urlparse(postgres_url)
            with socket.create_connection((pg.hostname, pg.port or 5432), timeout=3):
                ok("Postgres TCP reachable  " + pg.hostname)
        except Exception as e:
            warn("Postgres not reachable: " + str(e) + "  (agents may fail at cold start)")
            warnings.append("postgres-tcp")
    else:
        warn("POSTGRES_URL not set — Supervisor HITL checkpoints will fail")
        warnings.append("postgres-url")

    if os.environ.get("PINECONE_API_KEY"):
        ok("PINECONE_API_KEY set \u2713")
    else:
        warn("PINECONE_API_KEY not in environment — Research/Knowledge agents will fail")
        warnings.append("pinecone-key")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    sep()
    if not issues:
        ok("All required AWS resources exist — ready to start")
        if warnings:
            warn(str(len(warnings)) + " connectivity warning(s) — agents may degrade at runtime")
        sep()
        return True, []
    else:
        fail(str(len(issues)) + " required resource(s) missing — fix before starting:\n")
        seen_fixes = set()
        for resource, fix in issues:
            if fix not in seen_fixes:
                print("      " + fix)
                seen_fixes.add(fix)
        print()
        sep()
        return False, issues

# ── Process management ────────────────────────────────────────────────────

_procs = []

def _stream(proc, label):
    c = COL.get(label, "")
    for line in proc.stdout:
        sys.stdout.write(c + "[" + label.ljust(10) + "]" + RESET + " " + line)
        sys.stdout.flush()

def spawn(name, cmd, cwd, extra_env):
    full_env = {**os.environ, **extra_env}
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd            = str(cwd),
        env            = full_env,
        stdout         = subprocess.PIPE,
        stderr         = subprocess.STDOUT,
        text           = True,
        bufsize        = 1,
    )
    threading.Thread(target=_stream, args=(proc, name), daemon=True).start()
    _procs.append(proc)
    return proc

def stop_all():
    hdr("Stopping all services...")
    for p in _procs:
        if p.poll() is None:
            p.terminate()
    time.sleep(1)
    for p in _procs:
        if p.poll() is None:
            p.kill()
    print("All stopped.")

def _on_signal(sig, frame):
    stop_all()
    sys.exit(0)

# ── Commands ──────────────────────────────────────────────────────────────

def cmd_status():
    hdr("Service Status")
    rows = [(a["name"], a["port"]) for a in AGENTS] + [("platform", 8080), ("ui", 8501)]
    for name, port in rows:
        status = "UP  \u2705" if port_open(port) else "DOWN \u274c"
        print("  " + name.ljust(12) + "  :" + str(port) + "  " + status)


def cmd_start(start_platform=True, skip_preflight=False, env_path=None, postgres_url=None):
    hdr("VS AgentCore — Local Development")
    sep()

    # Load and merge .env.prod
    loaded = load_env(env_path)
    os.environ.update(loaded)

    # --postgres-url flag overrides .env.prod value.
    # Useful when RDS is recreated by Terraform and the endpoint changes
    # without needing to edit .env.prod before every local run.
    if postgres_url:
        os.environ["POSTGRES_URL"] = postgres_url
        ok("POSTGRES_URL overridden via --postgres-url flag")

    missing_creds = [k for k in ["OPENAI_API_KEY", "PINECONE_API_KEY"] if not os.environ.get(k)]
    if missing_creds:
        fail("Missing env vars: " + str(missing_creds))
        fail("Edit .env.prod and re-run")
        sys.exit(1)

    # Pre-flight
    if not skip_preflight:
        pf_ok, pf_issues = cmd_preflight()
        if not pf_ok:
            print()
            fail("Pre-flight failed — agents will crash at cold start.")
            fail("Fix the issues above, then re-run: python local.py start")
            info("To skip this check (not recommended): python local.py start --skip-preflight")
            sys.exit(1)
    else:
        warn("Pre-flight skipped — if AWS resources are missing, agents will fail at cold start")

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Start sub-agents in parallel
    hdr("Starting sub-agents...")
    for agent in AGENTS[:-1]:
        env_ov = {
            "AGENT_NAME": agent["name"],
            "AGENT_ENV":  "prod",
            **agent.get("extra_env", {}),
        }
        spawn(agent["name"], [PYTHON, "-m", agent["module"]], ROOT, env_ov)
        info("Launched " + agent["name"] + " -> :" + str(agent["port"]))

    print()
    up = all(wait_for_port(a["name"], a["port"]) for a in AGENTS[:-1])
    if not up:
        fail("Some sub-agents failed to start — check logs above")
        stop_all()
        sys.exit(1)

    # Start supervisor (after sub-agents are healthy)
    sup = AGENTS[-1]
    env_ov = {
        "AGENT_NAME": "supervisor",
        "AGENT_ENV":  "prod",
        **sup.get("extra_env", {}),
    }
    spawn(sup["name"], [PYTHON, "-m", sup["module"]], ROOT, env_ov)
    info("Launched supervisor -> :" + str(sup["port"]))
    wait_for_port("supervisor", sup["port"])

    # Start platform
    if start_platform:
        hdr("Starting platform API...")
        plat_env = {"AGENT_ENV": "prod", **PLATFORM.get("extra_env", {})}
        spawn(PLATFORM["name"], PLATFORM["cmd"], PLATFORM["cwd"], plat_env)
        wait_for_port("platform", PLATFORM["port"])

        # Start UI (proxies to platform on 8080)
        hdr("Starting UI...")
        ui_env = {**UI["extra_env"]}
        spawn(UI["name"], UI["cmd"], UI["cwd"], ui_env)
        wait_for_port("ui", UI["port"])

    # Summary
    hdr("All services running")
    sep()
    for agent in AGENTS:
        print("  " + col(agent["name"], agent["name"].ljust(14)) +
              "  http://localhost:" + str(agent["port"]))
    if start_platform:
        print("  " + col("platform", "platform".ljust(14)) +
              "  http://localhost:8080")
        print("  " + col("ui", "ui".ljust(14)) +
              "  http://localhost:8501")

    api_key = os.environ.get("PLATFORM_API_KEY", "<not set>")
    print()
    info("Platform API key : " + api_key[:16] + "...")
    info("Observability    : http://localhost:8080/observability")
    print()
    info("Press Ctrl+C to stop all services")
    sep()

    # Keep alive
    try:
        while True:
            time.sleep(2)
            dead = [p for p in _procs if p.poll() is not None]
            if dead:
                fail(str(len(dead)) + " process(es) died — check logs above")
    except KeyboardInterrupt:
        stop_all()


def cmd_test(category="all"):
    try:
        import requests
    except ImportError:
        fail("requests not installed — run: pip install requests")
        sys.exit(1)

    loaded  = load_env()
    os.environ.update(loaded)
    api_key = os.environ.get("PLATFORM_API_KEY", "")
    if not api_key:
        fail("PLATFORM_API_KEY not set")
        sys.exit(1)

    base    = "http://localhost:8080/api/v1/clinical-trial"
    headers = {"Content-Type": "application/json", "X-API-Key": api_key}

    try:
        import requests as _req
        health = _req.get("http://localhost:8080/health", timeout=5)
        if health.status_code != 200:
            fail("Platform health check failed — is local.py start running?")
            sys.exit(1)
        probe = _req.post(base + "/chat",
                          json={"message": "ping", "thread_id": "probe", "domain": "pharma"},
                          headers=headers, timeout=5, stream=True)
        if probe.status_code == 404:
            fail("HTTP 404 on chat endpoint — checking available routes...")
            routes = _req.get("http://localhost:8080/openapi.json", timeout=5)
            if routes.ok:
                import json as _json
                paths = list(_json.loads(routes.text).get("paths", {}).keys())
                info("Available routes: " + str(paths))
                fail("Expected: /api/v1/clinical-trial/chat")
                fail("Check AGENT variable in platform/main.py")
            sys.exit(1)
    except Exception as e:
        fail("Cannot reach platform: " + str(e))
        fail("Run: python3.11 local.py start")
        sys.exit(1)

    def post_stream(endpoint, body, label):
        hdr("TEST: " + label)
        try:
            r = requests.post(base + "/" + endpoint, json=body,
                              headers=headers, stream=True, timeout=120)
            if r.status_code != 200:
                fail("HTTP " + str(r.status_code) + ": " + r.text[:200])
                return None
            result = []
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                event = json.loads(raw[6:])
                etype = event.get("type", "")
                if etype == "token":
                    chunk = event.get("content", "")
                    print(chunk, end="", flush=True)
                    result.append(chunk)
                elif etype == "interrupt":
                    print()
                    print("[HITL] " + str(event.get("question")))
                    for i, opt in enumerate(event.get("options", []), 1):
                        print("  " + str(i) + ". " + opt)
                    return {"type": "interrupt", "event": event}
                elif etype == "done":
                    latency_s = round(event.get("latency_ms", 0) / 1000, 1)
                    print("\n")
                    print(col("supervisor", "Done — " + str(latency_s) + "s"))
                elif etype == "error":
                    print()
                    fail(event.get("message", "unknown error"))
            return "".join(result)
        except Exception as e:
            fail("Request failed: " + str(e))
            return None

    def test_research():
        post_stream("chat",
            {"message": "What are Phase 3 results for mRNA-1273?",
             "thread_id": "test-research-" + str(int(time.time())),
             "domain": "pharma"},
            "Research — mRNA-1273 Phase 3 results")

    def test_knowledge():
        post_stream("chat",
            {"message": "Which cancer trials are in the knowledge base?",
             "thread_id": "test-knowledge-" + str(int(time.time())),
             "domain": "pharma"},
            "Knowledge — cancer trials")

    def test_chart():
        post_stream("chat",
            {"message": "Compare efficacy across COVID-19 vaccine trials and show me a chart",
             "thread_id": "test-chart-" + str(int(time.time())),
             "domain": "pharma"},
            "Chart — COVID vaccine efficacy")

    def test_hitl():
        import requests as req
        thread_id = "test-hitl-" + str(int(time.time()))
        hdr("TEST: HITL — vague query + resume")
        r = req.post(base + "/chat",
                     json={"message": "What were the efficacy results?",
                           "thread_id": thread_id, "domain": "pharma"},
                     headers=headers, stream=True, timeout=60)
        selection = None
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            event = json.loads(raw[6:])
            if event.get("type") == "interrupt":
                opts = event.get("options", [])
                print("[HITL] " + str(event.get("question")))
                for i, o in enumerate(opts, 1):
                    print("  " + str(i) + ". " + o)
                selection = opts[0] if opts else "NCT03374254"
                break

        if not selection:
            fail("No HITL interrupt received")
            return

        info("Resuming with: " + selection)
        r2 = req.post(base + "/resume",
                      json={"thread_id": thread_id, "user_answer": selection, "domain": "pharma"},
                      headers=headers, stream=True, timeout=120)
        for raw in r2.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            event = json.loads(raw[6:])
            if event.get("type") == "token":
                print(event.get("content", ""), end="", flush=True)
            elif event.get("type") == "done":
                latency_s = round(event.get("latency_ms", 0) / 1000, 1)
                print("\n")
                print(col("supervisor", "Done — " + str(latency_s) + "s"))

    def test_direct():
        import requests as req
        hdr("TEST: Direct — Supervisor (bypass platform)")
        r = req.post("http://localhost:8000/invocations",
                     json={"message": "What trials target breast cancer?",
                           "thread_id": "direct-" + str(int(time.time())),
                           "domain": "pharma"},
                     stream=True, timeout=120)
        for raw in r.iter_lines(decode_unicode=True):
            if raw and raw.startswith("data: "):
                event = json.loads(raw[6:])
                if event.get("type") == "token":
                    print(event.get("content", ""), end="", flush=True)

        hdr("TEST: Direct — Research Agent")
        r = req.post("http://localhost:8001/invocations",
                     json={"message": "What are Phase 3 results for mRNA-1273?",
                           "session_id": "r-" + str(int(time.time())),
                           "domain": "pharma"},
                     stream=True, timeout=60)
        for raw in r.iter_lines(decode_unicode=True):
            if raw and raw.startswith("data: "):
                event = json.loads(raw[6:])
                if event.get("type") == "token":
                    print(event.get("content", ""), end="", flush=True)

    dispatch = {
        "research":  test_research,
        "knowledge": test_knowledge,
        "chart":     test_chart,
        "hitl":      test_hitl,
        "direct":    test_direct,
    }

    if category == "all":
        for fn in [test_research, test_knowledge, test_chart, test_hitl]:
            fn()
            print()
    elif category in dispatch:
        dispatch[category]()
    else:
        fail("Unknown test: " + category)
        info("Available: " + ", ".join(dispatch))
        sys.exit(1)


def cmd_clean():
    hdr("Clean — Postgres checkpoints + DynamoDB traces")
    loaded = load_env()
    os.environ.update(loaded)

    postgres_url = os.environ.get("POSTGRES_URL", "")
    if not postgres_url:
        fail("POSTGRES_URL not set — skipping Postgres")
    else:
        try:
            import psycopg
            with psycopg.connect(postgres_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM checkpoints")
                    cur.execute("DELETE FROM checkpoint_writes")
                conn.commit()
            ok("Postgres checkpoints cleared")
        except Exception as e:
            fail("Postgres cleanup failed: " + str(e))

    try:
        import boto3
        region = os.environ.get("AWS_REGION", "us-east-1")
        ddb    = boto3.resource("dynamodb", region_name=region)
        table  = ddb.Table("vs-agentcore-ma-traces")
        items  = table.scan()["Items"]
        with table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={"run_id": item["run_id"]})
        ok("DynamoDB — cleared " + str(len(items)) + " traces")
    except Exception as e:
        fail("DynamoDB cleanup failed: " + str(e))


# ── CLI dispatch ──────────────────────────────────────────────────────────

USAGE = """
VS AgentCore — Local Development

Commands:
  preflight                     Check all required AWS resources exist
  start                         Preflight + start all agents + platform API
  start --no-platform           Start agents only (no platform)
  start --skip-preflight        Skip resource check (not recommended)
  start --env FILE              Use a custom env file
  start --postgres-url URL      Override POSTGRES_URL from .env.prod
  status                        Health-check all local ports
  test                          Run all test queries (needs platform running)
  test research                 Specific category: research / knowledge / chart / hitl / direct
  clean                         Clear Postgres checkpoints + DynamoDB traces

--postgres-url flag:
  Useful when RDS is recreated by Terraform and the endpoint changes.
  Overrides POSTGRES_URL without editing .env.prod.
  Example:
    python3.11 local.py start --postgres-url "postgresql://user:pass@host:5432/db"

What preflight checks:
  Secrets Manager  openai, pinecone, neo4j, postgres, platform_api_key
  SSM Prompts      prompt_id + prompt_version for all 5 agents
  SSM Gateway      mcp/gateway_url
  SSM Registry     agents/registry
  SSM Guardrail    bedrock/guardrail_id + guardrail_version
  SSM Misc         dynamodb/trace_table_name, pinecone index name
  Connectivity     Postgres TCP, Pinecone API key present

Ports:
  8000  supervisor    8001  research
  8002  knowledge     8003  safety
  8005  chart         8080  platform API
  8501  ui
"""

def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(USAGE)
        return

    cmd = args[0].lower()

    if cmd == "preflight":
        loaded = load_env()
        os.environ.update(loaded)
        ok_result, _ = cmd_preflight()
        sys.exit(0 if ok_result else 1)

    elif cmd == "start":
        no_platform    = "--no-platform"    in args
        skip_preflight = "--skip-preflight" in args
        env_path       = None
        postgres_url   = None

        if "--env" in args:
            idx      = args.index("--env")
            env_path = args[idx + 1] if idx + 1 < len(args) else None

        if "--postgres-url" in args:
            idx          = args.index("--postgres-url")
            postgres_url = args[idx + 1] if idx + 1 < len(args) else None
            if not postgres_url:
                fail("--postgres-url requires a value")
                sys.exit(1)

        cmd_start(
            start_platform = not no_platform,
            skip_preflight = skip_preflight,
            env_path       = env_path,
            postgres_url   = postgres_url,
        )

    elif cmd == "status":
        cmd_status()

    elif cmd == "test":
        category = args[1] if len(args) > 1 else "all"
        cmd_test(category)

    elif cmd == "clean":
        cmd_clean()

    else:
        fail("Unknown command: " + cmd)
        print(USAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()