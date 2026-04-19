#!/bin/bash
# scripts/deploy.sh — vs-agentcore-multiagent
# =============================================
# Multi-agent deployment. Steps:
#
#   ./scripts/deploy.sh prompts    # Step 0: create 6 Bedrock prompts → SSM
#   ./scripts/deploy.sh secrets    # Step 1: push secrets + SSM params
#   ./scripts/deploy.sh iam        # Step 2: IAM roles (same as single agent)
#   ./scripts/deploy.sh lambdas    # Step 3: build + deploy 5 Lambda tools (+ chart)
#   ./scripts/deploy.sh gateway    # Step 4: MCP Gateway + 5 targets (+ chart)
#   ./scripts/deploy.sh platform   # Step 5: Terraform (ECS, ALB, RDS)
#   ./scripts/deploy.sh agents     # Step 6: build + deploy 6 AgentCore Runtimes
#   ./scripts/deploy.sh ssm-arns   # Step 7: write sub-agent ARNs to SSM (Supervisor reads these)
#   ./scripts/deploy.sh all        # All steps in order
#   ./scripts/deploy.sh redeploy   # Quick ECS redeploy after platform code changes
#
# DEPLOYMENT ORDER MATTERS:
#   Sub-agents deployed BEFORE Supervisor because:
#   - step_ssm_arns writes sub-agent ARNs to SSM after step_agents
#   - Supervisor reads those ARNs at cold start via a2a_tools.py
#   - If Supervisor starts before sub-agents, _get_runtime_arns() fails
#
# KEY DIFFERENCE from single agent deploy.sh:
#   step_prompts   — 6 Bedrock prompts instead of 1
#                    each agent reads its own prompt via AGENT_NAME env var
#   step_agents    — 6 AgentCore Runtimes instead of 1
#                    each gets AGENT_NAME env var set by this script
#   step_ssm_arns  — NEW: write sub-agent ARNs to SSM for Supervisor
#
# ARCHITECTURE NOTES (same as single agent):
#   AgentCore Runtimes → linux/arm64
#   Lambda MCP tools   → linux/amd64
#   ECS Platform/UI    → linux/amd64
#
# GATEWAY TARGET NAMING (same as single agent + chart):
#   "tool-search"    → "tool-search___search_tool"
#   "tool-graph"     → "tool-graph___graph_tool"
#   "clarify"        → "clarify___ask_user_input"
#   "tool-summariser"→ "tool-summariser___summariser_tool"
#   "chart"          → "chart___chart_tool"           ← NEW

set -euo pipefail

# ── Pre-requisite checks ───────────────────────────────────────────────────
check_prereqs() {
  local missing=()
  command -v aws        &>/dev/null || missing+=("aws-cli")
  command -v docker     &>/dev/null || missing+=("docker")
  command -v terraform  &>/dev/null || missing+=("terraform")
  command -v python3    &>/dev/null || missing+=("python3")
  python3 -c "import boto3" &>/dev/null || missing+=("boto3 — run: pip install boto3")
  if [ ${#missing[@]} -gt 0 ]; then
    echo "❌ Missing prerequisites: ${missing[*]}"
    exit 1
  fi
  aws sts get-caller-identity &>/dev/null || {
    echo "❌ AWS credentials not configured."
    exit 1
  }
}
check_prereqs

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
PREFIX="vs-agentcore-ma"
SSM_PREFIX="/vs-agentcore-multiagent/prod"
GATEWAY_NAME="vs-agentcore-ma-mcp"
ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

ACTION="${1:-plan}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# All 6 agents — sub-agents first, supervisor last
AGENTS=(research knowledge hitl safety chart supervisor)
# Sub-agents only (Supervisor deployed last in step_agents)
SUB_AGENTS=(research knowledge hitl safety chart)

echo "================================================"
echo "VS AgentCore Multi-Agent — ${ACTION}"
echo "Account: ${ACCOUNT_ID}  Region: ${REGION}"
echo "Root:    ${ROOT}"
echo "================================================"

# ── Helpers ────────────────────────────────────────────────────────────────

ecr_login() {
  aws ecr get-login-password --region "${REGION}" | \
    docker login --username AWS --password-stdin "${ECR_BASE}"
}

ensure_ecr_repo() {
  local name="$1"
  aws ecr create-repository \
    --repository-name "${PREFIX}/${name}" \
    --region "${REGION}" > /dev/null 2>/dev/null || true
  echo "${ECR_BASE}/${PREFIX}/${name}"
}

wait_for_lambda() {
  local name="$1"
  echo -n "  Waiting for ${name}..."
  for i in {1..30}; do
    STATE=$(aws lambda get-function --function-name "${name}" \
      --region "${REGION}" --query 'Configuration.State' --output text 2>/dev/null || echo "NotFound")
    [ "${STATE}" = "Active" ] && echo " ✅" && return
    echo -n "."
    sleep 3
  done
  echo " ⚠️  timeout"
}

wait_for_runtime() {
  local runtime_id="$1"
  echo -n "  Waiting for READY..."
  for i in {1..60}; do
    STATUS=$(aws bedrock-agentcore-control get-agent-runtime \
      --region "${REGION}" \
      --agent-runtime-id "${runtime_id}" \
      --query "status" --output text 2>/dev/null || echo "UNKNOWN")
    [ "${STATUS}" = "READY" ] && echo " ✅" && return
    echo -n "."
    sleep 5
  done
  echo " ⚠️  timeout — check AWS Console"
}

deploy_runtime() {
  # Deploy one AgentCore Runtime for one agent.
  # Args: $1=agent_name  $2=description  $3=image_tag
  local agent_name="$1"
  local description="$2"
  local image_tag="$3"
  local runtime_name="${PREFIX//-/_}_${agent_name//-/_}"
  local agent_role="arn:aws:iam::${ACCOUNT_ID}:role/${PREFIX}-agent-role"

  # ENV_JSON: SSM_PREFIX, AWS_REGION, AGENT_NAME (KEY DIFFERENCE — drives prompt + config)
  local env_json="{\"SSM_PREFIX\":\"${SSM_PREFIX}\",\"AWS_REGION\":\"${REGION}\",\"AWS_DEFAULT_REGION\":\"${REGION}\",\"AGENT_ENV\":\"prod\",\"AGENT_NAME\":\"${agent_name}-agent\"}"

  echo ""
  echo "  ── ${agent_name} agent"

  EXISTING_ARN=$(aws bedrock-agentcore-control list-agent-runtimes \
    --region "${REGION}" \
    --query "agentRuntimes[?agentRuntimeName=='${runtime_name}'].agentRuntimeArn | [0]" \
    --output text 2>/dev/null || echo "")

  local runtime_arn=""

  if [ -n "${EXISTING_ARN}" ] && [ "${EXISTING_ARN}" != "None" ]; then
    echo "  Runtime exists — updating..."
    RUNTIME_ID=$(echo "${EXISTING_ARN}" | sed 's/.*runtime\///')
    aws bedrock-agentcore-control update-agent-runtime \
      --region "${REGION}" \
      --agent-runtime-id "${RUNTIME_ID}" \
      --role-arn "${agent_role}" \
      --network-configuration "{\"networkMode\":\"PUBLIC\"}" \
      --agent-runtime-artifact "{\"containerConfiguration\":{\"containerUri\":\"${image_tag}\"}}" \
      --environment-variables "${env_json}" > /dev/null
    runtime_arn="${EXISTING_ARN}"
    wait_for_runtime "${RUNTIME_ID}"
  else
    echo "  Creating AgentCore Runtime: ${runtime_name}..."
    RUNTIME_RESPONSE=$(aws bedrock-agentcore-control create-agent-runtime \
      --region "${REGION}" \
      --agent-runtime-name "${runtime_name}" \
      --description "${description}" \
      --role-arn "${agent_role}" \
      --agent-runtime-artifact "{\"containerConfiguration\":{\"containerUri\":\"${image_tag}\"}}" \
      --network-configuration "{\"networkMode\":\"PUBLIC\"}" \
      --environment-variables "${env_json}" 2>/dev/null || echo "{}")

    runtime_arn=$(echo "${RUNTIME_RESPONSE}" | python3 -c \
      "import sys,json; d=json.load(sys.stdin); print(d.get('agentRuntimeArn',''))" 2>/dev/null || echo "")

    if [ -z "${runtime_arn}" ] || [ "${runtime_arn}" = "None" ]; then
      echo "  ❌ Could not create ${agent_name} runtime — check AWS Console"
      exit 1
    fi

    RUNTIME_ID=$(echo "${runtime_arn}" | sed 's/.*runtime\///')
    wait_for_runtime "${RUNTIME_ID}"
  fi

  echo "  ARN: ${runtime_arn}"

  # Write ARN to SSM so Supervisor can discover sub-agents at cold start
  aws ssm put-parameter \
    --name "${SSM_PREFIX}/agents/${agent_name}/runtime_arn" \
    --value "${runtime_arn}" \
    --type String --overwrite \
    --region "${REGION}" > /dev/null
  echo "  ✅ SSM: ${SSM_PREFIX}/agents/${agent_name}/runtime_arn"
}


# ── Step 0: Bedrock Prompts (6 prompts — one per agent) ───────────────────

step_prompts() {
  echo ""
  echo "► Step 0: Bedrock Prompts (6 agents)"

  # Prompt files must exist in prompts/ directory
  # One file per agent: prompts/supervisor.txt, prompts/research.txt, etc.
  PROMPTS_DIR="${ROOT}/prompts"
  if [ ! -d "${PROMPTS_DIR}" ]; then
    mkdir -p "${PROMPTS_DIR}"
    echo "  Created prompts/ directory"
    echo "  Add prompt files: supervisor.txt research.txt knowledge.txt hitl.txt safety.txt chart.txt"
    echo "  Then re-run: ./scripts/deploy.sh prompts"
    exit 1
  fi

  python3 - << PYEOF
import boto3, json, os, sys

region     = "${REGION}"
ssm_prefix = "${SSM_PREFIX}"
client     = boto3.client("bedrock-agent", region_name=region)
ssm_client = boto3.client("ssm",           region_name=region)

# One prompt per agent — each agent reads its own via AGENT_NAME env var
agents = ["supervisor", "research", "knowledge", "hitl", "safety", "chart"]

for agent in agents:
    prompt_file = f"${PROMPTS_DIR}/{agent}.txt"
    if not os.path.exists(prompt_file):
        print(f"  ⏭  Skipping {agent} — {prompt_file} not found")
        continue

    prompt_text = open(prompt_file).read()
    app_name    = f"{agent}-agent"  # e.g. "research-agent"

    # SSM paths this agent reads at cold start via get_bedrock_prompt(AGENT_NAME):
    #   /{app_name}/prod/bedrock/prompt_id
    #   /{app_name}/prod/bedrock/prompt_version
    ssm_id_key  = f"/{app_name}/prod/bedrock/prompt_id"
    ssm_ver_key = f"/{app_name}/prod/bedrock/prompt_version"

    # Check if prompt already exists
    existing_id = None
    try:
        existing_id = ssm_client.get_parameter(Name=ssm_id_key)["Parameter"]["Value"]
        if existing_id in ("", "CHANGE_ME"):
            existing_id = None
    except ssm_client.exceptions.ParameterNotFound:
        pass

    if existing_id:
        print(f"  [{agent}] Creating new version of prompt {existing_id}...")
        resp = client.create_prompt_version(
            promptIdentifier=existing_id,
            description=f"Deployed by deploy.sh — {agent}-agent"
        )
        prompt_id      = existing_id
        prompt_version = str(resp["version"])
    else:
        print(f"  [{agent}] Creating new Bedrock prompt...")
        resp = client.create_prompt(
            name=f"vs-agentcore-ma-{agent}",
            description=f"{agent.capitalize()} Agent system prompt",
            variants=[{
                "name":         "default",
                "modelId":      "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "templateType": "TEXT",
                "templateConfiguration": {
                    "text": {"text": prompt_text, "inputVariables": []}
                },
                "inferenceConfiguration": {
                    "text": {"temperature": 0.0, "maxTokens": 4096}
                }
            }],
            defaultVariant="default"
        )
        prompt_id = resp["id"]
        v_resp    = client.create_prompt_version(
            promptIdentifier=prompt_id,
            description=f"Initial version — {agent}-agent"
        )
        prompt_version = str(v_resp["version"])

    # Write to SSM — agent reads these at cold start
    ssm_client.put_parameter(Name=ssm_id_key,  Value=prompt_id,      Type="String", Overwrite=True)
    ssm_client.put_parameter(Name=ssm_ver_key, Value=prompt_version, Type="String", Overwrite=True)
    print(f"  [{agent}] ✅  prompt_id={prompt_id}  version={prompt_version}")

print("")
print("  Prompts done ✅")
PYEOF
}


# ── Step 1: Secrets + SSM ─────────────────────────────────────────────────
# Same as single agent — just update SSM_PREFIX

step_secrets() {
  echo ""
  echo "► Step 1: Secrets + SSM"

  python3 - << PYEOF
import boto3, json, os, sys
from urllib.parse import urlparse

region     = "${REGION}"
ssm_prefix = "${SSM_PREFIX}"
sm         = boto3.client("secretsmanager", region_name=region)
ssm_client = boto3.client("ssm",            region_name=region)

def put_secret(name, value):
    try:
        sm.create_secret(Name=name, SecretString=json.dumps(value))
        print(f"  ✅ Created secret: {name}")
    except sm.exceptions.ResourceExistsException:
        sm.update_secret(SecretId=name, SecretString=json.dumps(value))
        print(f"  ✅ Updated secret: {name}")

def put_param(name, value, secure=False):
    ssm_client.put_parameter(
        Name=name, Value=value,
        Type="SecureString" if secure else "String",
        Overwrite=True,
    )
    print(f"  ✅ SSM param: {name}")

required = ["OPENAI_API_KEY", "PINECONE_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "PLATFORM_API_KEY"]
missing  = [k for k in required if not os.environ.get(k)]
if missing:
    print(f"ERROR: Missing env vars: {missing}")
    print("Run: source .env.prod")
    sys.exit(1)

put_secret(f"{ssm_prefix}/openai",            {"api_key": os.environ["OPENAI_API_KEY"]})
put_secret(f"{ssm_prefix}/pinecone",          {"api_key": os.environ["PINECONE_API_KEY"]})
put_secret(f"{ssm_prefix}/neo4j",             {"uri": os.environ["NEO4J_URI"], "user": os.environ["NEO4J_USER"], "password": os.environ["NEO4J_PASSWORD"]})
put_secret(f"{ssm_prefix}/platform_api_key",  {"api_key": os.environ["PLATFORM_API_KEY"]})

# Postgres — only push if RDS endpoint is known (after step_platform)
postgres_url = os.environ.get("POSTGRES_URL", "")
if postgres_url and "<rds-endpoint>" not in postgres_url:
    pg = urlparse(postgres_url)
    put_secret(f"{ssm_prefix}/postgres", {
        "username": pg.username, "password": pg.password,
        "host": pg.hostname, "port": str(pg.port or 5432),
        "dbname": pg.path.lstrip("/"),
    })
    print("  ✅ Postgres secret written")
else:
    print("  ⏭  Skipping postgres — fill POSTGRES_URL after step_platform")

# Pinecone SSM params (core/aws.py reads these)
put_param("/clinical-agent/prod/pinecone/api_key",   os.environ["PINECONE_API_KEY"], secure=True)
put_param("/clinical-agent/prod/pinecone/index_name", os.environ.get("PINECONE_INDEX_NAME", "clinical-agent"))

put_param(f"{ssm_prefix}/pinecone/clinical_trials_index", os.environ.get("CLINICAL_TRIALS_INDEX", "clinical-trials-index"))
put_param(f"{ssm_prefix}/dynamodb/trace_table_name",      "${PREFIX}-traces")
put_param("/clinical-agent/prod/dynamodb/trace_table_name", "${PREFIX}-traces")

print("")
print("  Secrets done ✅")
PYEOF
}


# ── Step 2: IAM ────────────────────────────────────────────────────────────
# Same as single agent — just update resource names

step_iam() {
  echo ""
  echo "► Step 2: IAM roles"

  # Lambda role (same as single agent)
  cat > /tmp/lambda-trust.json << 'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
  aws iam create-role --role-name "${PREFIX}-lambda-mcp" \
    --assume-role-policy-document file:///tmp/lambda-trust.json 2>/dev/null || true
  aws iam attach-role-policy --role-name "${PREFIX}-lambda-mcp" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
  cat > /tmp/lambda-secrets.json << POLICY
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":["secretsmanager:GetSecretValue","ssm:GetParameter","kms:Decrypt"],"Resource":"*"}]}
POLICY
  aws iam put-role-policy --role-name "${PREFIX}-lambda-mcp" \
    --policy-name SecretsAccess \
    --policy-document file:///tmp/lambda-secrets.json
  echo "  ✅ ${PREFIX}-lambda-mcp"

  # Gateway role (updated with chart Lambda)
  cat > /tmp/gateway-trust.json << 'JSON'
{"Version":"2012-10-17","Statement":[{"Sid":"GatewayAssumeRolePolicy","Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
  aws iam create-role --role-name "${PREFIX}-gateway-role" \
    --assume-role-policy-document file:///tmp/gateway-trust.json 2>/dev/null || true
  cat > /tmp/gateway-policy.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunction"],
      "Resource": [
        "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-search-tool",
        "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-graph-tool",
        "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-hitl-tool",
        "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-summariser-tool",
        "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${PREFIX}-chart-tool"
      ]
    },
    {"Effect":"Allow","Action":["logs:CreateLogGroup"],"Resource":"*"},
    {"Effect":"Allow","Action":["logs:CreateLogStream","logs:PutLogEvents","logs:DescribeLogStreams"],"Resource":"arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/bedrock-agentcore/runtimes/*:*"}
  ]
}
POLICY
  aws iam put-role-policy --role-name "${PREFIX}-gateway-role" \
    --policy-name GatewayPolicy \
    --policy-document file:///tmp/gateway-policy.json
  echo "  ✅ ${PREFIX}-gateway-role"

  # AgentCore Runtime role (added bedrock-agentcore:InvokeAgentRuntime for Supervisor→Sub-agent calls)
  cat > /tmp/agentcore-trust.json << 'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
  aws iam create-role --role-name "${PREFIX}-agent-role" \
    --assume-role-policy-document file:///tmp/agentcore-trust.json 2>/dev/null || true
  cat > /tmp/agentcore-policy.json << POLICY
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid":"SecretsAndConfig","Effect":"Allow","Action":["secretsmanager:GetSecretValue","ssm:GetParameter","ssm:PutParameter","kms:Decrypt"],"Resource":"*"},
    {"Sid":"BedrockGateway","Effect":"Allow","Action":["bedrock-agentcore:InvokeGateway"],"Resource":"*"},
    {"Sid":"BedrockAgentRuntime","Effect":"Allow","Action":["bedrock-agentcore:InvokeAgentRuntime"],"Resource":"arn:aws:bedrock-agentcore:${REGION}:${ACCOUNT_ID}:runtime/*"},
    {"Sid":"BedrockPrompt","Effect":"Allow","Action":["bedrock:GetPrompt"],"Resource":"*"},
    {"Sid":"DynamoDB","Effect":"Allow","Action":["dynamodb:PutItem","dynamodb:GetItem","dynamodb:UpdateItem","dynamodb:Scan","dynamodb:DescribeTable","dynamodb:CreateTable"],"Resource":"*"},
    {"Sid":"ECRAuth","Effect":"Allow","Action":["ecr:GetAuthorizationToken"],"Resource":"*"},
    {"Sid":"ECRPull","Effect":"Allow","Action":["ecr:BatchGetImage","ecr:GetDownloadUrlForLayer","ecr:BatchCheckLayerAvailability"],"Resource":"arn:aws:ecr:${REGION}:${ACCOUNT_ID}:repository/*"},
    {"Sid":"CloudWatchLogs","Effect":"Allow","Action":["logs:CreateLogGroup"],"Resource":"*"},
    {"Sid":"CloudWatchLogStreams","Effect":"Allow","Action":["logs:CreateLogStream","logs:PutLogEvents","logs:DescribeLogStreams"],"Resource":"arn:aws:logs:${REGION}:${ACCOUNT_ID}:log-group:/aws/bedrock-agentcore/runtimes/*:*"}
  ]
}
POLICY
  aws iam put-role-policy --role-name "${PREFIX}-agent-role" \
    --policy-name AgentCorePolicy \
    --policy-document file:///tmp/agentcore-policy.json
  echo "  ✅ ${PREFIX}-agent-role"
  echo "  (Added bedrock-agentcore:InvokeAgentRuntime for Supervisor→Sub-agent calls)"
  echo ""
  echo "  IAM done ✅"
}


# ── Step 3: Lambda tools (4 existing + chart NEW) ─────────────────────────

step_lambdas() {
  echo ""
  echo "► Step 3: Lambda MCP tools (search, graph, hitl, summariser, chart)"
  ecr_login

  LAMBDA_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/${PREFIX}-lambda-mcp"

  for tool in search graph hitl summariser chart; do
    echo ""
    echo "  ── ${tool}_lambda"
    REPO=$(ensure_ecr_repo "${tool}-tool")
    TAG="${REPO}:latest"
    FUNC="${PREFIX}-${tool}-tool"

    docker buildx build \
      --platform linux/amd64 \
      --output type=registry \
      --provenance=false \
      --no-cache \
      -t "${TAG}" \
      "${ROOT}/mcp_tools/${tool}_lambda"

    ENV_VARS="Variables={SSM_PREFIX=${SSM_PREFIX},AWS_REGION=${REGION}}"

    if aws lambda get-function --function-name "${FUNC}" --region "${REGION}" &>/dev/null; then
      aws lambda update-function-code \
        --function-name "${FUNC}" --image-uri "${TAG}" \
        --region "${REGION}" > /dev/null
    else
      aws lambda create-function \
        --function-name "${FUNC}" \
        --package-type Image \
        --code ImageUri="${TAG}" \
        --role "${LAMBDA_ROLE}" \
        --architectures x86_64 \
        --timeout 30 \
        --memory-size 512 \
        --image-config '{"Command":["handler.handler"]}' \
        --environment "${ENV_VARS}" \
        --region "${REGION}" > /dev/null
    fi

    wait_for_lambda "${FUNC}"
    echo "  ✅ ${FUNC}"
  done

  echo ""
  echo "  Lambdas done ✅"
}


# ── Step 4: MCP Gateway (same as single agent + chart target) ──────────────

step_gateway() {
  echo ""
  echo "► Step 4: MCP Gateway + targets"

  GATEWAY_ROLE="arn:aws:iam::${ACCOUNT_ID}:role/${PREFIX}-gateway-role"

  EXISTING_GW=$(aws bedrock-agentcore-control list-gateways \
    --region "${REGION}" \
    --query "items[?name=='${GATEWAY_NAME}'].gatewayId | [0]" \
    --output text 2>/dev/null || echo "")

  if [ -n "${EXISTING_GW}" ] && [ "${EXISTING_GW}" != "None" ]; then
    echo "  Gateway exists: ${EXISTING_GW}"
    GATEWAY_ID="${EXISTING_GW}"
    GATEWAY_URL=$(aws bedrock-agentcore-control get-gateway \
      --region "${REGION}" --gateway-identifier "${GATEWAY_ID}" \
      --query "gatewayUrl" --output text 2>/dev/null || echo "")
  else
    echo "  Creating gateway ${GATEWAY_NAME}..."
    RESPONSE=$(aws bedrock-agentcore-control create-gateway \
      --region "${REGION}" \
      --name "${GATEWAY_NAME}" \
      --authorizer-type AWS_IAM \
      --protocol-type MCP \
      --role-arn "${GATEWAY_ROLE}" \
      --protocol-configuration "{\"mcp\":{\"supportedVersions\":[\"2025-03-26\"],\"instructions\":\"VS AgentCore Multi-Agent MCP Gateway\"}}")

    GATEWAY_ID=$(echo "${RESPONSE}"  | python3 -c "import sys,json; print(json.load(sys.stdin)['gatewayId'])")
    GATEWAY_URL=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['gatewayUrl'])")

    echo -n "  Waiting for ACTIVE..."
    for i in {1..24}; do
      STATUS=$(aws bedrock-agentcore-control get-gateway \
        --region "${REGION}" --gateway-identifier "${GATEWAY_ID}" \
        --query 'status' --output text 2>/dev/null || echo "UNKNOWN")
      [ "${STATUS}" = "ACTIVE" ] && echo " ✅" && break
      echo -n "."
      sleep 5
    done

    register_target() {
      local tgt_name="$1" tool_name="$2" tool_desc="$3" lambda_func="$4" schema="$5"
      echo "  Registering ${tgt_name}..."
      aws bedrock-agentcore-control create-gateway-target \
        --region "${REGION}" \
        --gateway-identifier "${GATEWAY_ID}" \
        --name "${tgt_name}" \
        --description "${tool_desc}" \
        --target-configuration "{\"mcp\":{\"lambda\":{\"lambdaArn\":\"arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${lambda_func}\",\"toolSchema\":{\"inlinePayload\":[{\"name\":\"${tool_name}\",\"description\":\"${tool_desc}\",\"inputSchema\":${schema}}]}}}}" \
        --credential-provider-configurations "[{\"credentialProviderType\":\"GATEWAY_IAM_ROLE\"}]" > /dev/null
      echo "  ✅ ${tgt_name}"
    }

    # Same 4 targets as single agent
    register_target "tool-search" "search_tool" \
      "Semantic search over 5,772 clinical trial chunks in Pinecone." \
      "${PREFIX}-search-tool" \
      '{"type":"object","properties":{"query":{"type":"string"},"top_k":{"type":"integer"}},"required":["query"]}'

    register_target "tool-graph" "graph_tool" \
      "Cypher query on Neo4j biomedical graph. Read-only." \
      "${PREFIX}-graph-tool" \
      '{"type":"object","properties":{"cypher":{"type":"string"}},"required":["cypher"]}'

    register_target "clarify" "ask_user_input" \
      "Ask user to clarify an ambiguous query. Options must be real trial names from search/graph." \
      "${PREFIX}-hitl-tool" \
      '{"type":"object","properties":{"question":{"type":"string"},"options":{"type":"array","items":{"type":"string"}},"allow_freetext":{"type":"boolean"},"user_answer":{"type":"string"}},"required":[]}'

    register_target "tool-summariser" "summariser_tool" \
      "Synthesise chunks into one cited answer. Call LAST after search/graph." \
      "${PREFIX}-summariser-tool" \
      '{"type":"object","properties":{"chunks":{"type":"array","items":{"type":"string"}},"query":{"type":"string"}},"required":["chunks"]}'

    # NEW: chart target
    register_target "chart" "chart_tool" \
      "Generate Chart.js config from numerical clinical trial data. Returns chart JSON for inline UI rendering." \
      "${PREFIX}-chart-tool" \
      '{"type":"object","properties":{"data":{"type":"string","description":"Text or JSON with numerical data to visualise"},"chart_type":{"type":"string","enum":["auto","bar","line","pie","doughnut"],"default":"auto"},"title":{"type":"string"}},"required":["data"]}'
  fi

  aws ssm put-parameter \
    --name "${SSM_PREFIX}/mcp/gateway_url" \
    --value "${GATEWAY_URL}" \
    --type String --overwrite \
    --region "${REGION}" > /dev/null

  echo ""
  echo "  Gateway done ✅  URL: ${GATEWAY_URL}"
}


# ── Step 5: Platform + UI (Terraform) ────────────────────────────────────

step_platform() {
  echo ""
  echo "► Step 5: Platform + UI (ECS Fargate)"
  ecr_login

  PLATFORM_REPO=$(ensure_ecr_repo "platform")
  PLATFORM_TAG="${PLATFORM_REPO}:latest"
  docker buildx build --platform linux/amd64 --output type=registry \
    --provenance=false --no-cache -t "${PLATFORM_TAG}" "${ROOT}/platform"

  UI_REPO=$(ensure_ecr_repo "ui")
  UI_TAG="${UI_REPO}:latest"
  docker buildx build --platform linux/amd64 --output type=registry \
    --provenance=false --no-cache -t "${UI_TAG}" "${ROOT}/ui"

  cd "${ROOT}/infra"
  aws s3 mb s3://${PREFIX}-tfstate --region "${REGION}" 2>/dev/null || true

  if [ -z "${RDS_PASSWORD:-}" ]; then
    echo "  ❌ RDS_PASSWORD not set in .env.prod"
    exit 1
  fi
  export TF_VAR_postgres_password="${RDS_PASSWORD}"

  terraform init -upgrade -input=false
  TF_VARS="-var=platform_image_uri=${PLATFORM_TAG} -var=ui_image_uri=${UI_TAG} -var=aws_region=${REGION}"

  if [ "${ACTION}" = "plan" ]; then
    terraform plan ${TF_VARS}
  else
    terraform apply -auto-approve -input=false ${TF_VARS}
    ALB_DNS=$(terraform output -raw alb_dns 2>/dev/null || echo "check-output")
    RDS_EP=$(terraform output -raw rds_endpoint 2>/dev/null || echo "check-output")
    echo ""
    echo "  Platform done ✅"
    echo "  ALB: http://${ALB_DNS}"
    echo "  RDS: ${RDS_EP}"
    echo ""
    echo "  ⚠️  Fill POSTGRES_URL in .env.prod then re-run:"
    echo "     source .env.prod && ./scripts/deploy.sh secrets"
  fi
}


# ── Step 6: AgentCore Runtimes (6 agents, sub-agents first) ───────────────

step_agents() {
  echo ""
  echo "► Step 6: AgentCore Runtimes (6 agents)"
  ecr_login

  build_and_push() {
    local agent="$1"
    local repo
    repo=$(ensure_ecr_repo "agent-${agent}")
    local tag="${repo}:latest"
    echo "  Building ${agent} (linux/arm64)..."
    docker buildx build \
      --platform linux/arm64 \
      --output type=registry \
      --provenance=false \
      --no-cache \
      -t "${tag}" \
      --build-arg AGENT_NAME="${agent}" \
      "${ROOT}/agents/${agent}"
    echo "  ✅ Pushed: ${tag}"
    echo "${tag}"
  }

  # Deploy sub-agents first — Supervisor reads their ARNs at cold start
  for agent in "${SUB_AGENTS[@]}"; do
    tag=$(build_and_push "${agent}")
    case "${agent}" in
      research)  desc="Research Agent — Pinecone semantic search + GPT-4o synthesis" ;;
      knowledge) desc="Knowledge Agent — Neo4j Cypher queries" ;;
      hitl)      desc="HITL Agent — clarification + NodeInterrupt" ;;
      safety)    desc="Safety Agent — faithfulness + consistency evaluation" ;;
      chart)     desc="Chart Agent — Chart.js generation from trial data" ;;
    esac
    deploy_runtime "${agent}" "${desc}" "${tag}"
  done

  # Supervisor last — all sub-agent ARNs now in SSM
  echo ""
  echo "  ── supervisor (deploying last — reads sub-agent ARNs from SSM)"
  tag=$(build_and_push "supervisor")
  deploy_runtime "supervisor" "Supervisor Agent — A2A routing + full middleware stack" "${tag}"

  echo ""
  echo "  Agents done ✅"
}


# ── Step 7: Write sub-agent ARNs to SSM ───────────────────────────────────
# Called automatically by deploy_runtime() — run manually to refresh

step_ssm_arns() {
  echo ""
  echo "► Step 7: Verify sub-agent ARNs in SSM"

  for agent in "${SUB_AGENTS[@]}"; do
    ARN=$(aws ssm get-parameter \
      --name "${SSM_PREFIX}/agents/${agent}/runtime_arn" \
      --region "${REGION}" \
      --query "Parameter.Value" --output text 2>/dev/null || echo "NOT SET")
    echo "  ${agent}: ${ARN}"
  done

  echo ""
  echo "  Supervisor reads these at cold start via a2a_tools._get_runtime_arns()"
}


# ── Quick ECS redeploy ────────────────────────────────────────────────────

step_redeploy() {
  local target="${2:-both}"
  echo ""
  echo "► Quick ECS redeploy — ${target}"

  if [ "${target}" = "platform" ] || [ "${target}" = "both" ]; then
    aws ecs update-service \
      --cluster "${PREFIX}-cluster" --service "${PREFIX}-platform" \
      --force-new-deployment --region "${REGION}" > /dev/null
    echo "  ✅ ${PREFIX}-platform redeploy triggered"
  fi

  if [ "${target}" = "ui" ] || [ "${target}" = "both" ]; then
    aws ecs update-service \
      --cluster "${PREFIX}-cluster" --service "${PREFIX}-ui" \
      --force-new-deployment --region "${REGION}" > /dev/null
    echo "  ✅ ${PREFIX}-ui redeploy triggered"
  fi
}


# ── Main dispatch ─────────────────────────────────────────────────────────

case "${ACTION}" in
  prompts)   step_prompts  ;;
  secrets)   step_secrets  ;;
  iam)       step_iam      ;;
  lambdas)   step_lambdas  ;;
  gateway)   step_gateway  ;;
  platform)  step_platform ;;
  plan)      step_platform ;;
  agents)    step_agents   ;;
  ssm-arns)  step_ssm_arns ;;
  redeploy)  step_redeploy "$@" ;;

  all)
    # ORDER:
    # 0. prompts  — 6 Bedrock prompts → SSM
    # 1. secrets  — API keys (postgres skipped until RDS exists)
    # 2. iam      — IAM roles
    # 3. lambdas  — 5 Lambda tools
    # 4. gateway  — MCP Gateway + 5 targets
    # 5. platform — Terraform: RDS, ECS, ALB ← MUST be before agents
    # 6. (manual) — fill POSTGRES_URL, re-run secrets
    # 7. agents   — 6 AgentCore Runtimes (sub-agents first, supervisor last)
    # 8. ssm-arns — verified automatically by step_agents via deploy_runtime()
    step_prompts
    step_secrets
    step_iam
    step_lambdas
    step_gateway
    step_platform
    echo ""
    echo "  ════════════════════════════════════════════════════"
    echo "  ⚠️  MANUAL STEP REQUIRED before deploying agents:"
    echo ""
    echo "  1. Edit .env.prod:"
    echo "     POSTGRES_URL=postgresql://postgres:<pwd>@<rds-endpoint>/clinical_agent"
    echo "  2. Push postgres secret:"
    echo "     source .env.prod && ./scripts/deploy.sh secrets"
    echo "  3. Deploy all 6 agents:"
    echo "     ./scripts/deploy.sh agents"
    echo "  ════════════════════════════════════════════════════"
    ;;

  destroy)
    echo "⚠️  Destroying Terraform resources..."
    cd "${ROOT}/infra"
    terraform destroy -auto-approve \
      -var="platform_image_uri=placeholder" \
      -var="ui_image_uri=placeholder" \
      -var="aws_region=${REGION}"
    ;;

  *)
    echo "Usage: $0 {prompts|secrets|iam|lambdas|gateway|platform|agents|ssm-arns|redeploy|all|plan|destroy}"
    echo ""
    echo "  prompts   — Create 6 Bedrock prompts (one per agent) → SSM"
    echo "  agents    — Build + deploy 6 AgentCore Runtimes (sub-agents first, supervisor last)"
    echo "  ssm-arns  — Verify sub-agent ARNs in SSM (Supervisor reads these)"
    echo "  all       — Steps 0-7 (pause after platform to fill POSTGRES_URL)"
    exit 1
    ;;
esac