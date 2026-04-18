#!/bin/bash
# scripts/deploy.sh
# =================
# Deploy all agents in order.
# Usage: ./scripts/deploy.sh [all|supervisor|research|knowledge|hitl|safety|chart|infra]
#
# TODO: implement each step
#   step_infra       — terraform apply
#   step_supervisor  — build + push supervisor image, deploy AgentCore Runtime
#   step_research    — build + push research image, deploy AgentCore Runtime
#   step_knowledge   — build + push knowledge image, deploy AgentCore Runtime
#   step_hitl        — build + push hitl image, deploy AgentCore Runtime
#   step_safety      — build + push safety image, deploy AgentCore Runtime
#   step_chart       — build + push chart image, deploy AgentCore Runtime

set -e

COMMAND=${1:-all}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-us-east-1}

echo "================================================"
echo "VS AgentCore Multi-Agent — deploy: $COMMAND"
echo "Account: $ACCOUNT_ID  Region: $REGION"
echo "Root:    $ROOT"
echo "================================================"

case $COMMAND in
  all)        echo "TODO: deploy all agents" ;;
  supervisor) echo "TODO: deploy supervisor" ;;
  research)   echo "TODO: deploy research"   ;;
  knowledge)  echo "TODO: deploy knowledge"  ;;
  hitl)       echo "TODO: deploy hitl"       ;;
  safety)     echo "TODO: deploy safety"     ;;
  chart)      echo "TODO: deploy chart"      ;;
  infra)      echo "TODO: terraform apply"   ;;
  *)          echo "Unknown command: $COMMAND"; exit 1 ;;
esac
