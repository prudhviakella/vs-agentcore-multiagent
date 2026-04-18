# infra/main.tf
# =============
# Terraform — 6 AgentCore Runtimes + A2A Gateway
#
# TODO: implement
#   - AgentCore Runtime for each agent (supervisor, research, knowledge, hitl, safety, chart)
#   - AgentCore A2A Gateway
#   - ECR repositories for each agent image
#   - SSM parameters for agent URLs
#   - IAM roles
#   - ECS for Platform API + UI (reuse from vs-agentcore-platform-aws)

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    # TODO: set bucket + key
  }
}

provider "aws" {
  region = var.aws_region
}
