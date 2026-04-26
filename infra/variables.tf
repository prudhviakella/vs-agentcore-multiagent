# infra/variables.tf — vs-agentcore-multiagent
# ==============================================
# Copy of single agent variables.tf with image URIs for all 6 agents.

variable "aws_region" {
  default = "us-east-1"
}

variable "postgres_password" {
  description = "RDS Postgres password — set in .env.prod"
  sensitive   = true
}

variable "ssm_prefix" {
  default = "/vs-agentcore-multiagent/prod"
}

# ── ECS image URIs (set by deploy.sh step_agents) ─────────────────────────
variable "platform_image_uri" { description = "Platform API ECR image URI" }
variable "ui_image_uri"       { description = "UI ECR image URI" }

# ── AgentCore runtime image URIs ───────────────────────────────────────────
# Not used directly by Terraform (AgentCore runtimes created via AWS CLI)
# Kept here for reference and for deploy.sh to read via terraform output
variable "supervisor_image_uri" {
  description = "Supervisor Agent ECR image URI"
  default     = ""
}
variable "research_image_uri" {
  description = "Research Agent ECR image URI"
  default     = ""
}
variable "knowledge_image_uri" {
  description = "Knowledge Agent ECR image URI"
  default     = ""
}
variable "hitl_image_uri" {
  description = "HITL Agent ECR image URI"
  default     = ""
}
variable "safety_image_uri" {
  description = "Safety Agent ECR image URI"
  default     = ""
}
variable "chart_image_uri" {
  description = "Chart Agent ECR image URI"
  default     = ""
}