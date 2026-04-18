# infra/variables.tf

variable "aws_region" {
  default = "us-east-1"
}

variable "account_id" {
  description = "AWS Account ID"
}

variable "project" {
  default = "vs-agentcore-multiagent"
}

variable "env" {
  default = "prod"
}

# Agent image tags — set per deployment
variable "supervisor_image_tag" { default = "latest" }
variable "research_image_tag"   { default = "latest" }
variable "knowledge_image_tag"  { default = "latest" }
variable "hitl_image_tag"       { default = "latest" }
variable "safety_image_tag"     { default = "latest" }
variable "chart_image_tag"      { default = "latest" }
