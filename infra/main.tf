# infra/main.tf — vs-agentcore-multiagent
# =========================================
# COPY of vs-agentcore-platform-aws/infra/main.tf with these changes:
#
#   1. backend.key    : "vs-agentcore-multiagent/terraform.tfstate"
#   2. locals.prefix  : "vs-agentcore-ma"   (shortened to avoid name length limits)
#   3. All SSM/secret paths use /vs-agentcore-multiagent/prod/
#   4. Platform ECS container gets SSM_PREFIX=/vs-agentcore-multiagent/prod
#   5. UI AGENT_API_KEY SSM path updated to /vs-agentcore-multiagent/prod/platform_api_key
#
# WHAT IS NOT IN THIS FILE (managed by deploy.sh via AWS CLI):
#   - AgentCore Runtimes (all 6) — create-agent-runtime per agent
#   - MCP Gateway + targets       — create-gateway + create-gateway-target
#   - Bedrock Prompt Management   — create-prompt per agent (6 prompts)
#
# WHY AgentCore Runtimes are NOT in Terraform:
#   The Terraform AWS provider does not yet have a resource for
#   aws_bedrock_agentcore_agent_runtime. We use the AWS CLI directly
#   in deploy.sh step_agents() — same approach as single agent.
#
# ARCHITECTURE (same as single agent, just prefix changes):
#   VPC → ECS Cluster → Platform (ECS Fargate) → ALB → Internet
#                     → UI (ECS Fargate) → ALB → Internet
#   RDS PostgreSQL (LangGraph AsyncPostgresSaver — all 6 agents share it)
#   DynamoDB (TracerMiddleware — Supervisor writes traces)

terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  backend "s3" {
    bucket = "vs-agentcore-tfstate"
    key    = "vs-agentcore-multiagent/terraform.tfstate"  # ← updated
    region = "us-east-1"
  }
}

provider "aws" { region = var.aws_region }

data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  prefix     = "vs-agentcore-ma"                          # ← updated (shortened)
  ssm_prefix = "/vs-agentcore-multiagent/prod"            # ← updated
}

# ── VPC ───────────────────────────────────────────────────────────────────
# Identical to single agent

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "${local.prefix}-vpc" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags = { Name = "${local.prefix}-public-${count.index}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags = { Name = "${local.prefix}-private-${count.index}" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ── ECS Cluster ───────────────────────────────────────────────────────────

resource "aws_ecs_cluster" "main" {
  name = "${local.prefix}-cluster"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# ── IAM ───────────────────────────────────────────────────────────────────
# Copy from single agent main.tf — update resource names only

resource "aws_iam_role" "ecs_task" {
  name = "${local.prefix}-ecs-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task" {
  name = "${local.prefix}-ecs-task-policy"
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["bedrock-agentcore:InvokeAgentRuntime"]
        Resource = "arn:aws:bedrock-agentcore:${var.aws_region}:${local.account_id}:runtime/*"
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter", "ssm:GetParameters", "ssm:PutParameter",
          "secretsmanager:GetSecretValue",
          "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
          "ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage",
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ── Security Groups ───────────────────────────────────────────────────────

resource "aws_security_group" "ecs" {
  name   = "${local.prefix}-ecs-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ── ALB ───────────────────────────────────────────────────────────────────

resource "aws_lb" "main" {
  name               = "${local.prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.ecs.id]
  subnets            = aws_subnet.public[*].id
}

resource "aws_lb_target_group" "platform" {
  name        = "${local.prefix}-platform"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"
  health_check {
    path                = "/health"
    timeout             = 10
    interval            = 30
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_target_group" "ui" {
  name        = "${local.prefix}-ui"
  port        = 8501
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"
  health_check {
    path                = "/health"
    timeout             = 10
    interval            = 30
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ui.arn
  }
}

resource "aws_lb_listener_rule" "platform_api" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 10
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.platform.arn
  }
  condition {
    path_pattern { values = ["/api/*"] }
  }
}

resource "aws_lb_listener_rule" "platform_observability" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 11
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.platform.arn
  }
  condition {
    path_pattern { values = ["/observability*"] }
  }
}

# ── RDS Postgres ──────────────────────────────────────────────────────────
# Shared by all 6 agents — each uses its own thread_id namespace

resource "aws_security_group" "rds" {
  name   = "${local.prefix}-rds-sg"
  vpc_id = aws_vpc.main.id
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_subnet_group" "main" {
  name       = "${local.prefix}-db-subnet"
  subnet_ids = aws_subnet.public[*].id
}

resource "aws_db_instance" "postgres" {
  identifier             = "${local.prefix}-postgres"
  engine                 = "postgres"
  engine_version         = "15"
  instance_class         = "db.t3.micro"
  allocated_storage      = 20
  storage_type           = "gp2"
  db_name                = "clinical_agent"
  username               = "postgres"
  password               = var.postgres_password
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  skip_final_snapshot    = true
  publicly_accessible    = true
  tags = { Name = "${local.prefix}-postgres" }
}

# ── DynamoDB traces ───────────────────────────────────────────────────────
# Supervisor's TracerMiddleware writes here (same as single agent)

resource "aws_dynamodb_table" "traces" {
  name         = "${local.prefix}-traces"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "run_id"
  attribute {
    name = "run_id"
    type = "S"
  }
  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }
  tags = { Name = "${local.prefix}-traces" }
}

# ── ECS: Platform ─────────────────────────────────────────────────────────
# CHANGE: SSM_PREFIX updated to /vs-agentcore-multiagent/prod

resource "aws_cloudwatch_log_group" "platform" {
  name              = "/ecs/${local.prefix}/platform"
  retention_in_days = 7
}

resource "aws_ecs_task_definition" "platform" {
  family                   = "${local.prefix}-platform"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_task.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "platform"
    image = var.platform_image_uri
    portMappings = [{ containerPort = 8000 }]
    environment = [
      { name = "AWS_REGION", value = var.aws_region },
      { name = "SSM_PREFIX", value = local.ssm_prefix }  # ← updated
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${local.prefix}/platform"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "platform"
      }
    }
  }])
}

resource "aws_ecs_service" "platform" {
  name            = "${local.prefix}-platform"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.platform.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.platform.arn
    container_name   = "platform"
    container_port   = 8000
  }
}

# ── ECS: UI ───────────────────────────────────────────────────────────────
# CHANGE: AGENT_API_KEY SSM path updated to /vs-agentcore-multiagent/prod

resource "aws_cloudwatch_log_group" "ui" {
  name              = "/ecs/${local.prefix}/ui"
  retention_in_days = 7
}

resource "aws_ecs_task_definition" "ui" {
  family                   = "${local.prefix}-ui"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_task.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "ui"
    image = var.ui_image_uri
    portMappings = [{ containerPort = 8501 }]
    environment = [
      { name = "AGENT_API_URL", value = "http://${aws_lb.main.dns_name}" },
      { name = "AGENT_DOMAIN",  value = "pharma" }
    ]
    secrets = [{
      name      = "AGENT_API_KEY"
      valueFrom = "arn:aws:ssm:${var.aws_region}:${local.account_id}:parameter/vs-agentcore-multiagent/prod/platform_api_key"  # ← updated
    }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/${local.prefix}/ui"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ui"
      }
    }
  }])
}

resource "aws_ecs_service" "ui" {
  name            = "${local.prefix}-ui"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.ui.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.ui.arn
    container_name   = "ui"
    container_port   = 8501
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────

output "alb_dns"        { value = aws_lb.main.dns_name }
output "rds_endpoint"   { value = aws_db_instance.postgres.endpoint }
output "dynamodb_table" { value = aws_dynamodb_table.traces.name }
