# VS AgentCore — Multi-Agent Platform (A2A)

> Vidya Sankalp · Applied GenAI Engineering

## Architecture

```
User → Platform API → Supervisor Agent (create_agent + full middleware)
                           ├── @tool → Research Agent  (search + summariser MCP tools)
                           ├── @tool → Knowledge Agent (graph + summariser MCP tools)
                           ├── @tool → HITL Agent      (clarify MCP tool + HITL middleware)
                           ├── @tool → Safety Agent    (GPT-4o-mini judge, no tools)
                           └── @tool → Chart Agent     (search + summariser + chart MCP tools)
```

## Key Design Decisions

**AGENT_NAME env var drives everything per container:**
  Terraform sets `AGENT_NAME=research-agent` on the Research Agent runtime.
  `core/aws.get_bedrock_prompt(os.environ["AGENT_NAME"])` reads the right prompt.
  Same pattern for all 6 agents — one code path, infrastructure drives routing.

**core/ is copied unchanged from vs-agentcore-platform-aws:**
  aws.py, cache.py, pinecone_store.py, middleware/ — zero changes.
  Supervisor uses the full middleware stack. Sub-agents use none.

**Supervisor A2A tools (agents/supervisor/a2a_tools.py):**
  Each sub-agent is wrapped as a @tool calling invoke_agent_runtime().
  Sub-agent tokens are re-streamed through Supervisor → Platform API → UI.
  HITLInterrupt exception propagates HITL back up the chain.

**deploy.sh order:**
  secrets → prompts (6 Bedrock prompts) → lambdas → sub-agents → ssm-arns → supervisor → platform

## What to Copy from vs-agentcore-platform-aws

```bash
# core layer (unchanged)
cp -r ../vs-agentcore-platform-aws/agent/core/ core/

# supervisor middleware (unchanged)
cp -r ../vs-agentcore-platform-aws/agent/middleware/ agents/supervisor/middleware/

# platform gateway (unchanged)
cp -r ../vs-agentcore-platform-aws/platform/gateway/ platform/gateway/
cp ../vs-agentcore-platform-aws/platform/static/traces_dashboard.html platform/static/

# ui (unchanged)
cp -r ../vs-agentcore-platform-aws/ui/ ui/

# mcp tools (unchanged except chart_lambda is new)
cp -r ../vs-agentcore-platform-aws/mcp_tools/search_lambda/   mcp_tools/search_lambda/
cp -r ../vs-agentcore-platform-aws/mcp_tools/graph_lambda/    mcp_tools/graph_lambda/
cp -r ../vs-agentcore-platform-aws/mcp_tools/hitl_lambda/     mcp_tools/hitl_lambda/
cp -r ../vs-agentcore-platform-aws/mcp_tools/summariser_lambda/ mcp_tools/summariser_lambda/
```
