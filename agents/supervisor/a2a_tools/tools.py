"""
a2a_tools/tools.py
===================
StructuredTool factory — builds LangChain tools from the agent registry.

WHAT IS A STRUCTUREDTOOL?
--------------------------
A StructuredTool is LangChain's way of exposing a Python function to an LLM
as a "tool" it can call. It has three parts:
  - name        : what gpt-5.5 calls it in its tool_calls JSON
  - description : what gpt-5.5 reads to decide WHEN to use it
  - args_schema : Pydantic model that validates the LLM's arguments

When gpt-5.5 decides to call knowledge_agent("Which trials are in Phase 3?"),
LangGraph's ToolNode looks up the StructuredTool named "knowledge_agent" and
calls its coroutine with the validated arguments.

WHY REBUILT PER REQUEST?
------------------------
build_a2a_tools() is called once per request in build_supervisor_agent().
It creates new tool closures every time. This is necessary because:

  session_id:  Each request has a different session_id. The tool closure
               uses it to namespace the sub-agent's Postgres checkpoint:
               "{session_id}__knowledge" — isolating each session's history.

  token_queue: Each request creates a fresh asyncio.Queue. The tool closure
               captures a reference to it. Chart events from chart_agent go
               into this queue and appear on the right user's SSE stream.
               If we reused closures, chart events from one user would appear
               on another user's screen.

THE make_tool_func WRAPPER — WHY IT EXISTS
------------------------------------------
This is a classic Python closure gotcha. Consider:

  # WRONG — all tools share the same agent_name ("chart" — last iteration)
  for agent_def in registry:
      name = agent_def["name"]
      async def tool_func(query):
          return await invoke_sub_agent(name, ...)  # 'name' captured by reference!

  # CORRECT — each tool has its own agent_name captured by value
  for agent_def in registry:
      name = agent_def["name"]
      def make_tool_func(agent_name):  # forces value capture via function argument
          async def tool_func(query):
              return await invoke_sub_agent(agent_name, ...)
          return tool_func

Python closures capture VARIABLES (references), not VALUES. In a loop,
all closures see the final value of the loop variable. make_tool_func()
forces value capture by passing the variable as a function argument.

SUB-AGENT SESSION NAMESPACING
-------------------------------
Each sub-agent has its own LangGraph graph with its own Postgres checkpoint.
If we passed the same session_id to all agents:
  - knowledge_agent and supervisor would share message history → confusion
  - Two concurrent users might interfere with each other's agent checkpoints

Namespacing: "{supervisor_session_id}__{agent_name}"
  e.g. "c73e7373-abc1-...__knowledge"
Each sub-agent sees only its own isolated conversation history.
"""
import asyncio
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from agents.supervisor.a2a_tools.hitl     import build_ask_user_tool
from agents.supervisor.a2a_tools.invoke   import invoke_sub_agent
from agents.supervisor.a2a_tools.registry import get_agent_registry

log = logging.getLogger(__name__)


class AgentInput(BaseModel):
    """
    Pydantic schema shared by all A2A sub-agent tools.

    Every sub-agent tool takes exactly one argument: a query string.
    gpt-5.5 generates this from the user's question, and Pydantic ensures
    it's always a string (not accidentally a dict, list, or int).
    """
    query: str = Field(
        description="The specific query or task to send to this agent. Be precise and include all relevant context."
    )


def build_a2a_tools(
    session_id:  str,
    domain:      str,
    token_queue: asyncio.Queue,
) -> list[Any]:
    """
    Build StructuredTool instances for all registered sub-agents.

    One tool per agent in the registry + the HITL clarification tool.

    Parameters
    ----------
    session_id  : str            — supervisor's session ID (namespaced per sub-agent)
    domain      : str            — "pharma", passed to sub-agents for context routing
    token_queue : asyncio.Queue  — chart/interrupt events go here and reach the UI

    Returns
    -------
    list[StructuredTool] — passed to create_agent() in agent.py as the tool list
    """
    registry = get_agent_registry()
    tools    = []

    for agent_def in registry:
        name        = agent_def["name"]
        description = agent_def["description"]

        def make_tool_func(agent_name: str):
            """
            Create tool coroutine with agent_name bound by value.

            The function argument 'agent_name' forces value capture.
            Without this wrapper, all tools would use the same 'name'
            from the last loop iteration (Python closure gotcha).
            """
            async def tool_func(query: str) -> str:
                log.info(f"[Supervisor] Calling {agent_name}  query={query[:80]}...")

                # Namespace: isolates this sub-agent's Postgres checkpoint
                # from the supervisor's checkpoint and from other sub-agents.
                agent_session_id = f"{session_id}__{agent_name}"

                answer = await invoke_sub_agent(
                    agent_name  = agent_name,
                    payload     = {
                        "message":    query,
                        "session_id": agent_session_id,
                        "domain":     domain,
                    },
                    token_queue = token_queue,  # chart events go here → UI
                )

                log.info(f"[Supervisor] {agent_name} returned  len={len(answer)}")
                # Return the answer — LangGraph wraps it in a ToolMessage
                # that gpt-5.5 reads in the next ReAct turn.
                return answer

            # Set __name__ for cleaner LangSmith traces and log messages
            tool_func.__name__ = f"{agent_name}_agent"
            return tool_func

        tool = StructuredTool.from_function(
            coroutine   = make_tool_func(name),  # value capture via argument
            name        = f"{name}_agent",        # what gpt-5.5 calls in tool_calls
            description = description,            # what gpt-5.5 reads to decide when to call
            args_schema = AgentInput,             # Pydantic validation of LLM arguments
        )
        tools.append(tool)
        log.info(f"[A2A] Registered tool: {name}_agent")

    # Always add the HITL clarification tool last.
    # Ordering does not affect which tool gpt-5.5 chooses — that's based on
    # descriptions. But putting it last makes logs easier to read.
    tools.append(build_ask_user_tool())
    log.info(f"[A2A] Total tools built: {len(tools)} → {[t.name for t in tools]}")
    return tools