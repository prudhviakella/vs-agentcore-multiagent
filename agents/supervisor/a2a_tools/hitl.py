"""
a2a_tools/hitl.py
==================
Human-In-The-Loop (HITL) — pauses the agent and asks the user a question.

WHAT IS HITL?
-------------
Sometimes the user's query is too vague to answer correctly. For example:
  User: "Tell me about the trial"
  Agent: Which trial? There are 5,772 in the database.

Instead of guessing, the agent pauses, shows the user real options from
the database, and waits for their selection before continuing.

HOW IT WORKS (end to end)
--------------------------
1. gpt-5.5 decides the query is ambiguous (Case D in the supervisor prompt)
2. gpt-5.5 calls clarify___ask_user_input with question + options
3. ask_user_func raises HITLInterrupt (intentionally — not an error)
4. LangGraph's ToolNode catches it, checkpoints the FULL message state
   to Postgres (keyed by thread_id), then surfaces the interrupt
5. supervisor/streaming.py catches the interrupt and yields:
   {"type": "interrupt", "question": "...", "options": [...]}
6. UI renders the HITL card — user clicks an option
7. UI sends POST /resume with the user's answer
8. supervisor/app.py calls repair_hitl_state() to fix the checkpoint
9. Supervisor resumes from exactly where it paused

WHY AN EXCEPTION FOR HITL?
---------------------------
LangGraph's interrupt mechanism requires raising an exception inside
a tool to signal a pause. This is NOT an error — it's the designed
mechanism. LangGraph catches it specifically and handles it differently
from real errors (saves checkpoint instead of propagating the error).

WHY A PYDANTIC SCHEMA (AskUserInput)?
--------------------------------------
gpt-5.5 generates tool call arguments as JSON. Without a schema, the LLM
might pass wrong types — e.g. options as a string instead of a list.
Pydantic validates and coerces the arguments before ask_user_func runs,
so we always get correctly typed data.
"""
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class HITLInterrupt(Exception):
    """
    Custom exception that signals a HITL pause.

    Raised inside ask_user_func → caught by LangGraph ToolNode →
    LangGraph checkpoints state → supervisor yields interrupt event → UI pauses.

    WHY STORE DATA ON THE EXCEPTION?
    ---------------------------------
    The exception propagates through multiple layers (ToolNode, astream_events,
    _stream_supervisor, handler). Storing question/options on the exception
    means any catch point has the full data needed to emit the interrupt event.

    Attributes
    ----------
    question      : str        — what to ask the user
    options       : list[str]  — pre-built choices from real database results
    allow_freetext: bool       — whether user can type a custom answer
    """
    def __init__(self, question: str, options: list[str], allow_freetext: bool = True):
        self.question       = question
        self.options        = options
        self.allow_freetext = allow_freetext
        super().__init__(f"HITL: {question}")


class AskUserInput(BaseModel):
    """
    Pydantic schema for the clarify___ask_user_input tool.

    gpt-5.5 must pass arguments matching this schema. If it passes the wrong
    type (e.g. options as a string), Pydantic raises a validation error before
    ask_user_func runs — fast fail with a clear error message.

    Field descriptions are part of the tool schema that gpt-5.5 reads when
    deciding how to fill in the arguments. Clear descriptions = better LLM output.
    """
    question:       str       = Field(
        description="The clarifying question to ask the user — specific and concise"
    )
    options:        list[str] = Field(
        description=(
            "2-6 specific options for the user to choose from. "
            "MUST come from real database results — never invented or guessed. "
            "Search with research_agent or knowledge_agent first."
        )
    )
    allow_freetext: bool      = Field(
        default=True,
        description="Whether to allow the user to type a custom answer in addition to the options"
    )


def build_ask_user_tool() -> Any:
    """
    Build and return the HITL StructuredTool.

    This tool is added to the supervisor's tool list alongside the A2A tools.
    gpt-5.5 sees it in the tool schema and can call it when it decides a
    query needs clarification.

    THE DESCRIPTION IS CRITICAL
    ---------------------------
    The tool description is what gpt-5.5 reads to decide WHEN to call this tool
    and HOW to fill in the arguments. Two key instructions in the description:

    1. "First search for real candidates" — prevents the LLM from inventing
       option values. Without this, gpt-5.5 tends to generate plausible-sounding
       but wrong trial names (hallucination).

    2. "This PAUSES execution" — tells gpt-5.5 this is a blocking action, so
       it should only use it when clarification is truly necessary.
    """
    async def ask_user_func(question: str, options: list[str], allow_freetext: bool = True) -> str:
        """
        Raise HITLInterrupt — this is the mechanism that pauses the agent.

        This function intentionally raises instead of returning. LangGraph's
        ToolNode is designed to catch this pattern and checkpoint the state.
        It is NOT a bug — it is the correct LangGraph HITL pattern.
        """
        log.info(f"[HITL] Pausing for user input: question='{question}'  options={options}")
        raise HITLInterrupt(
            question       = question,
            options        = options,
            allow_freetext = allow_freetext,
        )

    return StructuredTool.from_function(
        coroutine   = ask_user_func,
        name        = "clarify___ask_user_input",
        description = (
            "Ask the user a clarifying question when the query is too vague to answer precisely. "
            "IMPORTANT: First search for real candidates using research_agent or knowledge_agent. "
            "Then call this tool with REAL options from the database — never invented names. "
            "This PAUSES execution and checkpoints state until the user responds."
        ),
        args_schema = AskUserInput,
    )