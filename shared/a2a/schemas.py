"""
shared/a2a/schemas.py
=====================
Full A2A (Agent-to-Agent) protocol Pydantic models for the VS AgentCore
multi-agent clinical trial research platform.

All agents import from here — NEVER define A2A types locally in agent code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT IS A2A?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A2A is an open protocol by Google for agent-to-agent communication.
Every agent exposes two HTTP endpoints:

    GET  /.well-known/agent.json   → returns AgentCard (who am I, what can I do)
    POST /a2a                      → receives JSON-RPC 2.0 TaskRequest, returns TaskResult

The Supervisor discovers sub-agents by reading their AgentCards at cold start,
then routes tasks to the right agent based on Skills and intent classification.

Reference specs:
    https://google.github.io/A2A/
    https://docs.aws.amazon.com/bedrock/agentcore/a2a

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AGENT DISCOVERY (cold start)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When the Supervisor Agent starts, it reads 5 sub-agent URLs from SSM:

    /vs-agentcore-multiagent/prod/agents/research/url
    /vs-agentcore-multiagent/prod/agents/knowledge/url
    /vs-agentcore-multiagent/prod/agents/hitl/url
    /vs-agentcore-multiagent/prod/agents/safety/url
    /vs-agentcore-multiagent/prod/agents/chart/url

For each URL it calls GET /.well-known/agent.json and gets an AgentCard.
It then builds a skill registry: tag → {agent_url, skill_id}

    "search"    → Research Agent  (skill: search_clinical_trials)
    "synthesis" → Research Agent  (skill: synthesise_evidence)
    "graph"     → Knowledge Agent (skill: query_graph)
    "discovery" → Knowledge Agent (skill: discover_trials)
    "hitl"      → HITL Agent      (skill: generate_clarification)
    "guardrail" → Safety Agent    (skill: evaluate_faithfulness)
    "chart"     → Chart Agent     (skill: generate_chart)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FULL OBJECT HIERARCHY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  AgentCard                          Published by every sub-agent
    ├── name: str                    "Research Agent"
    ├── description: str             "Semantic search over 5,772 trial chunks"
    ├── url: str                     "https://<agentcore-runtime-url>"
    ├── capabilities: AgentCapabilities
    │     ├── streaming: True        SSE token streaming
    │     ├── pushNotifications: True  AgentCore async push
    │     └── stateTransitionHistory: True  LangGraph RDS checkpointing
    └── skills: Skill[]
          ├── id: str                "search_clinical_trials"
          ├── name: str              "Search Clinical Trials"
          ├── description: str       What this skill does
          ├── tags: list[str]        ["search", "rag", "pinecone"]
          └── examples: list[str]   ["What are Phase 3 results for BNT162b2?"]

  ─────────────────────────────────────────────────────
  HAPPY PATH  (specific query, no HITL)
  ─────────────────────────────────────────────────────

  Supervisor builds and sends:

  JSONRPCRequest                     POST /a2a on sub-agent
    ├── jsonrpc: "2.0"
    ├── method: "tasks/send"
    ├── id: str                      request UUID (for correlation)
    └── params: TaskSendParams
          ├── id: str                task UUID
          ├── sessionId: str         user's thread_id / session
          ├── message: Message
          │     ├── role: "user"
          │     └── parts: list
          │           └── TextPart
          │                 └── text: "What are the Phase 3 results for BNT162b2?"
          └── metadata: dict
                ├── domain: "pharma"
                ├── skill_id: "search_clinical_trials"
                └── intent: "specific"

  Sub-agent processes and returns:

  JSONRPCResponse
    ├── jsonrpc: "2.0"
    ├── id: str                      same request UUID
    └── result: TaskResult
          ├── id: str                same task UUID
          ├── sessionId: str
          ├── status: TaskStatus
          │     └── state: "completed"
          └── artifacts: Artifact[]
                └── Artifact[0]
                      └── parts: list
                            ├── TextPart
                            │     └── text: "BNT162b2 showed 95% efficacy [Source: NCT04368728]"
                            └── DataPart
                                  └── data: {"scores": [0.91, 0.87], "chunk_ids": [...]}

  ─────────────────────────────────────────────────────
  HITL PATH  (vague query — needs clarification)
  ─────────────────────────────────────────────────────

  HITL Agent returns input-required instead of completed:

  JSONRPCResponse
    └── result: TaskResult
          ├── status: TaskStatus
          │     └── state: "input-required"   ← triggers interrupt flow
          └── artifacts: Artifact[]
                └── Artifact[0]
                      └── parts: list
                            └── DataPart
                                  └── data: HITLInterruptData
                                        ├── type: "hitl_interrupt"
                                        ├── question: "Which cancer trial?"
                                        ├── options: ["NCI-MATCH", "BASIC3", ...]
                                        └── session_id: "abc-123"

  Supervisor receives input-required → emits SSE interrupt to UI:
      {"type": "interrupt", "question": "...", "options": [...]}
  UI shows HITL card → user selects → UI calls POST /resume
  Supervisor resumes with user_answer, sends new TaskRequest to Research Agent

  ─────────────────────────────────────────────────────
  CHART PATH  (numerical answer → Chart Agent)
  ─────────────────────────────────────────────────────

  Chart Agent returns completed with chart data:

  JSONRPCResponse
    └── result: TaskResult
          ├── status: TaskStatus
          │     └── state: "completed"
          └── artifacts: Artifact[]
                └── Artifact[0]
                      └── parts: list
                            ├── TextPart
                            │     └── text: "Here is the efficacy comparison:"
                            └── DataPart
                                  └── data: ChartOutput
                                        ├── type: "chart"
                                        ├── chart_type: "bar"
                                        ├── title: "COVID-19 Vaccine Efficacy"
                                        ├── labels: ["BNT162b2", "mRNA-1273", "Ad26.COV2.S"]
                                        └── datasets: [ChartDataset(data=[95, 94.1, 66.9])]

  UI detects DataPart with type="chart" → renders Chart.js canvas inline

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK STATUS STATE MACHINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    submitted ──► working ──► completed       (happy path)
                          └──► input-required  (HITL interrupt)
                          └──► failed          (tool error, timeout, guardrail block)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODELS DEFINED IN THIS FILE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Agent Card          AgentCard, Skill, AgentCapabilities, AgentAuthentication
  Message Parts       TextPart, DataPart, FilePart, Part (Union)
  Message             Message
  Task                Task, TaskSendParams
  Task Lifecycle      TaskStatus, Artifact, TaskResult
  Wire Format         JSONRPCRequest, JSONRPCResponse, JSONRPCError
  HITL                HITLInterruptData
  Chart               ChartOutput, ChartDataset
  Supervisor          AgentRegistryEntry, IntentResult
"""
from __future__ import annotations

import uuid
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# AGENT CARD  —  published at GET /.well-known/agent.json
# ══════════════════════════════════════════════════════════════════════════════

class Skill(BaseModel):
    """
    A discrete capability the agent can perform.
    The Supervisor indexes skills by tag to decide which agent to call.

    Example:
        Skill(
            id="search_clinical_trials",
            name="Search Clinical Trials",
            description="Semantic search over 5,772 trial chunks via Pinecone.",
            tags=["search", "rag", "pinecone"],
            examples=["What are the Phase 3 results for BNT162b2?"]
        )
    """
    id: str
    name: str
    description: str
    tags: list[str] = []
    examples: list[str] = []
    inputModes: list[str] = ["text"]
    outputModes: list[str] = ["text"]


class AgentCapabilities(BaseModel):
    streaming: bool = True               # all agents stream SSE tokens
    pushNotifications: bool = True       # AgentCore push via SNS/SQS
    stateTransitionHistory: bool = True  # LangGraph checkpointer (RDS Postgres)


class AgentAuthentication(BaseModel):
    schemes: list[str] = ["Bearer"]


class AgentCard(BaseModel):
    """
    Published by every sub-agent at GET /.well-known/agent.json.
    Supervisor reads this at cold start to discover agents and their skills.

    Example:
        AgentCard(
            name="Research Agent",
            description="Deep clinical trial document search and evidence synthesis.",
            url="https://<agentcore-runtime-url>",
            skills=[Skill(...), Skill(...)]
        )
    """
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    authentication: AgentAuthentication = Field(default_factory=AgentAuthentication)
    defaultInputModes: list[str] = ["text"]
    defaultOutputModes: list[str] = ["text"]
    skills: list[Skill] = []


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGE PARTS  —  content inside a Message
# ══════════════════════════════════════════════════════════════════════════════

class TextPart(BaseModel):
    """Plain text content — most common part type."""
    type: Literal["text"] = "text"
    text: str


class DataPart(BaseModel):
    """
    Structured JSON data — used for:
      - HITL interrupt payload  (HITLInterruptData)
      - Chart output            (ChartOutput)
      - Agent metadata          (routing hints, scores)
    """
    type: Literal["data"] = "data"
    data: dict[str, Any]


class FilePart(BaseModel):
    """File attachment — name, mimeType, and bytes (base64) or uri."""
    type: Literal["file"] = "file"
    file: dict[str, Any]   # {name: str, mimeType: str, bytes?: str, uri?: str}


Part = Union[TextPart, DataPart, FilePart]


# ══════════════════════════════════════════════════════════════════════════════
# MESSAGE  —  the content unit inside a Task
# ══════════════════════════════════════════════════════════════════════════════

class Message(BaseModel):
    """
    A message exchanged between Supervisor and sub-agent.

    role:  "user"  — message from Supervisor to sub-agent
           "agent" — message from sub-agent back to Supervisor
    parts: one or more Part objects (text, data, file)
    """
    role: Literal["user", "agent"]
    parts: list[dict[str, Any]]   # serialised Part dicts

    def get_text(self) -> str:
        """Extract all text content from parts."""
        return " ".join(
            p["text"] for p in self.parts
            if p.get("type") == "text" and p.get("text")
        )

    def get_data(self) -> Optional[dict]:
        """Extract first data part payload."""
        for p in self.parts:
            if p.get("type") == "data":
                return p.get("data")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TASK  —  a unit of work sent from Supervisor to a sub-agent
# ══════════════════════════════════════════════════════════════════════════════

class Task(BaseModel):
    """
    A unit of work routed by the Supervisor to a specific sub-agent.

    id:        unique task identifier (UUID)
    sessionId: ties this task to a user conversation thread
    message:   the actual request content
    metadata:  routing hints, domain, skill_id, etc.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sessionId: str
    message: Message
    metadata: dict[str, Any] = {}


class TaskSendParams(BaseModel):
    """Params block inside a tasks/send JSON-RPC request."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sessionId: str
    message: Message
    metadata: dict[str, Any] = {}


# ══════════════════════════════════════════════════════════════════════════════
# TASK STATUS  —  lifecycle state of a task
# ══════════════════════════════════════════════════════════════════════════════

class TaskStatus(BaseModel):
    """
    State machine for a task:

      submitted → working → completed
                          → input-required  (HITL interrupt)
                          → failed

    input-required means the HITL Agent needs human input before it can continue.
    The Supervisor propagates this back to the UI as an SSE interrupt event.
    """
    state: Literal["submitted", "working", "completed", "input-required", "failed"]
    message: Optional[Message] = None
    timestamp: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
# ARTIFACT  —  output produced by a sub-agent
# ══════════════════════════════════════════════════════════════════════════════

class Artifact(BaseModel):
    """
    Output from a sub-agent.

    A task result can have multiple artifacts — e.g. Research Agent returns:
      Artifact 0: answer text with citations
      Artifact 1: source metadata (scores, chunk IDs)
    """
    parts: list[dict[str, Any]]   # serialised Part dicts
    name: Optional[str] = None
    description: Optional[str] = None
    index: int = 0
    append: bool = False
    lastChunk: bool = True

    def get_text(self) -> str:
        """Extract text from all text parts."""
        return " ".join(
            p["text"] for p in self.parts
            if p.get("type") == "text" and p.get("text")
        )

    def get_data(self) -> Optional[dict]:
        """Extract first data part payload."""
        for p in self.parts:
            if p.get("type") == "data":
                return p.get("data")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# TASK RESULT  —  sub-agent's response to a TaskRequest
# ══════════════════════════════════════════════════════════════════════════════

class TaskResult(BaseModel):
    """
    Returned by a sub-agent after processing a Task.

    If status.state == "completed"      → artifacts contain the answer
    If status.state == "input-required" → artifacts contain HITLInterruptData
    If status.state == "failed"         → metadata contains error details
    """
    id: str
    sessionId: Optional[str] = None
    status: TaskStatus
    artifacts: list[Artifact] = []
    metadata: dict[str, Any] = {}

    def is_complete(self) -> bool:
        return self.status.state == "completed"

    def is_interrupt(self) -> bool:
        return self.status.state == "input-required"

    def is_failed(self) -> bool:
        return self.status.state == "failed"

    def get_answer(self) -> str:
        """Extract combined text from all artifacts."""
        return "\n\n".join(
            a.get_text() for a in self.artifacts if a.get_text()
        )

    def get_hitl_data(self) -> Optional["HITLInterruptData"]:
        """Extract HITL interrupt payload if state is input-required."""
        for artifact in self.artifacts:
            data = artifact.get_data()
            if data and data.get("type") == "hitl_interrupt":
                return HITLInterruptData(**data)
        return None

    def get_chart_data(self) -> Optional["ChartOutput"]:
        """Extract chart output if Chart Agent returned a chart."""
        for artifact in self.artifacts:
            data = artifact.get_data()
            if data and data.get("type") == "chart":
                return ChartOutput(**data)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# JSON-RPC 2.0 ENVELOPE  —  wire format for A2A HTTP calls
# ══════════════════════════════════════════════════════════════════════════════

class JSONRPCRequest(BaseModel):
    """
    Wire format for POST /a2a requests.

    method: "tasks/send"   — send a new task
            "tasks/get"    — poll task status
            "tasks/cancel" — cancel a running task
    """
    jsonrpc: str = "2.0"
    method: Literal["tasks/send", "tasks/get", "tasks/cancel"] = "tasks/send"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    params: dict[str, Any]


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCResponse(BaseModel):
    """Wire format for responses from POST /a2a."""
    jsonrpc: str = "2.0"
    id: str
    result: Optional[dict[str, Any]] = None
    error: Optional[JSONRPCError] = None


# ══════════════════════════════════════════════════════════════════════════════
# HITL INTERRUPT DATA  —  payload when state == "input-required"
# ══════════════════════════════════════════════════════════════════════════════

class HITLInterruptData(BaseModel):
    """
    Returned inside a DataPart when the HITL Agent needs user input.

    The Supervisor extracts this and emits an SSE interrupt event to the UI:
        {"type": "interrupt", "question": "...", "options": [...]}

    The UI renders the HITL card. When the user picks an option, the UI calls
    POST /resume with {user_answer: "selected option"}.

    Example:
        HITLInterruptData(
            type="hitl_interrupt",
            question="Which cancer trial are you interested in?",
            options=[
                "NCI-MATCH study",
                "Baylor College of Medicine BASIC3 study",
                "SAFIR01/UNICANCER study",
                "NCI-MPACT trial",
                "Dana-Farber Cancer Institute iCAT study",
            ],
            session_id="abc-123",
            allow_freetext=True
        )
    """
    type: Literal["hitl_interrupt"] = "hitl_interrupt"
    question: str
    options: list[str]
    session_id: str
    allow_freetext: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# CHART OUTPUT  —  returned by Chart Agent
# ══════════════════════════════════════════════════════════════════════════════

class ChartDataset(BaseModel):
    """A single dataset inside a Chart.js chart."""
    label: str
    data: list[float]
    backgroundColor: list[str] = []
    borderColor: list[str] = []
    borderWidth: int = 1
    fill: bool = False


class ChartOutput(BaseModel):
    """
    Returned by Chart Agent inside a DataPart.

    The UI detects this in the SSE stream and renders a Chart.js canvas
    inline in the chat bubble.

    chart_type options: bar | line | pie | doughnut | scatter | radar

    Example:
        ChartOutput(
            type="chart",
            chart_type="bar",
            title="COVID-19 Vaccine Trial Efficacy Comparison",
            labels=["BNT162b2 (Pfizer)", "mRNA-1273 (Moderna)", "Ad26.COV2.S (J&J)"],
            datasets=[
                ChartDataset(
                    label="Efficacy %",
                    data=[95.0, 94.1, 66.9],
                    backgroundColor=["#00c2ff", "#00e5a0", "#ffb340"]
                )
            ]
        )
    """
    type: Literal["chart"] = "chart"
    chart_type: Literal["bar", "line", "pie", "doughnut", "scatter", "radar"] = "bar"
    title: str
    labels: list[str]
    datasets: list[ChartDataset]
    options: dict[str, Any] = {}
    source_text: Optional[str] = None   # the original text the chart was generated from


# ══════════════════════════════════════════════════════════════════════════════
# SUPERVISOR AGENT REGISTRY  —  built at cold start from Agent Cards
# ══════════════════════════════════════════════════════════════════════════════

class AgentRegistryEntry(BaseModel):
    """
    One entry in the Supervisor's skill → agent mapping.
    Built by reading Agent Cards from all sub-agents at cold start.
    """
    agent_name: str
    agent_url: str
    skill_id: str
    skill_description: str
    tags: list[str]


# ══════════════════════════════════════════════════════════════════════════════
# INTENT  —  output from Supervisor intent classifier
# ══════════════════════════════════════════════════════════════════════════════

class IntentResult(BaseModel):
    """
    Output from agents/supervisor/intent.py classify() function.

    intent:       what type of query this is
    agents:       which sub-agents to call (in order or parallel)
    run_parallel: if True, call research + knowledge simultaneously
    needs_chart:  if True, route answer through Chart Agent after research

    Intent types:
      specific  — user asked about a known trial/drug → Research + Knowledge parallel
      vague     — query is broad → Research (candidates) → HITL → resume → Research
      chart     — query implies comparison/numbers → Research → Chart Agent
      list      — "list all trials" → Knowledge Agent only
      safety    — always append Safety Agent after any answer
    """
    intent: Literal["specific", "vague", "chart", "list", "safety_only"]
    agents: list[str]           # agent names in call order
    run_parallel: bool = False  # Research + Knowledge in parallel
    needs_chart: bool = False   # route through Chart Agent
    reasoning: str = ""         # why this intent was chosen (for debugging)