"""
Pydantic schemas for ChemGraph session memory.
"""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class SessionMessage(BaseModel):
    """A single message in a session conversation."""

    role: str = Field(description="Message role: 'human', 'ai', or 'tool'")
    content: str = Field(description="Message content text")
    tool_name: Optional[str] = Field(
        default=None, description="Tool name if role is 'tool'"
    )
    timestamp: datetime = Field(default_factory=datetime.now)
    ordinal: Optional[int] = Field(
        default=None, description="Stable position in the authoritative transcript"
    )
    message_id: Optional[str] = Field(
        default=None, description="Canonical serialized-message identity"
    )
    serialization_type: Optional[str] = Field(
        default=None, description="Serializer payload type tag"
    )
    serialized_payload: Optional[bytes] = Field(
        default=None, description="Lossless serialized LangChain message"
    )


SessionStatus = Literal["new", "running", "waiting_for_user", "completed", "failed"]
SubagentRunStatus = Literal[
    "running", "waiting_for_user", "completed", "failed"
]


class MainAgentGraphConfig(BaseModel):
    """Non-secret configuration needed to validate a durable graph topology."""

    model_name: str
    recursion_limit: int = 50
    reasoning_effort: Optional[str] = None
    structured_output: bool = False
    generate_report: bool = False
    max_retries: int = 1
    human_supervised: bool = False
    terminal_tool_names: tuple[str, ...] = ()
    enable_deepagent: bool = False
    deepagent_workspace: Optional[str] = None
    subagent_names: tuple[str, ...] = ("chemgraph",)
    tool_signatures: tuple[str, ...] = ()
    package_version: str = ""
    graph_schema_version: int = 1
    topology_fingerprint: str = ""


class MainAgentSessionMetadata(BaseModel):
    """Durable metadata shared by a main-agent graph and session driver."""

    graph_config: MainAgentGraphConfig
    checkpoint_backend: str = "memory"
    checkpoint_db: Optional[str] = None


class SubagentRun(BaseModel):
    """One direct supervisor-to-subagent invocation."""

    run_id: str
    root_session_id: str
    agent_name: str
    delegated_task: str
    checkpoint_namespace: str
    status: SubagentRunStatus
    created_at: datetime
    updated_at: datetime
    error_text: Optional[str] = None
    messages: list[SessionMessage] = Field(default_factory=list)


class Session(BaseModel):
    """Full session record with messages and metadata."""

    session_id: str = Field(description="Unique session identifier (UUID)")
    title: str = Field(
        default="", description="Human-readable session title (auto-generated)"
    )
    model_name: str = Field(description="LLM model used")
    workflow_type: str = Field(description="Workflow type used")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    messages: list[SessionMessage] = Field(
        default_factory=list, description="Conversation messages"
    )
    log_dir: Optional[str] = Field(
        default=None, description="Path to session log directory"
    )
    query_count: int = Field(default=0, description="Number of user queries")
    status: SessionStatus = "completed"
    graph_config: Optional[MainAgentGraphConfig] = None
    topology_fingerprint: Optional[str] = None
    checkpoint_backend: Optional[str] = None
    checkpoint_db: Optional[str] = None
    child_runs: list[SubagentRun] = Field(default_factory=list)

    @property
    def child_run_count(self) -> int:
        """Return the number of direct subagent invocations."""
        return len(self.child_runs)


class SessionSummary(BaseModel):
    """Lightweight session summary for listing sessions."""

    session_id: str
    title: str
    model_name: str
    workflow_type: str
    created_at: datetime
    updated_at: datetime
    query_count: int
    message_count: int
    status: SessionStatus = "completed"
    child_run_count: int = 0
