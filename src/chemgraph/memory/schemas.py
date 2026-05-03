"""
Pydantic schemas for ChemGraph session memory.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class SessionMessage(BaseModel):
    """A single message in a session conversation."""

    role: str = Field(description="Message role: 'human', 'ai', or 'tool'")
    content: str = Field(description="Message content text")
    tool_name: Optional[str] = Field(
        default=None, description="Tool name if role is 'tool'"
    )
    timestamp: datetime = Field(default_factory=datetime.now)


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
