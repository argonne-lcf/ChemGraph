"""The data passed from a peer's inbox to its application's turn handler."""

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class PeerMessage(BaseModel):
    """A request or reply. Sender names are labels, not authentication."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    sender: str = Field(min_length=1)
    content: str = Field(min_length=1)
    reply_to: str | None = None
