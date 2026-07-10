"""Peer message envelopes transported by Academy for academy_sim."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PeerEnvelope(BaseModel):
    """Minimal graph-to-graph message payload."""

    model_config = ConfigDict(extra='forbid')

    message_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    sender: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    content: str = Field(min_length=1)
    created_at: float
    correlation_id: str | None = None
    reply_to: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def build_envelope(
    *,
    run_id: str,
    sender: str,
    recipient: str,
    content: str,
    correlation_id: str | None = None,
    reply_to: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PeerEnvelope:
    """Create a peer envelope for graph-to-graph delivery."""

    return PeerEnvelope(
        message_id=f'cg-peer-{uuid.uuid4()}',
        run_id=run_id,
        sender=sender,
        recipient=recipient,
        content=content,
        created_at=time.time(),
        correlation_id=correlation_id,
        reply_to=reply_to,
        metadata=metadata or {},
    )


def envelope_to_prompt(envelope: PeerEnvelope) -> str:
    """Render a peer envelope as the user input passed to ChemGraph."""

    lines = [
        f"Peer message from {envelope.sender} to {envelope.recipient}.",
        f"Message id: {envelope.message_id}",
    ]
    if envelope.correlation_id:
        lines.append(f"Correlation id: {envelope.correlation_id}")
    if envelope.reply_to:
        lines.append(f"Reply to: {envelope.reply_to}")
    lines.append("")
    lines.append(envelope.content)
    return "\n".join(lines)
