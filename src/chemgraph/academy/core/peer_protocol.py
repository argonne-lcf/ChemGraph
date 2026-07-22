from __future__ import annotations

import uuid
import time
from typing import Any


REQUIRED_MESSAGE_KEYS = {
    'message_id',
    'sender',
    'recipient',
    'content',
}


def validate_message(message: dict[str, Any]) -> None:
    """Validate the generic Academy message envelope."""
    if missing := REQUIRED_MESSAGE_KEYS.difference(message):
        raise ValueError(f'message missing keys: {sorted(missing)}')


def new_correlation_id() -> str:
    return f'corr-{uuid.uuid4()}'


def build_message(
    *,
    sender: str,
    recipient: str,
    content: str,
    round_index: int | None = None,
    kind: str = 'message',
    tldr: str | None = None,
    artifact_refs: list[str] | None = None,
    tool_result_ids: list[str] | None = None,
    reply_requested: bool = False,
    reason: str | None = None,
    confidence: float | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Create the structured message payload sent through Academy handles.

    correlation_id groups messages that belong to the same logical task
    (e.g. one MOF's whole pipeline). Auto-generated when not supplied.
    Reply paths should propagate the incoming message's correlation_id.
    """
    payload: dict[str, Any] = {
        'message_id': f'msg-{uuid.uuid4()}',
        'timestamp': time.time(),
        'sender': sender,
        'recipient': recipient,
        'kind': kind,
        'content': content,
        'reply_requested': reply_requested,
        'artifact_refs': artifact_refs or [],
        'tool_result_ids': tool_result_ids or [],
        'correlation_id': correlation_id or new_correlation_id(),
    }
    if round_index is not None:
        payload['round'] = round_index
    if tldr is not None:
        payload['tldr'] = tldr
    if reason is not None:
        payload['reason'] = reason
    if confidence is not None:
        payload['confidence'] = confidence
    return payload
