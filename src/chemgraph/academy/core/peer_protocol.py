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
    reason: str | None = None,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Create the structured message payload sent through Academy handles."""
    payload: dict[str, Any] = {
        'message_id': f'msg-{uuid.uuid4()}',
        'timestamp': time.time(),
        'sender': sender,
        'recipient': recipient,
        'kind': kind,
        'content': content,
        'artifact_refs': artifact_refs or [],
        'tool_result_ids': tool_result_ids or [],
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
