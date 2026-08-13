"""Lossless, strict serialization helpers for durable conversation messages."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, convert_to_messages
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from chemgraph.memory.schemas import SessionMessage


_SERIALIZER = JsonPlusSerializer(
    pickle_fallback=False,
    allowed_msgpack_modules=None,
)


def _readable_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", json.dumps(item, default=str, sort_keys=True))
            if isinstance(item, dict)
            else str(item)
            for item in content
        )
    return str(content)


def serialize_message(message: Any, ordinal: int) -> SessionMessage:
    """Project one LangChain message into readable and lossless storage fields."""
    normalized: BaseMessage = convert_to_messages([message])[0]
    serialization_type, payload = _SERIALIZER.dumps_typed(normalized)
    identity = hashlib.sha256(
        serialization_type.encode("utf-8") + b"\0" + payload
    ).hexdigest()
    content = _readable_content(normalized.content)
    if not content and isinstance(normalized, AIMessage) and normalized.tool_calls:
        names = ", ".join(str(call.get("name", "tool")) for call in normalized.tool_calls)
        content = f"[tool calls: {names}]"
    role = normalized.type
    if role == "assistant":
        role = "ai"
    return SessionMessage(
        role=role,
        content=content,
        tool_name=normalized.name if isinstance(normalized, ToolMessage) else None,
        ordinal=ordinal,
        message_id=identity,
        serialization_type=serialization_type,
        serialized_payload=payload,
    )


def serialize_messages(messages: list[Any]) -> list[SessionMessage]:
    """Serialize an ordered LangChain message transcript."""
    return [serialize_message(message, index) for index, message in enumerate(messages)]
