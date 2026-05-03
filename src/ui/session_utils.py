"""Session persistence utilities for the ChemGraph Streamlit UI.

Bridges the gap between the UI's in-memory ``conversation_history``
format and the :class:`~chemgraph.memory.store.SessionStore` persistence
layer.  Every function is Streamlit-free so it can be unit-tested without
a running Streamlit runtime.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from chemgraph.memory.schemas import Session, SessionMessage

from ui.message_utils import normalize_message_content


# ---------------------------------------------------------------------------
# Session ID generation
# ---------------------------------------------------------------------------


def generate_session_id() -> str:
    """Return a short unique session identifier (first 8 chars of a UUID4).

    Mirrors the convention used by :class:`chemgraph.agent.llm_agent.ChemGraph`.
    """
    return str(uuid.uuid4())[:8]


# ---------------------------------------------------------------------------
# Conversation history  <-->  SessionMessage conversion
# ---------------------------------------------------------------------------


def messages_from_result(result: Any) -> list[SessionMessage]:
    """Extract :class:`SessionMessage` objects from a single agent run result.

    *result* is the value stored in ``conversation_history[i]["result"]``,
    which may be a list of LangChain messages, a dict with a ``"messages"``
    key, or a plain object.
    """
    raw_messages: list[Any] = []
    if isinstance(result, list):
        raw_messages = result
    elif isinstance(result, dict) and "messages" in result:
        raw_messages = list(result["messages"])
    else:
        raw_messages = [result]

    session_messages: list[SessionMessage] = []
    for msg in raw_messages:
        role: Optional[str] = None
        content = ""
        tool_name: Optional[str] = None

        if hasattr(msg, "type") and hasattr(msg, "content"):
            # LangChain message object
            role = _langchain_type_to_role(msg.type)
            content = normalize_message_content(msg.content)
            tool_name = getattr(msg, "name", None)
        elif isinstance(msg, dict):
            role = _langchain_type_to_role(msg.get("type", ""))
            content = normalize_message_content(msg.get("content", ""))
            tool_name = msg.get("name")
        else:
            role = "ai"
            content = normalize_message_content(str(msg))

        if role and content:
            session_messages.append(
                SessionMessage(role=role, content=content, tool_name=tool_name)
            )

    return session_messages


def conversation_entry_to_messages(entry: dict) -> list[SessionMessage]:
    """Convert a single conversation-history entry to :class:`SessionMessage` objects.

    An entry has the shape ``{"query": str, "result": ..., "thread_id": int}``.
    We produce one ``human`` message for the query, followed by messages
    extracted from the result.
    """
    out: list[SessionMessage] = []

    query = entry.get("query", "").strip()
    if query:
        out.append(SessionMessage(role="human", content=query))

    result = entry.get("result")
    if result is not None:
        out.extend(messages_from_result(result))

    return out


def session_to_conversation_history(session: Session) -> list[dict]:
    """Rebuild the UI ``conversation_history`` list from a stored :class:`Session`.

    Groups messages into exchanges by splitting on ``human`` role messages.
    Each exchange becomes ``{"query": str, "result": {"messages": [...]},
    "thread_id": 1}``.
    """
    history: list[dict] = []
    current_query: Optional[str] = None
    current_messages: list[dict] = []

    for msg in session.messages:
        if msg.role == "human":
            # Flush previous exchange
            if current_query is not None:
                history.append(
                    {
                        "query": current_query,
                        "result": {"messages": current_messages},
                        "thread_id": 1,
                    }
                )
            current_query = msg.content
            current_messages = []
        else:
            # Represent as a simple dict with the fields the UI renderers
            # inspect: type, content, name.
            entry: dict[str, Any] = {
                "type": msg.role,
                "content": msg.content,
            }
            if msg.tool_name:
                entry["name"] = msg.tool_name
            current_messages.append(entry)

    # Flush last exchange
    if current_query is not None:
        history.append(
            {
                "query": current_query,
                "result": {"messages": current_messages},
                "thread_id": 1,
            }
        )

    return history


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _langchain_type_to_role(msg_type: str) -> str:
    """Map a LangChain message ``type`` to a SessionMessage ``role``."""
    mapping = {
        "human": "human",
        "ai": "ai",
        "tool": "tool",
        "system": "ai",
        "function": "tool",
    }
    return mapping.get(msg_type, "ai")
