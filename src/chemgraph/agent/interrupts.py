"""Shared normalization helpers for resumable LangGraph interrupts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PendingInterrupt:
    """One pending request for user input."""

    id: str
    payload: Any


def normalize_interrupts(values: Any) -> list[PendingInterrupt]:
    """Convert LangGraph interrupt values into stable application records."""
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        values = [values]

    return [
        PendingInterrupt(
            id=(
                str(getattr(item, "id", "") or "")
                if getattr(item, "id", "") != "placeholder-id"
                else ""
            ),
            payload=getattr(item, "value", item),
        )
        for item in values
    ]


def deduplicate_interrupts(
    interrupts: list[PendingInterrupt],
) -> tuple[PendingInterrupt, ...]:
    """Deduplicate streamed and checkpointed copies of pending interrupts."""
    unique: list[PendingInterrupt] = []
    seen: set[str] = set()
    for item in interrupts:
        if item.id and item.id in seen:
            continue
        if item.id:
            seen.add(item.id)
        unique.append(item)
    return tuple(unique)


def collect_pending_interrupts(
    streamed: list[PendingInterrupt], snapshot: Any = None, *, fallback: bool = False,
) -> tuple[PendingInterrupt, ...]:
    """Prefer checkpoint requests over streamed copies of the same pause.

    Anonymous requests cannot be matched by payload: distinct actions may have
    identical payloads. Use one authoritative collection instead.
    """
    checkpointed = normalize_interrupts(getattr(snapshot, "interrupts", ()))
    if not checkpointed:
        for task in getattr(snapshot, "tasks", ()):
            checkpointed.extend(normalize_interrupts(getattr(task, "interrupts", ())))
    pending = deduplicate_interrupts(checkpointed or streamed)
    if fallback and not pending:
        return (PendingInterrupt(id="", payload={"question": "The workflow needs your input."}),)
    return pending


def is_tool_review(payload: Any) -> bool:
    """Return whether an interrupt contains structured action reviews."""
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("action_requests"), list)
        and isinstance(payload.get("review_configs"), list)
    )


def interrupt_question(payload: Any) -> str:
    """Extract readable question text from an interrupt payload."""
    if isinstance(payload, dict):
        return str(
            payload.get(
                "question",
                payload.get("message", payload.get("instruction", payload)),
            )
        )
    return str(payload)


__all__ = [
    "PendingInterrupt",
    "collect_pending_interrupts",
    "deduplicate_interrupts",
    "interrupt_question",
    "is_tool_review",
    "normalize_interrupts",
]
