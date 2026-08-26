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
            id=str(getattr(item, "id", "") or ""),
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
        key = item.id or repr(item.payload)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return tuple(unique)


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
    "deduplicate_interrupts",
    "interrupt_question",
    "normalize_interrupts",
]
