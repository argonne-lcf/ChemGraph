"""Shared event log for Academy-ChemGraph campaign runs.

The dynamic campaign layer treats agent messages and ChemGraph job updates as
one append-only event stream.  The dashboard and HPC run scripts can consume
this file without knowing which science use case created the event.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


EventKind = Literal[
    "campaign_started",
    "campaign_planned",
    "campaign_finished",
    "agent_started",
    "agent_stopped",
    "agent_decision",
    "agent_error",
    "message_sent",
    "message_received",
    "message_delivered",
    "message_delivery_failed",
    "belief_updated",
    "tool_call_started",
    "tool_call_finished",
    "tool_call_failed",
    "chemgraph_batch_submitted",
    "chemgraph_job_status",
    "chemgraph_job_result",
    "chemgraph_transfer_submitted",
    "chemgraph_transfer_done",
    "round_started",
    "round_finished",
    "self_wake_scheduled",
    "idle_timeout",
    "max_decisions_reached",
    "daemon_started",
    "daemon_stopped",
    "bootstrap_message_dispatched",
    "llm_tool_calls",
    "turn_finished_without_external_action",
    "chemgraph_reasoning_turn_started",
    "chemgraph_reasoning_turn_finished",
    "run_started",
    "run_finished",
    "workflow_started",
    "workflow_finished",
    "workflow_node_started",
    "workflow_node_finished",
    "llm_call_started",
    "llm_call_finished",
    "llm_call_failed",
    "llm_decision",
    "workflow_output",
]

__all__ = [
    'CampaignEvent',
    'EventKind',
    'EventLog',
    'read_events',
]


class CampaignEvent(BaseModel):
    """One durable event emitted by a campaign runtime."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: f"evt-{uuid.uuid4()}")
    timestamp: float = Field(default_factory=time.time)
    event: EventKind
    run_id: str | None = None
    agent_id: str | None = None
    role: str | None = None
    correlation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class EventLog:
    """Append/read helper for campaign JSONL event logs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, event: CampaignEvent) -> CampaignEvent:
        """Append *event* and return it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json())
            handle.write("\n")
        return event

    def emit(
        self,
        event: EventKind,
        *,
        run_id: str | None = None,
        agent_id: str | None = None,
        role: str | None = None,
        correlation_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> CampaignEvent:
        """Build and append a :class:`CampaignEvent`."""
        return self.append(
            CampaignEvent(
                event=event,
                run_id=run_id,
                agent_id=agent_id,
                role=role,
                correlation_id=correlation_id,
                payload=payload or {},
            )
        )

    def read(self) -> list[CampaignEvent]:
        """Read all valid JSONL events from the log."""
        return read_events(self.path)


def read_events(path: str | Path) -> list[CampaignEvent]:
    """Read valid campaign events from *path*.

    Partially written or malformed lines are skipped so live dashboards can
    poll while another process is appending.
    """
    event_path = Path(path)
    if not event_path.exists():
        return []
    events: list[CampaignEvent] = []
    with event_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                events.append(CampaignEvent.model_validate(payload))
            except (json.JSONDecodeError, ValueError):
                continue
    return events
