"""Generate dashboard summaries from saved ChemGraph events."""

from __future__ import annotations

import json
import pathlib
from collections import Counter
from typing import Any

from chemgraph.observability.event_log import RunEvent, read_events
from chemgraph.observability.run_files import write_json


def write_run_artifacts(run_dir: str | pathlib.Path) -> dict[str, Any]:
    """Write placement and summary artifacts."""
    root = pathlib.Path(run_dir)
    events = read_events(root / "events.jsonl")
    placement = build_placement(events, root / "status.json")
    summary = summarize_events(events)

    write_json(root / "placement.json", placement)
    write_json(root / "summary.json", summary)
    return {
        "placement": placement,
        "summary": summary,
    }


def build_placement(
    events: list[RunEvent],
    status_path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Build agent placement proof from events and latest status."""
    agents: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.event != "agent_started" or not event.agent_id:
            continue
        placement = event.payload.get("placement")
        if isinstance(placement, dict):
            agents[event.agent_id] = {
                "agent_id": event.agent_id,
                "role": event.role,
                **placement,
            }

    if status_path is not None:
        path = pathlib.Path(status_path)
        if path.exists():
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
                states = status.get("agent_states", {})
                if isinstance(states, dict):
                    for agent_id, state in states.items():
                        if not isinstance(state, dict):
                            continue
                        placement = state.get("placement")
                        if isinstance(placement, dict):
                            agents.setdefault(
                                agent_id,
                                {
                                    "agent_id": agent_id,
                                    "role": state.get("role"),
                                    **placement,
                                },
                            )
            except json.JSONDecodeError:
                pass

    hostnames = sorted(
        {
            str(record.get("hostname"))
            for record in agents.values()
            if record.get("hostname")
        },
    )
    return {
        "agent_count": len(agents),
        "hostnames": hostnames,
        "distinct_hostname_count": len(hostnames),
        "agents": dict(sorted(agents.items())),
    }


def summarize_events(events: list[RunEvent]) -> dict[str, Any]:
    """Return compact run summary from campaign events."""
    counts = Counter(event.event for event in events)
    final_reports = _final_reports(events)
    return {
        "event_count": len(events),
        "event_counts": dict(sorted(counts.items())),
        "finish": _last_payload(
            events,
            {"campaign_finished", "workflow_finished", "run_finished"},
        ),
        "agent_errors": _payloads_of(events, "agent_error"),
        "message_count": counts.get("message_sent", 0),
        "final_reports": final_reports,
        "tool_results": _tool_result_summaries(events),
    }


def _last_payload(
    events: list[RunEvent],
    kinds: set[str],
) -> dict[str, Any] | None:
    payloads = [event.payload for event in events if event.event in kinds]
    return payloads[-1] if payloads else None


def _payloads_of(events: list[RunEvent], kind: str) -> list[dict[str, Any]]:
    return [
        {
            "agent_id": event.agent_id,
            "role": event.role,
            **event.payload,
        }
        for event in events
        if event.event == kind
    ]


def _final_reports(events: list[RunEvent]) -> list[dict[str, Any]]:
    reports = []
    for event in events:
        payload = event.payload
        if event.event == "belief_updated":
            reports.append(
                {
                    "agent_id": event.agent_id,
                    "summary": payload.get("summary") or payload.get("hypothesis"),
                    "confidence": payload.get("confidence"),
                    "supporting_message_ids": payload.get("supporting_message_ids", []),
                    "supporting_tool_result_ids": payload.get(
                        "supporting_tool_result_ids",
                        [],
                    ),
                },
            )
    return reports[-10:]


def _tool_result_summaries(events: list[RunEvent]) -> list[dict[str, Any]]:
    results = []
    for event in events:
        if event.event != "tool_call_finished":
            continue
        payload = event.payload
        results.append(
            {
                "timestamp": event.timestamp,
                "agent_id": event.agent_id,
                "tool_name": payload.get("tool_name"),
                "tool_result_id": payload.get("tool_result_id"),
                "status": payload.get("status"),
                "content_preview": payload.get("content_preview"),
            },
        )
    return results
