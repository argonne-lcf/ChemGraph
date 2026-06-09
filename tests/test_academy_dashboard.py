from __future__ import annotations

import json

import chemgraph.academy.dashboard as dashboard
from chemgraph.academy.observability.event_log import EventLog


def test_dashboard_reads_canonical_events_jsonl(tmp_path) -> None:
    run_dir = tmp_path / "daemon-run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps({"mode": "mpi_daemon", "timestamp": 10.0, "agents": []})
        + "\n",
        encoding="utf-8",
    )
    log = EventLog(run_dir / "events.jsonl")
    log.emit(
        "agent_started",
        agent_id="agent-00",
        role="scheduler observer",
        payload={
            "role": "scheduler observer",
            "placement": {"hostname": "x1", "short_hostname": "x1"},
            "hostname": "x1",
            "short_hostname": "x1",
        },
    )
    log.emit(
        "agent_decision",
        agent_id="agent-00",
        role="scheduler observer",
        payload={
            "round": 1,
            "tool_names": ["send_message"],
            "actions": [{"action": "send_message"}],
        },
    )

    events = dashboard.events_payload(run_dir)["events"]

    assert events[0]["event"] == "agent_started"
    assert events[0]["payload"]["placement"]["hostname"] == "x1"
    assert events[1]["event"] == "agent_decision"
    assert events[1]["payload"]["actions"] == [{"action": "send_message"}]


def test_status_payload_builds_summary_and_proof_from_events(tmp_path) -> None:
    run_dir = tmp_path / "daemon-run"
    run_dir.mkdir()
    (run_dir / "status.json").write_text(
        json.dumps({"mode": "mpi_daemon", "agents": []}) + "\n",
        encoding="utf-8",
    )
    log = EventLog(run_dir / "events.jsonl")
    for agent_id, hostname in (("agent-00", "x0"), ("agent-01", "x1")):
        log.emit(
            "agent_started",
            agent_id=agent_id,
            role="observer",
            payload={
                "role": "observer",
                "placement": {"hostname": hostname, "short_hostname": hostname},
                "hostname": hostname,
                "short_hostname": hostname,
            },
        )
    log.emit(
        "message_sent",
        agent_id="agent-00",
        role="observer",
        payload={
            "message_id": "msg-1",
            "timestamp": 2.0,
            "sender": "agent-00",
            "recipient": "agent-01",
            "kind": "message",
            "content": "share evidence",
            "tldr": "evidence",
            "artifact_refs": [],
            "tool_result_ids": [],
        },
    )
    log.emit(
        "belief_updated",
        agent_id="agent-01",
        role="observer",
        payload={
            "hypothesis": "used peer evidence",
            "confidence": 0.8,
            "supporting_message_ids": ["msg-1"],
            "supporting_tool_result_ids": [],
        },
    )

    class Handler:
        pass

    handler = Handler()
    handler.run_dir = run_dir
    payload = dashboard.status_payload(handler)

    assert set(payload) == {
        "communication_proof",
        "placement",
        "run_dir",
        "schema",
        "status",
        "summary",
        "updated",
    }
    assert payload["summary"]["message_count"] == 1
    assert payload["communication_proof"]["passes"]["has_message"] is True
    assert payload["communication_proof"]["passes"]["has_cross_node_message"] is True
    assert payload["communication_proof"]["passes"]["has_belief_citing_message"] is True


def test_dashboard_ignores_legacy_trace_jsonl(tmp_path) -> None:
    run_dir = tmp_path / "old-run"
    run_dir.mkdir()
    (run_dir / "trace.jsonl").write_text(
        json.dumps(
            {
                "timestamp": 1.0,
                "agent": "agent-00",
                "event": "daemon_started",
                "payload": {"hostname": "x0"},
            },
        )
        + "\n",
        encoding="utf-8",
    )

    assert dashboard.events_payload(run_dir)["events"] == []
