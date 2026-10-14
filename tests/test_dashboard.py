from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import chemgraph.dashboard as dashboard
from chemgraph.distributed.protocol import PeerEnvelope
from chemgraph.distributed.runtime import PeerRuntime
from chemgraph.observability.event_log import EventLog
from chemgraph.observability.run_files import write_json_atomic


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


def test_status_payload_builds_summary_from_events(tmp_path) -> None:
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
        "placement",
        "run_dir",
        "schema",
        "status",
        "summary",
        "updated",
    }
    assert payload["summary"]["message_count"] == 1
    assert payload["summary"]["final_reports"] == [
        {
            "agent_id": "agent-01",
            "confidence": 0.8,
            "summary": "used peer evidence",
            "supporting_message_ids": ["msg-1"],
            "supporting_tool_result_ids": [],
        },
    ]


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


@pytest.fixture
def peer_snapshot(tmp_path, request):
    state = getattr(request, "param", "waiting_for_user")
    interrupts = ({"id": "review", "payload": {"choices": ("approve", "reject")}},) if state == "waiting_for_user" else ()
    peer = PeerRuntime(name="worker", run_id="run", launch_id="launch", allowed_senders=(),
                       runner=SimpleNamespace(interrupts=interrupts))
    peer.state = state
    peer.output = "<script>result</script>"
    peer.last_error = "model unavailable" if state == "failed" else None
    peer.inbox.put_nowait(PeerEnvelope(run_id="run", launch_id="launch", sender="operator",
                                     recipient="worker", content="queued"))
    write_json_atomic(tmp_path / "status.json", peer.status())
    log = EventLog(tmp_path / "events.jsonl")
    log.emit("agent_started", agent_id="worker", payload={
        "workflow_type": "main_agent", "placement": {"hostname": "compute.example", "short_hostname": "compute"},
    })
    log.emit("agent_error", agent_id="worker", payload={"error": "earlier error"})
    return {**dashboard.status_payload(SimpleNamespace(run_dir=tmp_path)), **dashboard.events_payload(tmp_path)}


def test_distributed_status_is_adapted_without_rewriting_raw_status(tmp_path, peer_snapshot):
    assert peer_snapshot["schema"] == "distributed_peer"
    assert peer_snapshot["status"]["peer"] == "worker"
    assert peer_snapshot["status"]["agents"][0]["agent_id"] == "worker"
    assert peer_snapshot["status"]["agents"][0]["role"] == "main_agent"
    assert peer_snapshot["placement"]["agents"]["worker"]["hostname"] == "compute.example"
    assert peer_snapshot["updated"]
    assert "agents" not in json.loads((tmp_path / "status.json").read_text())


@pytest.mark.parametrize("peer_snapshot", ["waiting_for_user", "failed", "idle", "stopped"], indirect=True)
def test_distributed_dashboard_renders_actual_peer_status(peer_snapshot):
    quickjs = pytest.importorskip("quickjs")
    runtime = quickjs.Context()
    runtime.eval("""
        const elements = {};
        const document = {
          getElementById: id => elements[id] || (elements[id] = {
            style: {}, innerHTML: '', textContent: '', value: '',
            setAttribute: () => {}, querySelectorAll: () => [], addEventListener: () => {},
          }),
        };
    """)
    source = (Path(dashboard.__file__).parent / "static" / "app.js").read_text()
    # Load the actual rendering functions without starting network polling or DOM event bindings.
    runtime.eval(source.split("    document.getElementById('refresh').addEventListener", 1)[0])
    runtime.eval("snapshot = " + json.dumps(peer_snapshot))
    runtime.eval("selectedAgent = 'worker'; renderGraph(); renderAgentPicker(); renderMetrics(); renderDetail();")
    graph = runtime.eval("elements.graph.innerHTML")
    assert 'data-agent="worker"' in graph
    assert "No agents yet" not in graph
    assert peer_snapshot["status"]["state"] in graph
    assert "worker" in runtime.eval("elements.agentSelect.innerHTML")
    assert "Queued messages" in runtime.eval("elements.detailCards.innerHTML")
    assert runtime.eval("elements.detailPrimary.textContent") == "<script>result</script>"
    assert runtime.eval("elements.detailSecondary.textContent") == (peer_snapshot["status"]["last_error"] or "None")
    assert json.loads(runtime.eval("elements.detailTertiary.textContent")) == peer_snapshot["status"]["interrupts"]
    assert runtime.eval("agents()[0].last_error") == peer_snapshot["status"]["last_error"]
