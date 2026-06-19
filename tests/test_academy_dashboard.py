from __future__ import annotations

import json

import pytest

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


# ---------------------------------------------------------------------------
# B.4c: federated dashboard (multi-site subdir layout)
# ---------------------------------------------------------------------------


def _seed_site(site_dir, *, status_timestamp, events) -> None:
    """Write a minimal per-site mirror: status.json + events.jsonl.

    Touches an empty events.jsonl even when ``events`` is empty so the
    federated-dashboard detector recognizes the dir as a real site
    (EventLog itself only creates the file on first emit, which is
    too late for the iterator's existence check).
    """
    site_dir.mkdir(parents=True)
    (site_dir / "status.json").write_text(
        json.dumps({"mode": "mpi_daemon", "timestamp": status_timestamp, "agents": []})
        + "\n",
        encoding="utf-8",
    )
    (site_dir / "events.jsonl").touch()
    log = EventLog(site_dir / "events.jsonl")
    for event_name, payload in events:
        log.emit(
            event_name,
            agent_id=payload["agent_id"],
            role="observer",
            payload=payload,
        )


def test_events_payload_merges_sites_and_tags_each_event(tmp_path) -> None:
    """Federated dashboard's core promise: pointed at a parent dir with
    per-site subdirs, ``events_payload`` returns a single timestamp-
    sorted stream where each event carries a ``site`` field. UI uses
    that field to color-code per-site events in the merged view."""
    root = tmp_path / "federated-run"
    _seed_site(
        root / "aurora",
        status_timestamp=10.0,
        events=[
            ("agent_started", {
                "agent_id": "agent-00", "role": "observer",
                "placement": {"hostname": "aur1", "short_hostname": "aur1"},
                "hostname": "aur1", "short_hostname": "aur1",
            }),
        ],
    )
    _seed_site(
        root / "crux",
        status_timestamp=20.0,
        events=[
            ("agent_started", {
                "agent_id": "agent-01", "role": "observer",
                "placement": {"hostname": "crux1", "short_hostname": "crux1"},
                "hostname": "crux1", "short_hostname": "crux1",
            }),
        ],
    )

    payload = dashboard.events_payload(root)

    assert {e["site"] for e in payload["events"]} == {"aurora", "crux"}
    # Both sites' agents are visible in the merged stream.
    agents_by_site = {e["site"]: e["agent_id"] for e in payload["events"]}
    assert agents_by_site == {"aurora": "agent-00", "crux": "agent-01"}


def test_events_payload_sorts_merged_stream_by_timestamp(tmp_path) -> None:
    """Per-site clocks don't have to agree, but the merged dashboard
    view must be readable -- order by event timestamp regardless of
    which site emitted each."""
    root = tmp_path / "federated-run"

    # Sites are seeded in reverse-time order; the merge must still
    # produce timestamp-ascending output.
    aurora_dir = root / "aurora"
    aurora_dir.mkdir(parents=True)
    (aurora_dir / "status.json").write_text("{}", encoding="utf-8")
    aurora_log = EventLog(aurora_dir / "events.jsonl")
    aurora_log.emit("agent_started", agent_id="ag-aur", role="r", payload={
        "agent_id": "ag-aur", "role": "r",
        "placement": {"hostname": "h"}, "hostname": "h",
    })

    crux_dir = root / "crux"
    crux_dir.mkdir(parents=True)
    (crux_dir / "status.json").write_text("{}", encoding="utf-8")
    crux_log = EventLog(crux_dir / "events.jsonl")
    crux_log.emit("agent_started", agent_id="ag-crux", role="r", payload={
        "agent_id": "ag-crux", "role": "r",
        "placement": {"hostname": "h"}, "hostname": "h",
    })

    events = dashboard.events_payload(root)["events"]
    timestamps = [e["timestamp"] for e in events]
    assert timestamps == sorted(timestamps)


def test_status_payload_nests_under_sites_for_federated_layout(tmp_path) -> None:
    """Single-site clients use ``payload['status']`` /
    ``payload['summary']`` etc directly. Federated clients want a
    ``sites: {<name>: {...}}`` shape so the UI can render per-site
    sub-panels. Pin the structural difference so a future "make them
    uniform" refactor must be a conscious choice."""
    root = tmp_path / "federated-run"
    _seed_site(
        root / "aurora",
        status_timestamp=10.0,
        events=[],
    )
    _seed_site(
        root / "crux",
        status_timestamp=15.0,
        events=[],
    )

    class Handler:
        pass
    handler = Handler()
    handler.run_dir = root

    payload = dashboard.status_payload(handler)
    assert "sites" in payload
    assert set(payload["sites"]) == {"aurora", "crux"}
    for site_name, site_payload in payload["sites"].items():
        assert "status" in site_payload
        assert "summary" in site_payload
        assert "placement" in site_payload
    # Top-level 'updated' reflects the latest per-site update so the
    # dashboard header has a meaningful timestamp.
    assert payload["updated"] == 15.0


def test_status_payload_keeps_legacy_shape_for_single_site(tmp_path) -> None:
    """Existing single-site dashboard clients must see exactly the
    pre-federation payload shape. The federated nesting only kicks in
    when ``events.jsonl`` is absent at the top level."""
    root = tmp_path / "single-run"
    _seed_site(
        root,
        status_timestamp=10.0,
        events=[],
    )

    class Handler:
        pass
    handler = Handler()
    handler.run_dir = root

    payload = dashboard.status_payload(handler)
    # Single-site keys, no ``sites`` nesting.
    assert "sites" not in payload
    assert set(payload) == {
        "placement", "run_dir", "schema", "status", "summary", "updated",
    }


def test_iter_site_dirs_recognizes_metadata_only_sites(tmp_path) -> None:
    """A site that's started but hasn't emitted any events yet still
    has a ``dashboard_metadata.json`` written by the launcher. The
    iterator must recognize it so a federated dashboard doesn't
    briefly look like 'empty single-site' during startup."""
    from chemgraph.academy.dashboard.server import _iter_site_dirs

    root = tmp_path / "early-startup"
    (root / "aurora").mkdir(parents=True)
    (root / "aurora" / "dashboard_metadata.json").write_text("{}", encoding="utf-8")
    (root / "crux").mkdir(parents=True)
    (root / "crux" / "dashboard_metadata.json").write_text("{}", encoding="utf-8")

    sites = _iter_site_dirs(root)
    assert {name for name, _ in sites} == {"aurora", "crux"}


def test_iter_site_dirs_falls_back_to_single_site_when_empty(tmp_path) -> None:
    """Just-created run dir with neither events.jsonl nor recognizable
    subdirs: behave as single-site so the dashboard renders an
    empty-but-valid view instead of erroring out."""
    from chemgraph.academy.dashboard.server import _iter_site_dirs

    root = tmp_path / "brand-new"
    root.mkdir()
    sites = _iter_site_dirs(root)
    assert sites == [(None, root)]


# ---------------------------------------------------------------------------
# Multi-site launcher argument parsing
# ---------------------------------------------------------------------------


def test_parse_systems_list_accepts_single_name() -> None:
    """Single-site invocation is the legacy case; tuple-of-one keeps
    the rest of the launcher uniform."""
    from chemgraph.academy.runtime.dashboard_launcher import _parse_systems_list
    assert _parse_systems_list("aurora") == ("aurora",)


def test_parse_systems_list_accepts_comma_list_and_trims() -> None:
    """The federated UX. Whitespace-tolerant for paste-from-doc."""
    from chemgraph.academy.runtime.dashboard_launcher import _parse_systems_list
    assert _parse_systems_list(" aurora , crux ") == ("aurora", "crux")
    assert _parse_systems_list("aurora,crux,") == ("aurora", "crux")


def test_parse_systems_list_rejects_empty() -> None:
    """Operator typo or unexpected expansion -- fail at argparse-resolve
    time with a clean message."""
    import argparse
    from chemgraph.academy.runtime.dashboard_launcher import _parse_systems_list
    with pytest.raises(argparse.ArgumentTypeError, match="at least one"):
        _parse_systems_list("")
    with pytest.raises(argparse.ArgumentTypeError, match="at least one"):
        _parse_systems_list(",")


def test_parse_systems_list_rejects_duplicates() -> None:
    """Listing the same site twice would set up duplicate tunnels +
    rsync threads racing on the same mirror dir. Fail closed."""
    import argparse
    from chemgraph.academy.runtime.dashboard_launcher import _parse_systems_list
    with pytest.raises(argparse.ArgumentTypeError, match="duplicate"):
        _parse_systems_list("aurora,crux,aurora")
