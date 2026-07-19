from __future__ import annotations

import pytest

# Skip when the optional 'academy' extra is absent. The event_log
# module itself is pure stdlib, but the import guard is applied
# uniformly across the academy test suite.
pytest.importorskip("academy")

from chemgraph.academy.observability.event_log import EventLog, read_events


def test_event_log_preserves_payload_shape(tmp_path) -> None:
    log = EventLog(tmp_path / "events.jsonl")

    log.emit(
        "message_sent",
        run_id="run-1",
        agent_id="agent-a",
        role="Worker",
        payload={
            "message_id": "msg-1",
            "recipient": "agent-b",
            "tldr": "short",
        },
    )

    event = read_events(tmp_path / "events.jsonl")[0]
    assert event.event == "message_sent"
    assert event.payload == {
        "message_id": "msg-1",
        "recipient": "agent-b",
        "tldr": "short",
    }
