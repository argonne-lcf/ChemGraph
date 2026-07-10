from __future__ import annotations

from chemgraph.academy_sim.envelopes import build_envelope, envelope_to_prompt


def test_envelope_round_trip_prompt():
    envelope = build_envelope(
        run_id="run-1",
        sender="planner",
        recipient="executor",
        content="Run ASE for water.",
        correlation_id="task-1",
    )

    prompt = envelope_to_prompt(envelope)

    assert envelope.message_id.startswith("cg-peer-")
    assert "planner to executor" in prompt
    assert "task-1" in prompt
    assert "Run ASE for water." in prompt
