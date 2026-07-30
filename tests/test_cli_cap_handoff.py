"""Tests for the wall-clock-cap resume handoff printed by the CLI.

Hermetic: exercises the ``_print_cap_handoff`` helper directly against a fake
agent/manifest and a captured Rich console. No LLM, no network, no CLI spin-up.
"""

import io

from rich.console import Console

from chemgraph.cli.commands import _print_cap_handoff


class _FakeManifest:
    def __init__(self, status, pending):
        self.status = status
        self._data = {"pending_next_step": pending}


class _FakeAgent:
    def __init__(self, status, pending, session_id="sess-abc123"):
        self.run_manifest = _FakeManifest(status, pending)
        self.session_id = session_id


def _capture(agent) -> str:
    console = Console(file=io.StringIO(), width=100)
    _print_cap_handoff(console, agent)
    return console.file.getvalue()


def test_handoff_printed_when_capped_with_restart():
    reason = "wall-clock cap; resume with restart_file=/x/restart_opt.json"
    agent = _FakeAgent(
        status="capped",
        pending={"tool": "run_ase", "args": {}, "reason": reason},
        session_id="sess-abc123",
    )
    out = _capture(agent)
    assert "resume" in out
    assert "sess-abc123" in out
    assert "restart_file" in out
    assert "chemgraph resume sess-abc123" in out


def test_handoff_printed_when_capped_no_restart():
    reason = (
        "wall-clock cap; no restart written, rerun with more wall-clock budget"
    )
    agent = _FakeAgent(
        status="capped",
        pending={"tool": "run_ase", "args": {}, "reason": reason},
        session_id="sess-def456",
    )
    out = _capture(agent)
    assert "rerun with more wall-clock budget" in out
    assert "chemgraph resume sess-def456" in out


def test_no_handoff_when_not_capped():
    agent = _FakeAgent(
        status="running",
        pending=None,
        session_id="sess-xyz789",
    )
    out = _capture(agent)
    assert "resume" not in out
    assert out.strip() == ""


def test_no_crash_when_no_manifest():
    class _AgentNoManifest:
        session_id = "sess-none"

    # Agent lacking a run_manifest attribute entirely.
    out_missing = _capture(_AgentNoManifest())
    assert out_missing.strip() == ""

    # Agent with run_manifest explicitly None.
    class _AgentNoneManifest:
        run_manifest = None
        session_id = "sess-none"

    out_none = _capture(_AgentNoneManifest())
    assert out_none.strip() == ""
