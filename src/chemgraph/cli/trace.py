"""Trace writer for traditional ChemGraph CLI runs.

Bridges the `run_turn` event callback into the dashboard's on-disk
schema (`events.jsonl` + `status.json` + `manifest.json`), so the
existing ``chemgraph dashboard`` browser UI can render a single-agent
ChemGraph run without going through the Academy daemon path.
"""

from __future__ import annotations

import time
from pathlib import Path

from chemgraph.academy.observability.event_log import EventLog
from chemgraph.academy.observability.run_files import write_json_atomic


_AGENT_ID = "chemgraph"
_AGENT_ROLE = "single_agent"


class CLIRunTrace:
    """Writer for a single traditional ChemGraph run.

    Produces the on-disk layout the dashboard expects:

    ::

        <trace_dir>/events.jsonl
        <trace_dir>/status.json
        <trace_dir>/manifest.json

    The ``status.json.mode`` field is ``"chemgraph_workflow"`` so the
    dashboard renders the per-agent workflow inspector (the "inner tab"
    you'd see if you clicked a logical-agent node in an Academy run).
    """

    def __init__(
        self,
        trace_dir: Path,
        *,
        run_id: str | None = None,
        model_name: str | None = None,
        workflow_type: str | None = None,
        query: str | None = None,
    ) -> None:
        self.trace_dir = Path(trace_dir)
        self.run_id = run_id or self.trace_dir.name
        self.model_name = model_name
        self.workflow_type = workflow_type
        self.query = query
        self._log = EventLog(self.trace_dir / "events.jsonl")

    def start(self) -> None:
        """Initialise the run directory and write the static metadata."""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            self.trace_dir / "manifest.json",
            {
                "mode": "chemgraph_workflow",
                "run_id": self.run_id,
                "model": self.model_name,
                "workflow_type": self.workflow_type,
            },
        )
        self._write_status()
        self._log.emit(
            "run_started",
            run_id=self.run_id,
            agent_id=_AGENT_ID,
            role=_AGENT_ROLE,
            payload={
                "model": self.model_name,
                "workflow_type": self.workflow_type,
                "query": self.query,
            },
        )

    def finish(self, *, status: str, error: str | None = None) -> None:
        """Mark the run as completed and refresh ``status.json``."""
        self._log.emit(
            "run_finished",
            run_id=self.run_id,
            agent_id=_AGENT_ID,
            role=_AGENT_ROLE,
            payload={"status": status, "error": error} if error else {"status": status},
        )
        self._write_status()

    def on_event(self, event: str, payload: dict) -> None:
        """Callback handed to :func:`chemgraph.agent.llm_agent.run_turn`."""
        self._log.emit(
            event,  # type: ignore[arg-type]
            run_id=self.run_id,
            agent_id=_AGENT_ID,
            role=_AGENT_ROLE,
            payload=payload,
        )

    def _write_status(self) -> None:
        write_json_atomic(
            self.trace_dir / "status.json",
            {
                "mode": "chemgraph_workflow",
                "updated": time.time(),
                "agents": [
                    {
                        "agent_id": _AGENT_ID,
                        "agent_name": _AGENT_ID,
                        "role": _AGENT_ROLE,
                    },
                ],
            },
        )
