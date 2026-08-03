"""CLI handlers for inspecting HPC job batches.

Exposes the JobTracker persistence layer (previously reachable only through MCP
tools) on the command line, so a user can answer "what did I submit last
allocation?" and reclaim results without going through the agent.

Reads the ``~/.chemgraph/*_jobs.json`` files the MCP-HPC servers persist. All
JobTracker calls here pass ``offline=True`` so listing/inspecting batches never
constructs a Globus Compute client or makes a network call -- the CLI must not
trigger an interactive Globus OAuth login. As a consequence, a disk-loaded task
whose result has not yet been cached is reported as still ``pending`` even if it
has actually finished remotely; the MCP tools (which run ``offline=False``) are
the path that fetches such results and writes them back to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from chemgraph.cli.formatting import console
from chemgraph.execution.job_tracker import JobTracker

# The MCP-HPC servers each persist to their own file under ~/.chemgraph.
# Keep in sync with the _JOBS_FILE constants in mcp/{mace,ase,graspa,xanes}_mcp_hpc.py.
_JOBS_DIR = Path("~/.chemgraph").expanduser()
_JOBS_FILES = {
    "mace": _JOBS_DIR / "mace_jobs.json",
    "ase": _JOBS_DIR / "ase_jobs.json",
    "graspa": _JOBS_DIR / "graspa_jobs.json",
    "xanes": _JOBS_DIR / "xanes_jobs.json",
}

_STATUS_STYLE = {
    "completed": "green",
    "partial": "yellow",
    "running": "cyan",
    "pending": "dim",
    "failed": "red",
}


def _discover_trackers() -> dict[str, JobTracker]:
    """Return ``{backend: JobTracker}`` for every persist file that exists."""
    trackers: dict[str, JobTracker] = {}
    for backend, path in _JOBS_FILES.items():
        if path.is_file():
            trackers[backend] = JobTracker(persist_file=path)
    return trackers


def _find_batch(trackers: dict[str, JobTracker], batch_id: str):
    """Locate the (backend, tracker) owning *batch_id*, or (None, None)."""
    for backend, tracker in trackers.items():
        status = tracker.get_status(batch_id, offline=True)
        if "error" not in status:
            return backend, tracker
    return None, None


def _fmt_status(status: str) -> str:
    return f"[{_STATUS_STYLE.get(status, 'white')}]{status}[/]"


_USAGE = "Usage: chemgraph jobs {list,status <batch>,results <batch>}."


def handle_jobs(args: argparse.Namespace) -> None:
    """Dispatch ``chemgraph jobs {list,status,results}``."""
    command = getattr(args, "jobs_command", None)

    # A bare ``chemgraph jobs`` (no subcommand) shows usage; it does not
    # silently run ``list``. The subcommand is required and this keeps
    # the noun/verb contract explicit.
    if command is None:
        console.print(_USAGE)
        return
    if command not in ("list", "status", "results"):
        console.print(_USAGE)
        return

    trackers = _discover_trackers()
    if not trackers:
        console.print(
            f"[dim]No job tracker files found under {_JOBS_DIR}. Nothing has "
            "been submitted via the MCP-HPC backends yet.[/dim]"
        )
        return

    if command == "list":
        _jobs_list(trackers)
    elif command == "status":
        _jobs_status(trackers, args.batch_id)
    elif command == "results":
        _jobs_results(trackers, args.batch_id, include_partial=args.partial)


def _jobs_list(trackers: dict[str, JobTracker]) -> None:
    """Print a combined table of every batch across all backends."""
    rows: list[tuple[str, dict]] = []
    for backend, tracker in trackers.items():
        for summary in tracker.list_batches(offline=True):
            rows.append((backend, summary))

    if not rows:
        console.print("[dim]No job batches tracked.[/dim]")
        return

    console.print(Panel(f"Tracked Job Batches ({len(rows)})", style="bold cyan"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Backend", style="cyan")
    table.add_column("Batch ID", style="cyan")
    table.add_column("Tool", style="green")
    table.add_column("Status")
    table.add_column("Progress", justify="right")
    table.add_column("Done", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Pending", justify="right")
    table.add_column("Submitted", style="dim")

    for backend, s in sorted(
        rows, key=lambda r: r[1].get("submitted_at", ""), reverse=True
    ):
        table.add_row(
            backend,
            s["batch_id"],
            s["tool_name"],
            _fmt_status(s["status"]),
            f"{s['progress_pct']}%",
            str(s["completed_tasks"]),
            str(s["failed_tasks"]),
            str(s["pending_tasks"]),
            s["submitted_at"].replace("T", " ")[:19],
        )

    console.print(table)
    console.print(
        "\n[dim]Use 'chemgraph jobs status <batch>' or "
        "'chemgraph jobs results <batch>' for details.[/dim]"
    )


def _jobs_status(trackers: dict[str, JobTracker], batch_id: str) -> None:
    backend, tracker = _find_batch(trackers, batch_id)
    if tracker is None:
        console.print(f"[red]Batch '{batch_id}' not found in any tracker.[/red]")
        console.print("[dim]Use 'chemgraph jobs list' to see batches.[/dim]")
        return

    s = tracker.get_status(batch_id, offline=True)
    meta = Table(show_header=False, box=None, padding=(0, 2))
    meta.add_column("Key", style="bold cyan")
    meta.add_column("Value")
    meta.add_row("Backend", backend)
    meta.add_row("Batch ID", s["batch_id"])
    meta.add_row("Tool", s["tool_name"])
    meta.add_row("Submitted", s["submitted_at"].replace("T", " ")[:19])
    meta.add_row("Status", _fmt_status(s["status"]))
    meta.add_row("Progress", f"{s['progress_pct']}%")
    meta.add_row("Total tasks", str(s["total_tasks"]))
    meta.add_row("Completed", str(s["completed_tasks"]))
    meta.add_row("Failed", str(s["failed_tasks"]))
    meta.add_row("Pending", str(s["pending_tasks"]))
    console.print(Panel(meta, title="Batch Status", style="bold cyan"))


def _jobs_results(
    trackers: dict[str, JobTracker], batch_id: str, include_partial: bool
) -> None:
    backend, tracker = _find_batch(trackers, batch_id)
    if tracker is None:
        console.print(f"[red]Batch '{batch_id}' not found in any tracker.[/red]")
        return

    res = tracker.get_results(
        batch_id, include_partial=include_partial, offline=True
    )
    if "error" in res:
        console.print(f"[red]{res['error']}[/red]")
        return
    if "results" not in res:
        # Still pending and include_partial=False. get_results supplies a
        # "message" on this branch today; fall back defensively so a future
        # path that returns neither "results" nor "message" cannot KeyError.
        console.print(
            f"[yellow]{res.get('message', 'No results available yet.')}[/yellow]"
        )
        return

    console.print(
        Panel(
            f"Results for {batch_id} ({backend}) - {len(res['results'])} task(s)",
            style="bold cyan",
        )
    )
    console.print(
        Syntax(
            json.dumps(res["results"], indent=2, default=str), "json", theme="monokai"
        )
    )
