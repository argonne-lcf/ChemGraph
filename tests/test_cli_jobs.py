"""Tests for the `chemgraph jobs` CLI subcommand.

Hermetic: writes JobTracker persist files on disk, no network, no LLM. Tasks
carry cached results or are left pending (globus_task_id=None), so no Globus
client is ever constructed.
"""

import argparse
import json
from datetime import datetime, timezone

import pytest

from chemgraph.cli import jobs as jobs_mod
from chemgraph.cli.formatting import console


def _write_jobs_file(path, batch_id, tool_name, task_results):
    """Write a persist file matching JobTracker._save's on-disk format.

    task_results: list where each element is a result dict (done) or None (pending).
    """
    tasks = [
        {
            "task_id": f"t{i}",
            "meta": {"idx": i},
            "globus_task_id": None,
            "result": result,
        }
        for i, result in enumerate(task_results)
    ]
    data = {
        batch_id: {
            "tool_name": tool_name,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "tasks": tasks,
        }
    }
    path.write_text(json.dumps(data, indent=2))


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    d = tmp_path / ".chemgraph"
    d.mkdir()
    files = {
        "mace": d / "mace_jobs.json",
        "ase": d / "ase_jobs.json",
        "graspa": d / "graspa_jobs.json",
        "xanes": d / "xanes_jobs.json",
    }
    monkeypatch.setattr(jobs_mod, "_JOBS_DIR", d)
    monkeypatch.setattr(jobs_mod, "_JOBS_FILES", files)
    return files


def _run(**ns):
    # Rich falls back to an 80-col width under capture(), which truncates the
    # 12-char batch IDs (e.g. "aaaa1…"). Force a realistic terminal width so the
    # rendered table matches what a user sees, then restore.
    prev_width = console.width
    console.width = 200
    try:
        with console.capture() as cap:
            jobs_mod.handle_jobs(argparse.Namespace(**ns))
        return cap.get()
    finally:
        console.width = prev_width


def test_no_files_message(jobs_dir):
    out = _run(jobs_command="list")
    assert "No job tracker files found" in out


def test_bare_jobs_prints_usage(jobs_dir):
    # No subcommand -> usage, not a silent default to `list`.
    out = _run(jobs_command=None)
    assert "Usage: chemgraph jobs" in out


def test_unknown_subcommand_prints_usage(jobs_dir):
    out = _run(jobs_command="bogus")
    assert "Usage: chemgraph jobs" in out


def test_list_offline_never_builds_globus_client(jobs_dir, monkeypatch):
    # A disk-loaded task with a globus_task_id but no result would trigger a
    # Globus query if offline were not forwarded. The CLI must stay offline.
    tasks = [
        {
            "task_id": "t0",
            "meta": {"idx": 0},
            "globus_task_id": "gc-uuid",
            "result": None,
        }
    ]
    data = {
        "ffff00001111": {
            "tool_name": "run_ase_opt",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "tasks": tasks,
        }
    }
    jobs_dir["ase"].write_text(json.dumps(data, indent=2))

    from chemgraph.execution.job_tracker import JobTracker

    def _boom(self):
        raise AssertionError("CLI must not construct a Globus client")

    monkeypatch.setattr(JobTracker, "_get_gc_client", _boom)

    out = _run(jobs_command="list")
    assert "ffff00001111" in out
    assert "pending" in out


def test_list_combines_backends(jobs_dir):
    _write_jobs_file(
        jobs_dir["mace"], "aaaa11112222", "run_mace_opt", [{"status": "success"}]
    )
    _write_jobs_file(jobs_dir["ase"], "bbbb33334444", "run_ase_opt", [None])
    out = _run(jobs_command="list")
    assert "aaaa11112222" in out
    assert "bbbb33334444" in out
    assert "mace" in out and "ase" in out
    assert "completed" in out and "pending" in out


def test_status_found(jobs_dir):
    _write_jobs_file(
        jobs_dir["mace"],
        "aaaa11112222",
        "run_mace_opt",
        [{"status": "success"}, None],
    )
    out = _run(jobs_command="status", batch_id="aaaa11112222")
    assert "run_mace_opt" in out
    assert "running" in out  # 1 done, 1 pending
    assert "50.0%" in out


def test_status_not_found(jobs_dir):
    _write_jobs_file(jobs_dir["mace"], "aaaa11112222", "run_mace_opt", [None])
    out = _run(jobs_command="status", batch_id="deadbeef0000")
    assert "not found" in out


def test_results_completed(jobs_dir):
    _write_jobs_file(
        jobs_dir["ase"],
        "cccc55556666",
        "run_ase_opt",
        [{"status": "success", "energy": -1.23}],
    )
    out = _run(jobs_command="results", batch_id="cccc55556666", partial=False)
    assert "energy" in out and "-1.23" in out


def test_results_pending_blocks(jobs_dir):
    _write_jobs_file(
        jobs_dir["ase"],
        "dddd77778888",
        "run_ase_opt",
        [{"status": "success"}, None],
    )
    out = _run(jobs_command="results", batch_id="dddd77778888", partial=False)
    assert "pending" in out


def test_results_partial(jobs_dir):
    _write_jobs_file(
        jobs_dir["ase"],
        "eeee9999aaaa",
        "run_ase_opt",
        [{"status": "success", "energy": -9.9}, None],
    )
    out = _run(jobs_command="results", batch_id="eeee9999aaaa", partial=True)
    assert "energy" in out
