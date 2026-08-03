"""Regression tests for JobTracker robustness fixes.

- _save must not crash on non-JSON-serializable task results (persist path).
- register_batch must not block the full globus wait-timeout for plain futures.
- offline=True must never construct a Globus Compute client (CLI path).
- _load must survive structurally incomplete / wrong-shape persist files.
"""

import json
import time
from concurrent.futures import Future

from chemgraph.execution.job_tracker import JobTracker


def _write_persist(path, data):
    path.write_text(json.dumps(data))


def _disk_batch(*, tool_name="run_ase", submitted_at="2026-07-28T12:00:00+00:00",
                tasks=None):
    """A minimal on-disk batch dict in JobTracker._save's format."""
    return {
        "tool_name": tool_name,
        "submitted_at": submitted_at,
        "tasks": tasks if tasks is not None else [],
    }


class _Unserializable:
    """A value json.dump cannot handle without default=str."""

    def __repr__(self):
        return "<Unserializable>"


def test_get_status_does_not_crash_on_unserializable_result(tmp_path):
    persist = tmp_path / "jobs.json"
    tracker = JobTracker(persist_file=persist)

    fut: Future = Future()
    fut.set_result(_Unserializable())  # non-dict, non-JSON-serializable
    batch_id = tracker.register_batch("run_ase", [({"task_id": "t0"}, fut)])

    # Previously raised TypeError from json.dump inside _save.
    status = tracker.get_status(batch_id)
    assert status["status"] == "completed"
    assert persist.is_file()  # best-effort persistence still wrote something


def test_register_batch_does_not_block_for_plain_futures():
    tracker = JobTracker()  # no persist
    fut: Future = Future()  # pending, plain future with no ``task_id`` attr

    start = time.monotonic()
    tracker.register_batch("run_ase", [({"task_id": "t0"}, fut)])
    elapsed = time.monotonic() - start

    # Must not wait out the 3s globus task-id deadline for a plain future.
    assert elapsed < 1.0, f"register_batch blocked for {elapsed:.2f}s"
    fut.set_result({"status": "success"})  # let the future resolve cleanly


# ── offline=True must never touch Globus ──────────────────────────────────


def test_offline_status_does_not_construct_globus_client(tmp_path, monkeypatch):
    persist = tmp_path / "jobs.json"
    # A batch loaded from disk with a globus_task_id but no cached result:
    # this is exactly the case get_status(offline=False) would query Globus for.
    _write_persist(
        persist,
        {
            "b0": _disk_batch(
                tasks=[{"task_id": "t0", "meta": {}, "globus_task_id": "gc-uuid",
                        "result": None}]
            )
        },
    )
    tracker = JobTracker(persist_file=persist)

    # Make any attempt to build a client an explicit test failure.
    def _boom():
        raise AssertionError("offline=True must not build a Globus client")

    monkeypatch.setattr(tracker, "_get_gc_client", _boom)

    status = tracker.get_status("b0", offline=True)
    assert status["status"] == "pending"
    assert status["pending_tasks"] == 1

    # list_batches and get_results must forward offline too.
    summaries = tracker.list_batches(offline=True)
    assert summaries and summaries[0]["status"] == "pending"
    res = tracker.get_results("b0", offline=True)
    assert res["status"] == "pending"
    assert "results" not in res  # blocked: still pending


# ── _load hardening: structurally-incomplete / wrong-shape files ──────────


def test_load_skips_batch_missing_required_fields(tmp_path):
    persist = tmp_path / "jobs.json"
    _write_persist(
        persist,
        {
            "good": _disk_batch(tasks=[{"task_id": "t0", "result": {"status": "success"}}]),
            "no_tool": {"submitted_at": "2026-07-28T12:00:00+00:00", "tasks": []},
            "no_time": {"tool_name": "run_ase", "tasks": []},
            "bad_time": _disk_batch(submitted_at="not-a-timestamp"),
            "not_a_dict": ["oops"],
        },
    )
    # Must not raise; the good batch survives, the malformed ones are skipped.
    tracker = JobTracker(persist_file=persist)
    assert tracker.get_status("good", offline=True)["status"] == "completed"
    assert "error" in tracker.get_status("no_tool", offline=True)
    assert "error" in tracker.get_status("no_time", offline=True)
    assert "error" in tracker.get_status("bad_time", offline=True)
    assert "error" in tracker.get_status("not_a_dict", offline=True)


def test_load_skips_batch_with_malformed_task(tmp_path):
    persist = tmp_path / "jobs.json"
    _write_persist(
        persist,
        {"b0": _disk_batch(tasks=[{"meta": {}}])},  # task missing task_id
    )
    tracker = JobTracker(persist_file=persist)
    assert "error" in tracker.get_status("b0", offline=True)


def test_load_ignores_non_object_top_level(tmp_path):
    persist = tmp_path / "jobs.json"
    for payload in ([], "a string", 42):
        _write_persist(persist, payload)
        tracker = JobTracker(persist_file=persist)  # must not raise
        assert tracker.list_batches(offline=True) == []
