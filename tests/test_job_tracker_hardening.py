"""Regression tests for JobTracker robustness fixes.

- _save must not crash on non-JSON-serializable task results (persist path).
- register_batch must not block the full globus wait-timeout for plain futures.
"""

import time
from concurrent.futures import Future
from unittest.mock import Mock

from chemgraph.execution.job_tracker import JobTracker


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


def test_registration_does_not_reprocess_a_result_collected_by_a_poll(monkeypatch):
    tracker = JobTracker()
    future = Future()
    future.set_result({"status": "success"})
    post = Mock(side_effect=lambda meta, raw: {**meta, **raw})

    def poll_during_registration(tasks, timeout):
        tracker.get_status(next(iter(tracker._batches)))

    monkeypatch.setattr(tracker, "_wait_for_globus_task_ids", poll_during_registration)
    batch_id = tracker.register_batch("tool", [({"task_id": "one"}, future)], post_fn=post)
    assert tracker.get_results(batch_id)["status"] == "completed"
    post.assert_called_once()
