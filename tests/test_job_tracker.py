"""Tests for the JobTracker and submit_or_gather utilities."""

import asyncio
from concurrent.futures import Future
from unittest.mock import MagicMock

import pytest

from chemgraph.execution.job_tracker import JobTracker
from chemgraph.execution.utils import gather_futures, submit_or_gather


# ── Helpers ────────────────────────────────────────────────────────────


def _make_done_future(result):
    """Create a Future that is already resolved with *result*."""
    fut = Future()
    fut.set_result(result)
    return fut


def _make_failed_future(exc):
    """Create a Future that is already resolved with an exception."""
    fut = Future()
    fut.set_exception(exc)
    return fut


def _make_pending_future():
    """Create a Future that is not yet resolved."""
    return Future()


# ── JobTracker.register_batch ──────────────────────────────────────────


class TestRegisterBatch:
    def test_returns_batch_id(self):
        tracker = JobTracker()
        fut = _make_pending_future()
        batch_id = tracker.register_batch(
            "test_tool", [({"key": "val"}, fut)]
        )
        assert isinstance(batch_id, str)
        assert len(batch_id) == 12

    def test_stores_tasks(self):
        tracker = JobTracker()
        futs = [_make_pending_future() for _ in range(3)]
        pending = [({"idx": i}, f) for i, f in enumerate(futs)]
        batch_id = tracker.register_batch("test_tool", pending)

        status = tracker.get_status(batch_id)
        assert status["total_tasks"] == 3

    def test_multiple_batches_unique_ids(self):
        tracker = JobTracker()
        ids = set()
        for _ in range(10):
            bid = tracker.register_batch(
                "tool", [({"x": 1}, _make_pending_future())]
            )
            ids.add(bid)
        assert len(ids) == 10


# ── JobTracker.get_status ──────────────────────────────────────────────


class TestGetStatus:
    def test_all_pending(self):
        tracker = JobTracker()
        pending = [({"i": i}, _make_pending_future()) for i in range(3)]
        batch_id = tracker.register_batch("tool", pending)

        status = tracker.get_status(batch_id)
        assert status["status"] == "pending"
        assert status["total_tasks"] == 3
        assert status["completed_tasks"] == 0
        assert status["pending_tasks"] == 3
        assert status["progress_pct"] == 0.0

    def test_all_completed(self):
        tracker = JobTracker()
        pending = [
            ({"i": i}, _make_done_future({"val": i})) for i in range(3)
        ]
        batch_id = tracker.register_batch("tool", pending)

        status = tracker.get_status(batch_id)
        assert status["status"] == "completed"
        assert status["completed_tasks"] == 3
        assert status["failed_tasks"] == 0
        assert status["pending_tasks"] == 0
        assert status["progress_pct"] == 100.0

    def test_partial_done(self):
        tracker = JobTracker()
        pending = [
            ({"i": 0}, _make_done_future({"val": 0})),
            ({"i": 1}, _make_pending_future()),
        ]
        batch_id = tracker.register_batch("tool", pending)

        status = tracker.get_status(batch_id)
        assert status["status"] == "running"
        assert status["completed_tasks"] == 1
        assert status["pending_tasks"] == 1
        assert status["progress_pct"] == 50.0

    def test_all_failed(self):
        tracker = JobTracker()
        pending = [
            ({"i": i}, _make_failed_future(ValueError(f"err_{i}")))
            for i in range(2)
        ]
        batch_id = tracker.register_batch("tool", pending)

        status = tracker.get_status(batch_id)
        assert status["status"] == "failed"
        assert status["failed_tasks"] == 2

    def test_mixed_success_and_failure(self):
        tracker = JobTracker()
        pending = [
            ({"i": 0}, _make_done_future({"val": 0})),
            ({"i": 1}, _make_failed_future(RuntimeError("boom"))),
        ]
        batch_id = tracker.register_batch("tool", pending)

        status = tracker.get_status(batch_id)
        assert status["status"] == "partial"
        assert status["completed_tasks"] == 1
        assert status["failed_tasks"] == 1

    def test_unknown_batch_id(self):
        tracker = JobTracker()
        status = tracker.get_status("nonexistent")
        assert "error" in status

    def test_with_post_fn(self):
        def post_fn(meta, result):
            return {"custom": True, "status": "success", **meta}

        tracker = JobTracker()
        pending = [({"i": 0}, _make_done_future({"raw": 1}))]
        batch_id = tracker.register_batch("tool", pending, post_fn=post_fn)

        status = tracker.get_status(batch_id)
        assert status["status"] == "completed"


# ── JobTracker.get_results ─────────────────────────────────────────────


class TestGetResults:
    def test_returns_results_when_complete(self):
        tracker = JobTracker()
        pending = [
            ({"i": 0}, _make_done_future({"val": 10})),
            ({"i": 1}, _make_done_future({"val": 20})),
        ]
        batch_id = tracker.register_batch("tool", pending)

        result = tracker.get_results(batch_id)
        assert "results" in result
        assert len(result["results"]) == 2

    def test_blocks_when_pending_and_partial_false(self):
        tracker = JobTracker()
        pending = [
            ({"i": 0}, _make_done_future({"val": 10})),
            ({"i": 1}, _make_pending_future()),
        ]
        batch_id = tracker.register_batch("tool", pending)

        result = tracker.get_results(batch_id, include_partial=False)
        assert "results" not in result
        assert "message" in result
        assert "still pending" in result["message"]

    def test_returns_partial_when_requested(self):
        tracker = JobTracker()
        pending = [
            ({"i": 0}, _make_done_future({"val": 10})),
            ({"i": 1}, _make_pending_future()),
        ]
        batch_id = tracker.register_batch("tool", pending)

        result = tracker.get_results(batch_id, include_partial=True)
        assert "results" in result
        assert len(result["results"]) == 1

    def test_unknown_batch_id(self):
        tracker = JobTracker()
        result = tracker.get_results("nonexistent")
        assert "error" in result


# ── JobTracker.list_batches ────────────────────────────────────────────


class TestListBatches:
    def test_empty(self):
        tracker = JobTracker()
        assert tracker.list_batches() == []

    def test_multiple_batches(self):
        tracker = JobTracker()
        tracker.register_batch("tool_a", [({"x": 1}, _make_pending_future())])
        tracker.register_batch("tool_b", [({"x": 2}, _make_done_future(42))])

        batches = tracker.list_batches()
        assert len(batches) == 2
        tool_names = {b["tool_name"] for b in batches}
        assert tool_names == {"tool_a", "tool_b"}


# ── JobTracker.cancel_batch ────────────────────────────────────────────


class TestCancelBatch:
    def test_cancel_pending(self):
        tracker = JobTracker()
        fut = _make_pending_future()
        batch_id = tracker.register_batch("tool", [({"i": 0}, fut)])

        result = tracker.cancel_batch(batch_id)
        # Future.cancel() may or may not succeed depending on state,
        # but the call should not raise
        assert "batch_id" in result

    def test_cancel_already_done(self):
        tracker = JobTracker()
        fut = _make_done_future({"val": 1})
        batch_id = tracker.register_batch("tool", [({"i": 0}, fut)])

        result = tracker.cancel_batch(batch_id)
        assert result["already_done"] == 1

    def test_unknown_batch_id(self):
        tracker = JobTracker()
        result = tracker.cancel_batch("nonexistent")
        assert "error" in result


# ── JobTracker.cleanup ─────────────────────────────────────────────────


class TestCleanup:
    def test_removes_old_completed(self):
        tracker = JobTracker()
        batch_id = tracker.register_batch(
            "tool", [({"i": 0}, _make_done_future(1))]
        )

        # Force the submitted_at to be old
        batch = tracker._batches[batch_id]
        from datetime import timedelta

        batch.submitted_at -= timedelta(hours=25)

        removed = tracker.cleanup(max_age_hours=24)
        assert removed == 1
        assert tracker.list_batches() == []

    def test_keeps_recent(self):
        tracker = JobTracker()
        tracker.register_batch("tool", [({"i": 0}, _make_done_future(1))])

        removed = tracker.cleanup(max_age_hours=24)
        assert removed == 0
        assert len(tracker.list_batches()) == 1

    def test_keeps_pending(self):
        tracker = JobTracker()
        batch_id = tracker.register_batch(
            "tool", [({"i": 0}, _make_pending_future())]
        )

        batch = tracker._batches[batch_id]
        from datetime import timedelta

        batch.submitted_at -= timedelta(hours=25)

        removed = tracker.cleanup(max_age_hours=24)
        assert removed == 0


# ── gather_futures with timeout ────────────────────────────────────────


class TestGatherFuturesTimeout:
    def test_completes_within_timeout(self):
        pending = [
            ({"i": 0}, _make_done_future({"val": 1})),
            ({"i": 1}, _make_done_future({"val": 2})),
        ]
        results = asyncio.get_event_loop().run_until_complete(
            gather_futures(pending, timeout=5.0)
        )
        assert len(results) == 2

    def test_timeout_raises(self):
        pending = [({"i": 0}, _make_pending_future())]
        with pytest.raises(asyncio.TimeoutError):
            asyncio.get_event_loop().run_until_complete(
                gather_futures(pending, timeout=0.1)
            )

    def test_no_timeout_default(self):
        pending = [({"i": 0}, _make_done_future(42))]
        results = asyncio.get_event_loop().run_until_complete(
            gather_futures(pending)
        )
        assert len(results) == 1


# ── submit_or_gather ───────────────────────────────────────────────────


class TestSubmitOrGather:
    def test_sync_backend_returns_completed(self):
        backend = MagicMock()
        backend.is_async_remote = False

        tracker = JobTracker()
        pending = [({"i": 0}, _make_done_future({"val": 10}))]

        result = asyncio.get_event_loop().run_until_complete(
            submit_or_gather(backend, pending, tracker, "test_tool")
        )
        assert result["status"] == "completed"
        assert "results" in result
        assert len(result["results"]) == 1

    def test_async_backend_returns_submitted(self):
        backend = MagicMock()
        backend.is_async_remote = True

        tracker = JobTracker()
        pending = [({"i": 0}, _make_pending_future())]

        result = asyncio.get_event_loop().run_until_complete(
            submit_or_gather(backend, pending, tracker, "test_tool")
        )
        assert result["status"] == "submitted"
        assert "batch_id" in result
        assert result["n_tasks"] == 1
        assert "check_job_status" in result["message"]

    def test_async_backend_batch_trackable(self):
        backend = MagicMock()
        backend.is_async_remote = True

        tracker = JobTracker()
        fut = _make_done_future({"val": 99})
        pending = [({"i": 0}, fut)]

        result = asyncio.get_event_loop().run_until_complete(
            submit_or_gather(backend, pending, tracker, "test_tool")
        )
        batch_id = result["batch_id"]

        # Verify the batch is tracked and status works
        status = tracker.get_status(batch_id)
        assert status["status"] == "completed"

        # Verify results can be retrieved
        results = tracker.get_results(batch_id)
        assert "results" in results
        assert len(results["results"]) == 1

    def test_async_backend_with_post_fn(self):
        backend = MagicMock()
        backend.is_async_remote = True

        def post_fn(meta, result):
            return {"processed": True, "status": "success"}

        tracker = JobTracker()
        fut = _make_done_future({"raw": 1})
        pending = [({"i": 0}, fut)]

        result = asyncio.get_event_loop().run_until_complete(
            submit_or_gather(
                backend, pending, tracker, "test_tool", post_fn=post_fn,
            )
        )
        batch_id = result["batch_id"]

        results = tracker.get_results(batch_id)
        assert results["results"][0]["processed"] is True
