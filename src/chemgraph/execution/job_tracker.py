"""In-memory job tracker for async remote execution backends.

Tracks ``concurrent.futures.Future`` objects returned by
:meth:`ExecutionBackend.submit` so that MCP tools can return
immediately after submission and provide separate status / result
retrieval endpoints.

Each MCP server process creates its own ``JobTracker`` instance
(mirroring the existing ``backend = get_backend()`` pattern).
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrackedTask:
    """A single task within a tracked batch."""

    task_id: str
    meta: dict
    future: Future
    result: Optional[dict] = None


@dataclass
class TrackedBatch:
    """A group of tasks submitted together."""

    batch_id: str
    tool_name: str
    submitted_at: datetime
    tasks: list[TrackedTask] = field(default_factory=list)
    post_fn: Optional[Callable[[dict, Any], dict]] = None


class JobTracker:
    """Track submitted job batches and their futures.

    Thread-safe: all public methods acquire an internal lock.
    """

    def __init__(self) -> None:
        self._batches: dict[str, TrackedBatch] = {}
        self._lock = threading.Lock()

    # ── registration ───────────────────────────────────────────────────

    def register_batch(
        self,
        tool_name: str,
        pending_tasks: list[tuple[dict, Future]],
        post_fn: Optional[Callable[[dict, Any], dict]] = None,
    ) -> str:
        """Register a batch of submitted tasks and return a batch ID.

        Parameters
        ----------
        tool_name : str
            Name of the MCP tool that submitted the batch.
        pending_tasks : list[tuple[dict, Future]]
            Each element is ``(metadata_dict, future)``.
        post_fn : callable, optional
            Post-processing function applied when collecting results.
            Called as ``post_fn(metadata, raw_result) -> dict``.

        Returns
        -------
        str
            A UUID batch identifier.
        """
        batch_id = uuid.uuid4().hex[:12]
        tracked = [
            TrackedTask(
                task_id=meta.get("task_id", meta.get("structure", f"task_{i}")),
                meta=meta,
                future=fut,
            )
            for i, (meta, fut) in enumerate(pending_tasks)
        ]
        batch = TrackedBatch(
            batch_id=batch_id,
            tool_name=tool_name,
            submitted_at=datetime.now(timezone.utc),
            tasks=tracked,
            post_fn=post_fn,
        )
        with self._lock:
            self._batches[batch_id] = batch

        logger.info(
            "Registered batch '%s' (%s) with %d tasks",
            batch_id,
            tool_name,
            len(tracked),
        )
        return batch_id

    # ── status ─────────────────────────────────────────────────────────

    def get_status(self, batch_id: str) -> dict:
        """Return the current status of a batch.

        Returns
        -------
        dict
            Keys: ``batch_id``, ``tool_name``, ``submitted_at``,
            ``status``, ``total_tasks``, ``completed_tasks``,
            ``failed_tasks``, ``pending_tasks``, ``progress_pct``.
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                return {"error": f"Unknown batch_id: '{batch_id}'"}

        total = len(batch.tasks)
        done = 0
        failed = 0

        for t in batch.tasks:
            if t.future.done():
                done += 1
                # Cache the result on first check
                if t.result is None:
                    try:
                        raw = t.future.result(timeout=0)
                        if batch.post_fn is not None:
                            t.result = batch.post_fn(t.meta, raw)
                        elif isinstance(raw, dict):
                            merged = {**t.meta, **raw}
                            merged.setdefault("status", "success")
                            t.result = merged
                        else:
                            t.result = {
                                **t.meta,
                                "result": raw,
                                "status": "success",
                            }
                    except Exception as e:
                        t.result = {
                            **t.meta,
                            "status": "failure",
                            "error_type": type(e).__name__,
                            "message": str(e),
                        }
                if t.result.get("status") == "failure":
                    failed += 1

        pending = total - done
        if pending == total:
            status = "pending"
        elif pending > 0:
            status = "running"
        elif failed == total:
            status = "failed"
        elif failed > 0:
            status = "partial"
        else:
            status = "completed"

        return {
            "batch_id": batch_id,
            "tool_name": batch.tool_name,
            "submitted_at": batch.submitted_at.isoformat(),
            "status": status,
            "total_tasks": total,
            "completed_tasks": done - failed,
            "failed_tasks": failed,
            "pending_tasks": pending,
            "progress_pct": round(done / total * 100, 1) if total else 0.0,
        }

    # ── results ────────────────────────────────────────────────────────

    def get_results(
        self, batch_id: str, include_partial: bool = False
    ) -> dict:
        """Collect results from a batch.

        Parameters
        ----------
        batch_id : str
            The batch identifier.
        include_partial : bool
            If ``True``, return results for completed tasks even if some
            are still pending.  If ``False`` (default) and the batch is
            not fully resolved, return a status message instead.

        Returns
        -------
        dict
            Contains ``status``, ``results`` list, and summary counts.
        """
        status_info = self.get_status(batch_id)
        if "error" in status_info:
            return status_info

        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                return {"error": f"Unknown batch_id: '{batch_id}'"}

        if not include_partial and status_info["pending_tasks"] > 0:
            return {
                **status_info,
                "message": (
                    f"{status_info['pending_tasks']} of "
                    f"{status_info['total_tasks']} tasks still pending. "
                    f"Call check_job_status('{batch_id}') to monitor, "
                    f"or use include_partial=True to get partial results."
                ),
            }

        results = []
        for t in batch.tasks:
            if t.result is not None:
                results.append(t.result)

        return {
            **status_info,
            "results": results,
        }

    # ── listing ────────────────────────────────────────────────────────

    def list_batches(self) -> list[dict]:
        """Return a summary of all tracked batches."""
        with self._lock:
            batch_ids = list(self._batches.keys())

        summaries = []
        for bid in batch_ids:
            summaries.append(self.get_status(bid))
        return summaries

    # ── cancellation ───────────────────────────────────────────────────

    def cancel_batch(self, batch_id: str) -> dict:
        """Attempt to cancel pending tasks in a batch.

        Returns a dict with the number of successfully cancelled tasks.
        Note: ``Future.cancel()`` only succeeds if the task has not yet
        started executing.
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                return {"error": f"Unknown batch_id: '{batch_id}'"}

        cancelled = 0
        already_done = 0
        for t in batch.tasks:
            if t.future.done():
                already_done += 1
            elif t.future.cancel():
                cancelled += 1

        return {
            "batch_id": batch_id,
            "cancelled": cancelled,
            "already_done": already_done,
            "could_not_cancel": len(batch.tasks) - cancelled - already_done,
        }

    # ── cleanup ────────────────────────────────────────────────────────

    def cleanup(self, max_age_hours: float = 24) -> int:
        """Remove completed batches older than *max_age_hours*.

        Returns the number of batches removed.
        """
        now = datetime.now(timezone.utc)
        to_remove: list[str] = []

        with self._lock:
            for bid, batch in self._batches.items():
                age_hours = (now - batch.submitted_at).total_seconds() / 3600
                if age_hours > max_age_hours and all(
                    t.future.done() for t in batch.tasks
                ):
                    to_remove.append(bid)
            for bid in to_remove:
                del self._batches[bid]

        if to_remove:
            logger.info("Cleaned up %d old batches", len(to_remove))
        return len(to_remove)
