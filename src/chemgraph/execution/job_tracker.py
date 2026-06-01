"""In-memory job tracker for async remote execution backends.

Tracks ``concurrent.futures.Future`` objects returned by
:meth:`ExecutionBackend.submit` so that MCP tools can return
immediately after submission and provide separate status / result
retrieval endpoints.

Each MCP server process creates its own ``JobTracker`` instance
(mirroring the existing ``backend = get_backend()`` pattern).

When a *persist_file* is provided, batch metadata and Globus Compute
task UUIDs are written to a JSON file so that a future session can
reload them and query Globus Compute directly for results.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrackedTask:
    """A single task within a tracked batch."""

    task_id: str
    meta: dict
    future: Optional[Future] = None
    globus_task_id: Optional[str] = None
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

    Parameters
    ----------
    persist_file : Path or str, optional
        Path to a JSON file for persisting batch metadata across
        sessions.  When set, batches are saved after registration and
        after results are cached.  On init, existing batches are loaded.
    """

    def __init__(self, persist_file: Optional[Path | str] = None) -> None:
        self._batches: dict[str, TrackedBatch] = {}
        self._lock = threading.Lock()
        self._gc_lock = threading.Lock()
        self._persist_file = Path(persist_file) if persist_file else None
        self._gc_client = None  # lazily initialised Globus Compute Client

        if self._persist_file is not None:
            self._load()

    # ── Globus Compute client (lazy) ──────────────────────────────────

    def _get_gc_client(self):
        """Return a Globus Compute ``Client`` (created once, reused)."""
        if self._gc_client is not None:
            return self._gc_client
        with self._gc_lock:
            if self._gc_client is None:
                try:
                    from globus_compute_sdk import Client

                    self._gc_client = Client()
                except Exception:
                    logger.warning(
                        "Could not create Globus Compute Client",
                        exc_info=True,
                    )
                    return None
            return self._gc_client

    # ── persistence ───────────────────────────────────────────────────

    def _save(self) -> None:
        """Write current batch metadata to *persist_file*."""
        if self._persist_file is None:
            return

        data: dict[str, Any] = {}
        with self._lock:
            for bid, batch in self._batches.items():
                data[bid] = {
                    "tool_name": batch.tool_name,
                    "submitted_at": batch.submitted_at.isoformat(),
                    "tasks": [
                        {
                            "task_id": t.task_id,
                            "meta": t.meta,
                            "globus_task_id": t.globus_task_id,
                            "result": t.result,
                        }
                        for t in batch.tasks
                    ],
                }

        self._persist_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._persist_file.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self._persist_file)

    def _load(self) -> None:
        """Load batch metadata from *persist_file* (if it exists)."""
        if self._persist_file is None or not self._persist_file.is_file():
            return

        try:
            with open(self._persist_file) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load job tracker state: %s", exc)
            return

        orphaned: list[tuple[str, str]] = []  # (batch_id, task_id)
        with self._lock:
            for bid, info in data.items():
                if bid in self._batches:
                    continue  # don't overwrite live batches

                tasks = []
                for t in info.get("tasks", []):
                    tracked = TrackedTask(
                        task_id=t["task_id"],
                        meta=t.get("meta", {}),
                        future=None,
                        globus_task_id=t.get("globus_task_id"),
                        result=t.get("result"),
                    )
                    # Tasks loaded from disk with no globus_task_id and
                    # no cached result are orphaned -- get_status cannot
                    # query Globus for them (see line ~320).
                    if tracked.globus_task_id is None and tracked.result is None:
                        orphaned.append((bid, tracked.task_id))
                    tasks.append(tracked)

                self._batches[bid] = TrackedBatch(
                    batch_id=bid,
                    tool_name=info["tool_name"],
                    submitted_at=datetime.fromisoformat(info["submitted_at"]),
                    tasks=tasks,
                )

        logger.info(
            "Loaded %d batches from %s", len(data), self._persist_file
        )
        if orphaned:
            logger.warning(
                "%d task(s) reloaded without a Globus task_id -- their "
                "results cannot be recovered. Examples: %s",
                len(orphaned),
                ", ".join(f"{b}/{t}" for b, t in orphaned[:5]),
            )

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

        # Wait briefly for the Executor background thread to set task_ids
        # on the ComputeFutures.  Typically takes ~1-2 s; we cap at 3 s
        # so the MCP tool response isn't delayed excessively.
        self._wait_for_globus_task_ids(tracked, timeout=3.0)
        self._save()
        return batch_id

    def _wait_for_globus_task_ids(
        self, tasks: list[TrackedTask], timeout: float = 3.0
    ) -> None:
        """Wait up to *timeout* seconds for Globus ``task_id`` to appear
        on each ComputeFuture, then store them for persistence."""
        deadline = time.monotonic() + timeout
        pending = [t for t in tasks if t.future is not None and t.globus_task_id is None]

        while pending and time.monotonic() < deadline:
            still_pending = []
            for t in pending:
                gc_id = getattr(t.future, "task_id", None)
                if gc_id is not None:
                    t.globus_task_id = str(gc_id)
                else:
                    still_pending.append(t)
            pending = still_pending
            if pending:
                time.sleep(0.25)

        if pending:
            # Promoted from debug -> warning: tasks without a task_id
            # at this point will be lost across a server restart, so the
            # user should see this immediately rather than only in the
            # post-mortem orphan warning at reload time.
            logger.warning(
                "%d task(s) did not receive a Globus task_id within %.1fs; "
                "they will be unrecoverable if the server restarts before "
                "the next get_status call",
                len(pending),
                timeout,
            )

    def _try_capture_globus_task_ids(self, tasks: list[TrackedTask]) -> bool:
        """Non-blocking: extract ``task_id`` from any ComputeFuture that
        has one available.  Returns True if any new IDs were captured."""
        captured = False
        for t in tasks:
            if t.globus_task_id is None and t.future is not None:
                gc_id = getattr(t.future, "task_id", None)
                if gc_id is not None:
                    t.globus_task_id = str(gc_id)
                    captured = True
        return captured

    # ── status ─────────────────────────────────────────────────────────

    def get_status(self, batch_id: str) -> dict:
        """Return the current status of a batch.

        For tasks loaded from disk (no in-memory ``Future``), queries
        Globus Compute directly if a ``globus_task_id`` is available.

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
        # Lazily capture Globus Compute task UUIDs (set asynchronously
        # by the Executor background thread after submission).
        dirty = self._try_capture_globus_task_ids(batch.tasks)

        for t in batch.tasks:
            task_done = False

            # --- live future path ---
            if t.future is not None and t.future.done():
                task_done = True
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
                    dirty = True

            # --- loaded-from-disk path (no future, use Globus client) ---
            elif t.future is None and t.result is None and t.globus_task_id:
                gc = self._get_gc_client()
                if gc is not None:
                    try:
                        task_info = gc.get_task(t.globus_task_id)
                        if not task_info.get("pending", True):
                            task_done = True
                            if "result" in task_info:
                                raw = task_info["result"]
                                if isinstance(raw, dict):
                                    merged = {**t.meta, **raw}
                                    merged.setdefault("status", "success")
                                    t.result = merged
                                else:
                                    t.result = {
                                        **t.meta,
                                        "result": raw,
                                        "status": "success",
                                    }
                            elif "exception" in task_info:
                                t.result = {
                                    **t.meta,
                                    "status": "failure",
                                    "error_type": "RemoteException",
                                    "message": str(task_info["exception"]),
                                }
                            dirty = True
                    except Exception as e:
                        logger.warning(
                            "Failed to query Globus task %s: %s",
                            t.globus_task_id,
                            e,
                            exc_info=True,
                        )

            # --- already have a cached result ---
            elif t.result is not None:
                task_done = True

            if task_done:
                done += 1
            if t.result is not None and t.result.get("status") == "failure":
                failed += 1

        if dirty:
            self._save()

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
            if t.future is None:
                already_done += 1
            elif t.future.done():
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
                all_done = all(
                    (t.future is not None and t.future.done())
                    or t.result is not None
                    for t in batch.tasks
                )
                if age_hours > max_age_hours and all_done:
                    to_remove.append(bid)
            for bid in to_remove:
                del self._batches[bid]

        if to_remove:
            logger.info("Cleaned up %d old batches", len(to_remove))
            self._save()
        return len(to_remove)
