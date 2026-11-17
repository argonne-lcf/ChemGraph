"""Durable local calculation queue; worker receipts do not depend on Futures."""

from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Protocol

from chemgraph.execution.base import TaskSpec
from chemgraph.execution.local_backend import LocalBackend
from chemgraph.runtime.artifacts import checked_path, output_manifest, sync_directory, verify_file
from chemgraph.runtime.policy import digest


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def read_json(path):
    return json.loads(Path(path).read_text())


class JobBackend(Protocol):
    """Calculation lifecycle. Submission acknowledges durable intent, not completion."""

    identity: str

    async def submit(self, job_id: str, spec: dict) -> dict: ...
    async def lookup(self, job_id: str) -> dict | None: ...
    async def status(self, job_id: str) -> dict: ...
    async def fetch(self, job_id: str) -> dict: ...
    async def cancel(self, job_id: str) -> dict: ...


def run_local_job(spec, receipt_directory):
    """Rebuild scope in a spawned child; write results before returning to the pool."""
    from chemgraph.runtime.context import RunContext, invocation, run_context
    from chemgraph.runtime.policy import ExecutionPolicy, PreparedAction, validate
    from chemgraph.schemas.ase_input import ASEInputSchema
    from chemgraph.tools.ase_core import run_ase_core
    from chemgraph.utils.artifacts import artifact_context

    receipt = Path(receipt_directory)
    identity = {"job_id": spec["job_id"], "fingerprint": spec["action"]["digest"]}
    atomic_json(receipt / "started.json", {**identity, "pid": os.getpid(), "at": time.time()})
    outputs, cleanup = set(), []
    try:
        action = PreparedAction.from_dict(spec["action"])
        context = RunContext(spec["owner"], spec["thread_id"], spec["task_id"],
                             Path(spec["workspace"]), ExecutionPolicy(**spec["policy"]), None)
        try:
            with run_context(context), invocation(action, outputs, cleanup), artifact_context(str(context.workspace)):
                validate(context, spec["call"], action)
                result = run_ase_core(ASEInputSchema.model_validate(action.arguments["params"]))
            manifest = output_manifest(context.workspace, outputs)
        except Exception as exc:
            result = {"status": "failure", "error_type": type(exc).__name__, "message": str(exc)}
            manifest, cleanup = [], []
        atomic_json(receipt / "result.json", {
            **identity, "result": result, "outputs": manifest, "cleanup": cleanup,
        })
    finally:
        atomic_json(receipt / "exited.json", {**identity, "pid": os.getpid(), "at": time.time()})


class LocalJobBackend:
    """Single-process FIFO with conservative recovery of lost worker ownership."""

    identity = "local-ase-jobs-v1"

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._pool = None
        self._futures = {}
        self._active = None
        self._broken = False

    def _directory(self, job_id):
        if not job_id or any(c not in "0123456789abcdef-" for c in job_id):
            raise ValueError("Invalid job identifier.")
        return checked_path(self.directory, self.directory / job_id)

    async def lookup(self, job_id):
        path = self._directory(job_id) / "job.json"
        return read_json(path) if path.exists() else None

    async def records(self):
        return sorted([read_json(p) for p in self.directory.glob("*/job.json")],
                      key=lambda r: (r["sequence"], r["job_id"]))

    async def _save(self, record):
        atomic_json(self._directory(record["job_id"]) / "job.json", record)
        return record

    async def submit(self, job_id, spec):
        existing = await self.lookup(job_id)
        fingerprint = digest(spec)
        if existing:
            if existing["fingerprint"] != fingerprint:
                raise ValueError("Job identity was reused with different inputs.")
            return existing
        records = await self.records()
        return await self._save({
            "job_id": job_id, "spec": spec, "fingerprint": fingerprint, "status": "queued",
            "sequence": max((r["sequence"] for r in records), default=0) + 1,
            "cancel_requested": False, "published": False,
        })

    async def fetch(self, job_id):
        record = await self.lookup(job_id)
        if record is None:
            raise KeyError(job_id)
        return await asyncio.to_thread(self._fetch, record)

    def _fetch(self, record):
        job_id = record["job_id"]
        envelope = read_json(self._directory(job_id) / "result.json")
        if (envelope["job_id"] != job_id
                or envelope["fingerprint"] != record["spec"]["action"]["digest"]):
            raise ValueError("Completion envelope does not match the submitted attempt.")
        root = Path(record["spec"]["workspace"])
        for output in envelope["outputs"]:
            verify_file(checked_path(root, root / output["source_path"]), output)
        return envelope

    async def status(self, job_id):
        record = await self.lookup(job_id)
        if record is None:
            raise KeyError(job_id)
        future = self._futures.get(job_id)
        if future is not None and future.done() and not future.cancelled():
            try:
                future.result()
            except BrokenProcessPool:
                self._broken = True
            except Exception:
                pass  # Only a valid envelope proves a calculation outcome.
        receipt = self._directory(job_id) / "result.json"
        if receipt.exists():
            try:
                # Keep the metadata read/modify/write atomic with respect to other
                # coroutines (notably cancellation and publication acknowledgment).
                envelope = read_json(receipt)
                if (envelope["job_id"] != job_id
                        or envelope["fingerprint"] != record["spec"]["action"]["digest"]):
                    raise ValueError("Mismatched attempt envelope.")
                record["status"] = "succeeded" if envelope["result"]["status"] == "success" else "failed"
            except (ValueError, KeyError, OSError) as exc:
                record.update(status="unknown", error=str(exc))
        elif record["status"] in {"dispatching", "running"}:
            if future is None or future.done():
                record["status"] = "unknown"
            else:
                record["status"] = "running"
        return await self._save(record)

    async def dispatch(self, job_id):
        record = await self.status(job_id)
        if record["status"] != "queued" or record["published"] or record["cancel_requested"]:
            return record
        if self._active is not None:
            previous = await self.status(self._active)
            if previous["published"] or previous["status"] in {"unknown", "cancelled"}:
                await self.acknowledge(self._active)
            if self._active is not None:
                return record
        # On reopen a completed but unpublished attempt still owns the queue slot.
        for previous in await self.records():
            if previous["job_id"] == job_id or previous["published"]:
                continue
            state = (await self.status(previous["job_id"]))["status"]
            if state in {"dispatching", "running", "succeeded", "failed"}:
                return record
        if self._broken and self._pool is not None:
            await asyncio.to_thread(self._pool.shutdown)
            self._pool = None
            self._broken = False
        if self._pool is None:
            self._pool = LocalBackend()
            self._pool.initialize(max_workers=1, max_tasks_per_child=1)
        # A crash after this fsync cannot make the attempt eligible for dispatch again.
        record.update(status="dispatching", dispatching_at=time.time())
        await self._save(record)
        self._active = job_id
        try:
            self._futures[job_id] = self._pool.submit(TaskSpec(
                task_id=job_id, callable=run_local_job,
                args=(record["spec"], str(self._directory(job_id))),
            ))
        except Exception as exc:
            record.update(status="unknown", error=str(exc))
            self._broken = True
            await self._save(record)
        return record

    async def acknowledge(self, job_id, *, promoted=None):
        record = await self.status(job_id)
        if promoted is not None:
            record.update(published=True, promoted=promoted)
            await self._save(record)
        future = self._futures.get(job_id)
        if self._active == job_id and (future is None or future.done()):
            self._active = None

    async def cancel(self, job_id):
        record = await self.status(job_id)
        record["cancel_requested"] = True
        future = self._futures.get(job_id)
        if record["status"] == "queued" or (future is not None and future.cancel()):
            record["status"] = "cancelled"
        return await self._save(record)

    async def is_live(self, job_id):
        future = self._futures.get(job_id)
        if future is not None:
            return not future.done()
        root = self._directory(job_id)
        if (root / "exited.json").exists():
            return False
        if (root / "started.json").exists():
            if os.name != "posix":
                # os.kill(pid, 0) is not a harmless existence probe on Windows.
                # Without an exit receipt, retain the conservative live verdict.
                return True
            try:
                os.kill(read_json(root / "started.json")["pid"], 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            return True
        return False  # Explicit operator evidence is still required for uncertainty.

    async def close(self):
        if self._pool is not None:
            await asyncio.to_thread(self._pool.shutdown)
            self._pool = None
