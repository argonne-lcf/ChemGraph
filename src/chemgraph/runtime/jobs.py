"""Job-aware extension of the existing scoped operation journal."""

from __future__ import annotations

import asyncio
import glob
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from langchain_core.messages import ToolMessage

from chemgraph.runtime.artifacts import (
    INTERNAL_DIRS, JOBS_DIR, artifact_file, capture_artifacts, checked_path,
    internal_path, publish_file, sync_directory, verify_file,
)
from chemgraph.runtime.context import JobPending, ScopedExecutionError
from chemgraph.runtime.journal import (
    ABANDONED, APPROVED, ARTIFACT_KEY, BLOCKED, FINALIZED, INVOKING, PROPOSED,
    REJECTED, RETURNED, UNCERTAIN, UNFINISHED, ApprovalRequired, OperationJournal,
    call_digest, encode_result, error_message, operation_id,
)
from chemgraph.runtime.policy import PreparedAction, digest, validate

SUBMITTING, SUBMITTED = "submitting", "submitted"


def job_context(context, action):
    execution = action.execution
    expected = f"{JOBS_DIR}/{action.operation_id}/{execution['attempt']}"
    if execution["root"] != expected:
        raise ValueError("Invalid prepared job root.")
    root = internal_path(context.workspace, JOBS_DIR, action.operation_id, str(execution["attempt"]))
    return replace(context, workspace=root)


def _policy_data(policy):
    data = asdict(policy)
    for key in ("tools", "calculators", "helpers"):
        data[key] = sorted(data[key])
    data["model_roots"] = list(data["model_roots"])
    return data


def _logical_path(workspace, relative):
    path = Path(relative)
    if path.is_absolute() or INTERNAL_DIRS.intersection(path.parts):
        raise ValueError("A job output must be an ordinary workspace-relative path.")
    return checked_path(workspace, workspace / path)


def stage_promotions(context, artifacts):
    staged = []
    for artifact in artifacts:
        destination = _logical_path(context.workspace, artifact["source_path"])
        source = artifact_file(context.workspace, artifact)
        verify_file(source, artifact)
        # Staging is private; the final replace is done under the workspace guard.
        temporary = internal_path(context.workspace, JOBS_DIR, "promotion",
                                  artifact["sha256"], artifact["source_path"])
        publish_file(context.workspace, source, temporary, artifact)
        staged.append((temporary, destination, artifact))
    return staged


def promote_outputs(context, staged, cleanup):
    for relative in cleanup:
        path = _logical_path(context.workspace, relative)
        path.unlink(missing_ok=True)
        sync_directory(path.parent)
    for temporary, destination, artifact in staged:
        destination = _logical_path(context.workspace, artifact["source_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        checked_path(context.workspace, destination)
        if destination.exists():
            try:
                verify_file(destination, artifact)
                continue
            except ValueError:
                pass  # Logical paths intentionally have synchronous overwrite semantics.
        # A retry stages again from the immutable artifact before entering the guard.
        os.replace(temporary, destination)
        for directory in destination.parents:
            sync_directory(directory)
            if directory == context.workspace:
                break


class JobJournal(OperationJournal):
    """Use the same operation records for reviews, submissions and final results."""

    unfinished_states = UNFINISHED | {SUBMITTING}

    def __init__(self, store, backend):
        super().__init__(store)
        self.backend = backend

    def prepare_action(self, context, call, record):
        if call["name"] != "run_ase":
            return super().prepare_action(context, call, record)
        previous = record.get("previous_prepared")
        action = (PreparedAction.from_dict(previous) if previous
                  else super().prepare_action(context, call, record))
        attempt = record.get("attempt", 0)
        execution = {"root": f"{JOBS_DIR}/{action.operation_id}/{attempt}",
                     "attempt": attempt, "backend": self.backend.identity,
                     "job_id": f"{action.operation_id}-{attempt:x}"}
        action = replace(action, version=2, execution=execution, digest="")
        worker = job_context(context, action)
        for item in action.inputs:
            source = checked_path(context.workspace, context.workspace / item["path"])
            verify_file(source, item)
            destination = checked_path(worker.workspace, worker.workspace / item["path"])
            publish_file(worker.workspace, source, destination, item)
        content = action.to_dict()
        content.pop("digest")
        return replace(action, digest=digest(content))

    def validate_job(self, context, record):
        action = PreparedAction.from_dict(record["prepared"])
        if action.execution["backend"] != self.backend.identity:
            raise ValueError("The prepared job backend changed.")
        call = {"id": record["tool_call_id"], "name": record["tool"],
                "args": action.requested_arguments}
        validate(job_context(context, action), call, action)
        return action, call

    def _authorized_job(self, record):
        # SUBMITTED keeps the exact reviewed decision while the local queue waits.
        return self.authorizer({**record, "state": APPROVED})

    async def execute(self, context, call, invoke):
        identifier = operation_id(context, call["id"])
        record = await self.get(identifier)
        if not record or not (record.get("prepared") or {}).get("execution"):
            return await super().execute(context, call, invoke)
        if record["call_digest"] != call_digest(call):
            raise ScopedExecutionError("Tool arguments changed after review.")
        if record["state"] in {FINALIZED, ABANDONED}:
            return await super().execute(context, call, invoke)
        if record["state"] in {UNCERTAIN, SUBMITTING}:
            raise ScopedExecutionError("The job outcome is uncertain; reconcile it explicitly.")
        others = await self.store.records(thread_id=context.thread_id)
        if any(r["id"] != identifier and r["state"] in self.unfinished_states for r in others):
            raise ScopedExecutionError("Reconcile unfinished operations before submitting work.")
        if record["state"] == APPROVED:
            if not self.authorized(record):
                raise ApprovalRequired("A current decision is required.")
            try:
                action, call = await asyncio.to_thread(self.validate_job, context, record)
            except (ValueError, PermissionError, OSError) as exc:
                await self._put(record, BLOCKED, error=str(exc))
                return error_message(call, f"Not executed: {exc}")
            if not self.authorized(record):
                raise ApprovalRequired("The approval expired before submission.")
            job_id = action.execution["job_id"]
            spec = {
                "job_id": job_id, "operation_id": identifier, "owner": context.owner,
                "thread_id": context.thread_id, "task_id": context.task_id,
                "workspace": str(job_context(context, action).workspace),
                "policy": _policy_data(context.policy), "action": action.to_dict(), "call": call,
            }
            await self._put(record, SUBMITTING, job_id=job_id)
            try:
                await self.backend.submit(job_id, spec)
            except Exception:
                await self._put(record, UNCERTAIN)
                raise
            await self._put(record, SUBMITTED)
        if record["state"] != SUBMITTED:
            raise ScopedExecutionError(f"Job cannot execute from {record['state']}.")
        raise JobPending(identifier, record["job_id"])

    async def dispatchable(self, context, record):
        if record["state"] != SUBMITTED:
            return False
        if any(r["state"] in self.unfinished_states
               for r in await self.store.records(thread_id=context.thread_id)):
            return False
        try:
            if not self._authorized_job(record):
                raise ValueError("The queued job's approval expired before dispatch.")
            await asyncio.to_thread(self.validate_job, context, record)
        except (ValueError, PermissionError, OSError) as exc:
            await self.backend.cancel(record["job_id"])
            record["dispatch_error"] = str(exc)
            await self._put(record, SUBMITTED)
            return False
        return True

    async def advance(self, context, *, cancelled=False):
        """Called under the service advancement barrier, never inside a graph step."""
        for record in await self.store.records(thread_id=context.thread_id):
            if record["state"] == FINALIZED and record.get("job_id"):
                job = await self.backend.lookup(record["job_id"])
                if job is not None and not job["published"]:
                    # Finalization and the backend acknowledgment are separate
                    # durable writes. Recover this gap without republishing files.
                    await self.backend.acknowledge(record["job_id"], promoted=record["result"]["promoted"])
                continue
            if record["state"] not in {SUBMITTING, SUBMITTED, UNCERTAIN, RETURNED} or not record.get("job_id"):
                continue
            if record["state"] != RETURNED:
                job = await self.backend.lookup(record["job_id"])
                if job is None:
                    if record["state"] != UNCERTAIN:
                        await self._put(record, UNCERTAIN)
                    continue
                job = await self.backend.status(record["job_id"])
                state = job["status"]
                if state == "unknown":
                    if record["state"] != UNCERTAIN:
                        await self._put(record, UNCERTAIN)
                    await self.backend.acknowledge(record["job_id"])
                    continue
                if state in {"queued", "running", "dispatching"}:
                    if record["state"] == SUBMITTING:
                        await self._put(record, SUBMITTED)
                    continue
                if state == "cancelled":
                    envelope = {"result": {"status": "failure",
                                "message": record.get("dispatch_error", "Job cancelled before execution.")},
                                "outputs": [], "cleanup": []}
                else:
                    envelope = await self.backend.fetch(record["job_id"])
                await self._put(record, RETURNED, returned_job=envelope, promotion=None)
            await self.finalize_job(context, record, cancelled=cancelled)

    async def finalize_job(self, context, record, *, cancelled=False):
        action = PreparedAction.from_dict(record["prepared"])
        root = job_context(context, action).workspace
        envelope = record["returned_job"]
        manifest = envelope["outputs"]
        for output in manifest:
            _logical_path(context.workspace, output["source_path"])
        artifacts = await asyncio.to_thread(capture_artifacts, context, record["tool_call_id"],
                                           manifest, source_root=root)
        if record["promotion"] is None:
            async with context.lock:
                destinations = {o["source_path"] for o in manifest}
                cleanup = set()
                if not cancelled:
                    for pattern in envelope["cleanup"]:
                        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                            raise ValueError("Invalid calculator cleanup intent.")
                        for path in glob.glob(os.path.join(glob.escape(str(context.workspace)), pattern)):
                            relative = Path(path).relative_to(context.workspace).as_posix()
                            _logical_path(context.workspace, relative)
                            cleanup.add(relative)
                await self._put(record, RETURNED, promotion={
                    "cleanup": sorted(cleanup - destinations), "promoted": not cancelled,
                    "artifacts": artifacts,
                })
        # Once cancelled, a retry can never promote a previously staged result.
        promoted = record["promotion"]["promoted"] and not cancelled and not context.cancelled
        if promoted:
            staged = await asyncio.to_thread(stage_promotions, context, artifacts)
            async with context.lock:
                promoted = not context.cancelled
                if promoted:
                    await asyncio.to_thread(promote_outputs, context, staged, record["promotion"]["cleanup"])
        result = dict(envelope["result"])
        paths = {a["source_path"]: a["source_path"] if promoted else a["path"] for a in artifacts}
        for key in ("results_file", "trajectory_file"):
            if result.get(key):
                path = Path(result[key])
                relative = path.relative_to(root).as_posix() if path.is_absolute() else str(path)
                result[key] = paths.get(relative, relative if promoted else None)
        result["message"] = (f"ASE {action.arguments['params']['driver']}: {result['status']}. "
                             f"Outputs: {', '.join(paths.values()) or 'none'}.")
        result["promoted"] = promoted
        if envelope["result"].get("status") == "failure":
            error = envelope["result"].get("message", "Calculation failed.")
            for relative, public in sorted(paths.items(), key=lambda item: len(item[0]), reverse=True):
                error = error.replace(str(root / relative), public)
            for item in action.inputs:
                error = error.replace(str(root / item["path"]), item["source"])
            result["error"] = error.replace(str(root), "<calculation workspace>")
        message = ToolMessage(content=json.dumps(result), name="run_ase",
                              tool_call_id=record["tool_call_id"],
                              status="success" if result["status"] == "success" else "error",
                              additional_kwargs={ARTIFACT_KEY: artifacts})
        await self._put(record, FINALIZED, result={
            "message": encode_result(message), "artifacts": artifacts, "promoted": promoted,
        })
        await self.backend.acknowledge(record["job_id"], promoted=promoted)

    async def confirm_nonexecution(self, identifier, *, decided_by, evidence):
        record = await self.get(identifier)
        if not evidence.strip() or not record or record["state"] != UNCERTAIN:
            raise ValueError("Nonexecution requires an uncertain operation and explicit evidence.")
        if not record.get("job_id"):
            return await super().confirm_nonexecution(identifier, decided_by=decided_by, evidence=evidence)
        if await self.backend.lookup(record["job_id"]) is not None:
            if await self.backend.is_live(record["job_id"]):
                raise ValueError("The attempt is still live; nonexecution cannot be confirmed.")
            if (await self.backend.status(record["job_id"]))["status"] in {"succeeded", "failed"}:
                raise ValueError("The attempt has a completion envelope; recover it instead.")
            await self.backend.cancel(record["job_id"])
            await self.backend.acknowledge(record["job_id"], promoted=False)
        record["history"].append({"event": "nonexecution", "evidence": evidence, "decided_by": decided_by})
        return await self._put(record, PROPOSED, previous_prepared=record["prepared"],
                               prepared=None, decision=None, attempt=record.get("attempt", 0) + 1,
                               job_id=None)

    async def abandon(self, identifier, *, decided_by, reason):
        record = await self.get(identifier)
        if (record and record["state"] not in {FINALIZED, ABANDONED, REJECTED, BLOCKED}
                and record.get("job_id") and await self.backend.lookup(record["job_id"]) is not None):
            await self.backend.cancel(record["job_id"])
            await self.backend.acknowledge(record["job_id"], promoted=False)
        return await super().abandon(identifier, decided_by=decided_by, reason=reason)

    async def mark_interrupted(self, thread_id):
        for record in await self.store.records(thread_id=thread_id):
            if record["state"] == INVOKING:
                await self._put(record, UNCERTAIN)
