"""Durable acceptance, serialized graph execution, and conservative recovery."""

import asyncio
import hashlib
import json
import logging
import time
from uuid import uuid4

from a2a.server.tasks.database_task_store import DatabaseTaskStore
from a2a.types import a2a_pb2 as a2a
from a2a.utils.errors import (
    InternalError,
    InvalidParamsError,
    TaskNotCancelableError,
    TaskNotFoundError,
)
from sqlalchemy.ext.asyncio import create_async_engine

from .adapters import CheckpointReviewRequired, ChemGraphAdapter, Outcome
from dataclasses import asdict
from .protocol import TERMINAL, WAITING, context, task_from, to_dict, transition
from .store import Store

logger = logging.getLogger(__name__)


class Service:
    def __init__(self, settings, adapter=None):
        self.settings = settings
        self.lock = asyncio.Lock()
        self.wake = asyncio.Event()
        self.changed = asyncio.Condition()
        self.adapter = adapter or ChemGraphAdapter(settings)
        self.store = Store(settings.state_dir / "service.db")
        from .operations import Operations

        self.operations = Operations(self, getattr(self.adapter, "backend", None))
        self.adapter.operations = self.operations
        self.running = False
        self.ready = False
        self.worker = None
        self.monitor = None

    async def start(self):
        import fcntl

        self.settings.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if (self.settings.state_dir / "connect.db").exists():
            raise RuntimeError("Use fresh service state; retain standalone Connect state separately for inspection")
        self.lease = (self.settings.state_dir / "service.lock").open("a")
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lease.close()
            raise RuntimeError("Only one ChemGraph service instance may use this state directory") from None
        try:
            await self.store.open()
            profile_dir = self.settings.state_dir / "profiles"
            profile_dir.mkdir(exist_ok=True)
            profile_path = profile_dir / (self.settings.profile.identity + ".json")
            profile_text = self.settings.profile.model_dump_json(indent=2)
            if profile_path.exists() and profile_path.read_text() != profile_text:
                raise RuntimeError("Saved execution profile does not match its identity")
            profile_path.write_text(profile_text)
            self.engine = create_async_engine(
                f"sqlite+aiosqlite:///{self.settings.state_dir / 'a2a.db'}"
            )
            self.tasks = DatabaseTaskStore(self.engine)
            await self.tasks.initialize()
            await self.adapter.start()
            await self.operations.recover()
            async with self.lock:
                for record in await self.store.records():
                    await self.project(record)
                # A delivery may have succeeded just before the notifier/service died.
                async with self.store.transaction():
                    await self.store.db.execute(
                        "UPDATE outbox SET state='ambiguous' WHERE state='delivering'"
                    )
            for record in await self.store.records():
                if record.get("pending_outcome"):
                    await self.finish(record, Outcome(**record["pending_outcome"]))
                    continue
                operations = await self.operations.list(record["id"])
                if any(op["state"] == "uncertain" for op in operations):
                    await self.finish(
                        record,
                        Outcome(
                            state="input_required",
                            reason="reconciliation",
                            text="A computational submission needs operator reconciliation.",
                            interrupts=record.get("interrupts", []),
                        ),
                    )
                    continue
                if any(op["state"] == "returned" for op in operations):
                    record.update(phase="queued", action="recover")
                    transition(record, a2a.TASK_STATE_WORKING, "Finalizing saved execution results")
                    async with self.store.transaction():
                        await self.store.put(record)
                    await self.project(record)
                    continue
                if record["phase"] not in {"queued", "preparing", "executing", "backend"}:
                    continue
                if record["profile"] != self.settings.profile.identity:
                    await self.finish(
                        record,
                        Outcome(
                            state="input_required",
                            reason="configuration",
                            text="Execution profile changed; restore the original profile before continuing.",
                        ),
                    )
                elif record["phase"] == "executing":
                    try:
                        outcome = await self.adapter.recover(record)
                    except Exception:
                        logger.exception("Checkpoint recovery failed for %s", record["id"])
                        outcome = Outcome(
                            state="input_required",
                            reason="reconciliation",
                            text="Checkpoint recovery requires operator review.",
                        )
                    await self.finish(record, outcome)
                elif record["phase"] == "preparing":
                    record["phase"] = "queued"
                    async with self.store.transaction():
                        await self.store.put(record)
            self.running = self.ready = True
            self.worker = asyncio.create_task(self.work(), name="chemgraph-runner")
            self.monitor = asyncio.create_task(self.monitor_jobs(), name="backend-monitor")
            self.worker.add_done_callback(self.background_done)
            self.monitor.add_done_callback(self.background_done)
            self.wake.set()
        except BaseException:
            await self.close()
            raise

    async def close(self):
        self.ready = self.running = False
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        if self.monitor:
            self.monitor.cancel()
            await asyncio.gather(self.monitor, return_exceptions=True)
        # A canceled coroutine is not confirmation of scientific cancellation.
        try:
            await self.adapter.close()
        finally:
            await self.store.close()
            if hasattr(self, "engine"):
                await self.engine.dispose()
            if hasattr(self, "lease") and not self.lease.closed:
                self.lease.close()

    def background_done(self, task):
        if not task.cancelled() and task.exception():
            self.ready = False
            logger.error("Background service failed", exc_info=task.exception())

    async def project(self, record):
        await self.tasks.save(task_from(record), context(record["owner"]))
        task = task_from(record)
        if task.status.state in WAITING:
            wait_identity = hashlib.sha256(
                json.dumps(record.get("interrupts", []), sort_keys=True).encode()
            ).hexdigest()[:16]
            key = f"{task.id}:{record['generation']}:{task.status.state}:{record.get('reason', '')}:{wait_identity}"
            payload = {
                "id": key,
                "task_id": task.id,
                "owner": record["owner"],
                "state": a2a.TaskState.Name(task.status.state),
                "reason": record.get("reason", ""),
                "created_at": task.status.timestamp.ToNanoseconds() / 1e9,
            }
            async with self.store.transaction():
                await self.store.db.execute(
                    "INSERT OR IGNORE INTO outbox(id,task_id,state,payload) "
                    "VALUES(?,?,'pending',?)",
                    (key, task.id, json.dumps(payload)),
                )
        async with self.changed:
            self.changed.notify_all()

    async def owned(self, task_id, owner):
        record = await self.store.get(task_id)
        if not record or record["owner"] != owner:
            raise TaskNotFoundError()
        return record

    async def get(self, task_id, owner):
        await self.owned(task_id, owner)
        return await self.tasks.get(task_id, context(owner))

    async def accept(self, request, owner):
        if not self.ready:
            raise InternalError("Service is not ready to accept scientific work")
        message = request.message
        if message.role != a2a.ROLE_USER or not message.message_id or not message.parts:
            raise InvalidParamsError("A user message with messageId and text is required")
        if any(part.WhichOneof("content") != "text" for part in message.parts):
            raise InvalidParamsError("Only text input is supported")
        text = "\n".join(p.text for p in message.parts)
        if not text.strip() or len(text) > 100_000:
            raise InvalidParamsError("Text must be nonempty and at most 100000 characters")
        if request.configuration.HasField("task_push_notification_config"):
            raise InvalidParamsError("Push subscriptions are disabled")
        if (
            request.configuration.accepted_output_modes
            and "text/plain" not in request.configuration.accepted_output_modes
        ):
            raise InvalidParamsError("text/plain output must be accepted")
        digest = hashlib.sha256(message.SerializeToString(deterministic=True)).hexdigest()
        async with self.lock:
            duplicate = await self.store.one(
                "SELECT * FROM messages WHERE owner=? AND message_id=?", (owner, message.message_id)
            )
            if duplicate:
                if duplicate["digest"] != digest:
                    raise InvalidParamsError("messageId was already used with different content")
                return await self.get(duplicate["task_id"], owner)
            record = None
            context_id = message.context_id or str(uuid4())
            if message.task_id:
                record = await self.owned(message.task_id, owner)
                if message.context_id and message.context_id != record["context_id"]:
                    raise InvalidParamsError("taskId and contextId disagree")
                context_id = record["context_id"]
            active = [
                r
                for r in await self.store.records()
                if r["owner"] == owner
                and r["context_id"] == context_id
                and task_from(r).status.state not in TERMINAL
            ]
            if active:
                if record and record["id"] != active[0]["id"]:
                    raise InvalidParamsError("Context already has an active task")
                record = active[0]
            if record:
                if task_from(record).status.state != a2a.TASK_STATE_INPUT_REQUIRED:
                    raise InvalidParamsError(
                        "Only an interrupted task accepts a taskId continuation"
                    )
                if record.get("reason") != "clarification":
                    raise InvalidParamsError("This task requires operator action, not a peer reply")
                if record["profile"] != self.settings.profile.identity:
                    raise InvalidParamsError("Execution profile is incompatible")
                interrupts = record.get("interrupts", [])
                metadata = to_dict(message.metadata)
                interrupt_id = metadata.get("interrupt_id")
                responses = metadata.get("interrupt_responses")
                if responses is not None:
                    expected = {item["id"] for item in interrupts}
                    if (
                        not isinstance(responses, dict)
                        or not expected
                        or set(responses) != expected
                        or not all(
                            isinstance(value, str) and value.strip() for value in responses.values()
                        )
                    ):
                        raise InvalidParamsError(
                            "interrupt_responses must answer the exact pending interrupt IDs"
                        )
                    record["responses"] = responses
                else:
                    if len(interrupts) != 1 or (
                        interrupt_id and interrupt_id != interrupts[0]["id"]
                    ):
                        raise InvalidParamsError("Reply must identify the pending interrupts")
                    record["responses"] = {interrupts[0]["id"]: text}
                record["action"] = "resume"
                record["generation"] += 1
                task = task_from(record)
                task.history.append(message)
                record["snapshot"] = to_dict(task)
            else:
                mapping = await self.store.one(
                    "SELECT * FROM contexts WHERE owner=? AND context_id=?", (owner, context_id)
                )
                if mapping and mapping["profile"] != self.settings.profile.identity:
                    raise InvalidParamsError("Context belongs to a different execution profile")
                task_id = str(uuid4())
                task = a2a.Task(id=task_id, context_id=context_id, history=[message])
                record = {
                    "id": task_id,
                    "owner": owner,
                    "context_id": context_id,
                    "thread_id": mapping["thread_id"] if mapping else str(uuid4()),
                    "profile": self.settings.profile.identity,
                    "generation": 1,
                    "created_at": time.time(),
                    "action": "run",
                    "snapshot": to_dict(task),
                    "interrupts": [],
                }
            record.update(phase="queued", text=text, message_id=message.message_id, digest=digest)
            task = transition(record, a2a.TASK_STATE_SUBMITTED, "Accepted for execution")
            async with self.store.transaction():
                await self.store.db.execute(
                    "INSERT OR IGNORE INTO contexts VALUES(?,?,?,?)",
                    (owner, context_id, record["thread_id"], record["profile"]),
                )
                await self.store.db.execute(
                    "INSERT INTO messages VALUES(?,?,?,?)",
                    (owner, message.message_id, digest, record["id"]),
                )
                await self.store.put(record)
            await self.project(record)
            self.wake.set()
            return task

    async def wait(self, task_id, owner):
        while True:
            # Capture a notification without racing the subsequent state read.
            async with self.changed:
                task = await self.get(task_id, owner)
                if task.status.state in WAITING:
                    return task
                await self.changed.wait()

    async def work(self):
        while self.running:
            await self.wake.wait()
            self.wake.clear()
            while self.running:
                async with self.lock:
                    queued = [r for r in await self.store.records() if r["phase"] == "queued"]
                    if not queued:
                        break
                    record = queued[0]
                    record["phase"] = "preparing"
                    transition(record, a2a.TASK_STATE_WORKING, "Scientific turn is running")
                    async with self.store.transaction():
                        await self.store.put(record)
                    await self.project(record)
                try:
                    # Creating/restoring a graph is serialized with every graph operation.
                    checkpoint = await self.adapter.prepare(record)
                    record["checkpoint_before"] = checkpoint
                    record["phase"] = "executing"
                    async with self.lock:
                        async with self.store.transaction():
                            await self.store.put(record)
                    outcome = await self.adapter.run(record)
                except asyncio.CancelledError:
                    raise
                except CheckpointReviewRequired:
                    outcome = Outcome(
                        state="input_required",
                        reason="configuration",
                        text="The saved checkpoint requires its original compatible configuration.",
                    )
                except Exception:
                    logger.exception("Scientific execution failed for %s", record["id"])
                    outcome = Outcome(
                        state="failed", text="Scientific execution failed; inspect service logs."
                    )
                await self.finish(record, outcome)

    async def finish(self, record, outcome):
        # Save the graph outcome before artifact registration/projection. Reporting
        # and delivery can then be retried independently of scientific invocation.
        record["pending_outcome"] = asdict(outcome)
        async with self.lock:
            async with self.store.transaction():
                await self.store.put(record)
        try:
            await self._finish(record, outcome)
        except (ValueError, OSError):
            logger.exception("Artifact finalization failed for %s", record["id"])
            record.update(phase="finalization", reason="finalization")
            transition(record, a2a.TASK_STATE_INPUT_REQUIRED,
                       "Saved results await artifact finalization; no calculation will be repeated.",
                       "finalization")
            async with self.lock:
                async with self.store.transaction():
                    await self.store.put(record)
                await self.project(record)

    async def _finish(self, record, outcome):
        async with self.lock:
            record.pop("responses", None)
            operations = await self.operations.list(record["id"])
            if any(op["state"] in {"submitting", "invoking", "uncertain"} for op in operations):
                outcome = Outcome(
                    state="input_required",
                    reason="reconciliation",
                    text="Submission outcome is unknown. An operator must reconcile it; no resubmission will occur.",
                    interrupts=outcome.interrupts,
                )
            elif any(op["state"] in {"returned", "nonexecution", "authorized"} for op in operations):
                outcome = Outcome(
                    state="input_required", reason="reconciliation",
                    text="Execution needs reconciliation; saved results can be finalized without another calculation.",
                    interrupts=outcome.interrupts,
                )
            elif outcome.state in {"completed", "failed"} and any(
                op["state"] == "running" for op in operations
            ):
                outcome = Outcome(
                    state="working",
                    reason="backend",
                    text="Remote computation is running; results and reporting are pending.",
                )
            record["phase"] = "waiting" if outcome.state == "input_required" else "done"
            if outcome.reason == "backend":
                record["phase"] = "backend"
            record["interrupts"] = outcome.interrupts
            task = transition(
                record,
                getattr(a2a, "TASK_STATE_" + outcome.state.upper()),
                outcome.text,
                outcome.reason,
                interrupts=outcome.interrupts,
            )
            async with self.store.transaction():
                await self.operations.bind_interrupts(record, outcome.interrupts)
                remote_operations = [op for op in operations if op.get("kind") == "remote"]
                if remote_operations and outcome.state in {"completed", "failed"}:
                    directory = self.settings.state_dir / "artifacts" / record["id"]
                    directory.mkdir(parents=True, exist_ok=True)
                    path = directory / "remote-results.json"
                    results = [
                        {
                            k: op.get(k)
                            for k in ("id", "state", "results", "calculator", "model", "endpoint")
                        }
                        for op in remote_operations
                    ]
                    await asyncio.to_thread(path.write_text, json.dumps(results, indent=2))
                    outcome.artifacts.append(
                        {
                            "path": str(path),
                            "media_type": "application/json",
                            "settings": {"profile": record["profile"]},
                            "units": "as_reported",
                            "validation": "requires_scientific_review",
                        }
                    )
                existing_artifacts = {item.artifact_id for item in task.artifacts}
                for artifact in outcome.artifacts:
                    registered = await self.register_artifact(record, artifact)
                    if registered.artifact_id not in existing_artifacts:
                        task.artifacts.append(registered)
                        existing_artifacts.add(registered.artifact_id)
                record["snapshot"] = to_dict(task)
                record.pop("pending_outcome", None)
                await self.store.put(record)
            await self.project(record)

    async def reconcile_jobs(self):
        await self.operations.poll()
        async with self.lock:
            for record in await self.store.records():
                if record["phase"] != "backend":
                    continue
                operations = await self.operations.list(record["id"])
                if not operations or any(
                    op["state"] not in {"completed", "failed", "canceled"} for op in operations
                ):
                    continue
                record.update(
                    phase="queued",
                    action="run",
                    text=(
                        "The tracked remote computations have finished. Interpret the following results, "
                        "including failures, convergence, units and scientific comparability. Do not resubmit.\n"
                        + json.dumps(
                            [
                                {
                                    k: op.get(k)
                                    for k in ("id", "state", "results", "calculator", "model")
                                }
                                for op in operations
                            ]
                        )
                    ),
                )
                transition(
                    record, a2a.TASK_STATE_WORKING, "Results collected; scientific report pending"
                )
                async with self.store.transaction():
                    await self.store.put(record)
                await self.project(record)
                self.wake.set()

    async def monitor_jobs(self):
        while self.running:
            await self.reconcile_jobs()
            await asyncio.sleep(1)

    async def register_artifact(self, record, artifact):
        from pathlib import Path

        path = Path(artifact["path"]).resolve(strict=True)
        root = (self.settings.state_dir / "artifacts").resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Artifact is outside the allowed location")
        expected = {root / record["id"], root / record["thread_id"]}
        if not any(path.is_relative_to(directory) for directory in expected):
            raise ValueError("Artifact does not belong to this task/thread")
        content = await asyncio.to_thread(path.read_bytes)
        digest = hashlib.sha256(content).hexdigest()
        if artifact.get("sha256") and artifact["sha256"] != digest:
            raise ValueError("Upstream artifact snapshot content changed")
        artifact_id = hashlib.sha256(f"{record['id']}:{path}:{digest}".encode()).hexdigest()
        destination = root / "registered" / artifact_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Write the bytes that were hashed, not a second read of a mutable source.
        await asyncio.to_thread(destination.write_bytes, content)
        metadata = {k: v for k, v in artifact.items() if k != "path"}
        metadata.update(sha256=digest, size=len(content))
        data = {"path": str(destination), "name": path.name, **metadata}
        await self.store.db.execute(
            "INSERT OR REPLACE INTO artifacts VALUES(?,?,?)",
            (artifact_id, record["id"], json.dumps(data)),
        )
        return a2a.Artifact(
            artifact_id=artifact_id,
            name=path.name,
            metadata=metadata,
            parts=[
                a2a.Part(
                    url=f"{self.settings.public_url.rstrip('/')}/artifacts/{artifact_id}",
                    media_type=artifact["media_type"],
                )
            ],
        )

    async def artifact(self, artifact_id, owner):
        from pathlib import Path

        row = await self.store.one("SELECT * FROM artifacts WHERE id=?", (artifact_id,))
        if not row:
            raise TaskNotFoundError()
        await self.owned(row["task_id"], owner)
        data = json.loads(row["data"])
        path = Path(data["path"])
        if path.is_symlink() or not path.is_file():
            raise TaskNotFoundError("Artifact unavailable")
        body = await asyncio.to_thread(path.read_bytes)
        if hashlib.sha256(body).hexdigest() != data["sha256"]:
            raise TaskNotFoundError("Artifact content changed")
        return body, data

    async def cancel(self, task_id, owner):
        await self.owned(task_id, owner)
        await self.operations.cancel(task_id)
        async with self.lock:
            record = await self.owned(task_id, owner)
            if task_from(record).status.state in TERMINAL:
                raise TaskNotCancelableError()
            operations = await self.operations.list(task_id)
            if any(op["state"] in {"running", "submitting", "invoking", "returned", "uncertain"} for op in operations):
                raise TaskNotCancelableError(
                    "Remote cancellation is not confirmed; tracking continues"
                )
            if (
                record["phase"] in {"preparing", "executing"}
                or record.get("reason") == "reconciliation"
            ):
                raise TaskNotCancelableError(
                    "Execution cannot be confirmed stopped; tracking continues"
                )
            record["phase"] = "done"
            task = transition(record, a2a.TASK_STATE_CANCELED, "Cancellation confirmed")
            async with self.store.transaction():
                await self.store.put(record)
                # A canceled graph may still have a pending checkpoint. Retire its thread.
                await self.store.db.execute(
                    "DELETE FROM contexts WHERE owner=? AND context_id=?",
                    (owner, record["context_id"]),
                )
            await self.project(record)
            return task

    async def metrics(self):
        records = await self.store.records()
        outbox = await self.store.all("SELECT state,COUNT(*) AS count FROM outbox GROUP BY state")
        undelivered = await self.store.all("SELECT payload FROM outbox WHERE state!='delivered'")
        ages = [
            time.time() - json.loads(r["payload"]).get("created_at", time.time())
            for r in undelivered
        ]
        jobs = [json.loads(r["data"]) for r in await self.store.all("SELECT data FROM operations")]
        return {
            "ready": self.ready,
            "tasks": len(records),
            "reconciliation": sum(r.get("reason") == "reconciliation" for r in records),
            "outbox": {r["state"]: r["count"] for r in outbox},
            "oldest_undelivered_seconds": max(ages, default=0),
            "oldest_active_task_seconds": max(
                (
                    time.time() - r.get("created_at", time.time())
                    for r in records
                    if r["phase"] != "done"
                ),
                default=0,
            ),
            "backend_unavailable_jobs": sum(bool(op.get("availability_error")) for op in jobs),
            "unreconciled_submissions": sum(op["state"] == "uncertain" for op in jobs),
            "time": time.time(),
        }
