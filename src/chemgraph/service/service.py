"""Single-process orchestration over MainAgentSession and the scoped journal."""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from chemgraph.agent.main_session import MainAgentSession, MissingCheckpointError
from chemgraph.execution.jobs import LocalJobBackend
from chemgraph.graphs.main_agent import construct_main_agent_graph
from chemgraph.memory.graph_config import GRAPH_SCHEMA_VERSION, fingerprint
from chemgraph.memory.schemas import MainAgentGraphConfig, MainAgentSessionMetadata
from chemgraph.memory.store import SessionStore
from chemgraph.registry.agents import AgentRegistry
from chemgraph.registry.tools import ToolRegistry
from chemgraph.runtime import ExecutionPolicy, FileOperationStore, RunContext, run_context
from chemgraph.runtime.artifacts import artifact_file, verify_file
from chemgraph.runtime.journal import ABANDONED, APPROVED, FINALIZED, PROPOSED, REJECTED, UNCERTAIN
from chemgraph.runtime.jobs import JobJournal
from chemgraph.service.records import Artifact, Interrupt, Job, Run, Session
from chemgraph.service.store import ServiceStore

logger = logging.getLogger(__name__)
_TERMINAL = {"completed", "failed", "cancelled", "recovery_required"}


@dataclass
class _Entry:
    context: RunContext
    session: MainAgentSession | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task | None = None


class RunService:
    """Durable local conversations, reviewed calculations and immutable artifacts.

    A custom model_factory(session_id) requires a stable, non-secret
    configuration_id and must be supplied again on reopen. The default uses the
    existing model loader. Graceful close waits for running local calculations.
    """

    def __init__(self, data_root, *, model_name="gpt-4o-mini", base_url=None,
                 model_factory=None, configuration_id=None, policy=None,
                 on_event=None, poll_interval=0.1):
        if model_factory is not None and not configuration_id:
            raise ValueError("A model_factory requires a stable configuration_id.")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive.")
        self.root = Path(data_root).expanduser().resolve()
        self.model_name, self.base_url = model_name, base_url
        self.model_factory, self.configuration_id = model_factory, configuration_id
        self.policy = policy or ExecutionPolicy()
        if self.policy.calculators - {"emt"}:
            raise ValueError("The local service milestone supports EMT only.")
        self.on_event, self.poll_interval = on_event, poll_interval
        self._entries = {}
        self._stack = None
        self._closing = False
        self._monitor = None

    async def __aenter__(self):
        if self._stack is not None:
            raise RuntimeError("RunService is already open.")
        self._closing = False
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        try:
            self.store = ServiceStore(self.root)
            self._stack.callback(self.store.close)
            self.history = SessionStore(str(self.root / "sessions.db"))
            self.backend = LocalJobBackend(self.root / "jobs")
            self.saver = await self._stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(str(self.root / "checkpoints.db")))
            self.saver.serde = JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=None)
            await self.saver.setup()
            launches = []
            for record in self.store.records("session"):
                snapshot = Session.model_validate(record)
                entry = await self._entry(snapshot)
                await entry.context.journal.recover()
                if snapshot.status == "cancelled" or snapshot.current_run_id is None:
                    continue
                run = self.get_run(snapshot.current_run_id)
                entry.context.task_id = run.id
                try:
                    result = await entry.session.restore()
                except MissingCheckpointError:
                    launches.append(run)
                else:
                    checkpoint = await entry.session.workflow.aget_state(entry.session.config)
                    if (checkpoint.metadata or {}).get("chemgraph_service_run_id") != run.id:
                        # The turn intent may have committed before its first graph
                        # checkpoint. Do not mistake the preceding turn for this one.
                        if result.status != "completed":
                            raise ValueError("The current run does not match its graph checkpoint.")
                        launches.append(run)
                    else:
                        await self._record_result(run, result)
            for run in launches:
                self._launch(run, "run", run.message)
            self._monitor = asyncio.create_task(self._monitor_loop())
            return self
        except BaseException:
            self._closing = True
            tasks = [entry.task for entry in self._entries.values() if entry.task]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._stack.aclose()
            self._stack = None
            self._entries.clear()
            raise

    async def __aexit__(self, *exc):
        await self.close()

    def _open(self):
        if self._stack is None:
            raise RuntimeError("Use RunService as an async context manager.")

    def get_session(self, session_id):
        self._open()
        return Session.model_validate(self.store.get("session", session_id))

    def list_sessions(self):
        self._open()
        return tuple(Session.model_validate(r) for r in self.store.records("session"))

    def get_run(self, run_id):
        self._open()
        return Run.model_validate(self.store.get("run", run_id))

    def _update_run(self, run, **changes):
        run = run.model_copy(update=changes)
        self.store.put("run", run)
        return run

    def _active(self, run_id):
        run = self.get_run(run_id)
        session = self.get_session(run.session_id)
        if self._closing or session.status == "cancelled":
            raise RuntimeError("This session cannot advance; create a new session.")
        if session.current_run_id != run_id:
            raise ValueError("The run is no longer current.")
        return run, self._entries[run.session_id]

    async def _entry(self, snapshot):
        if snapshot.id in self._entries:
            return self._entries[snapshot.id]
        journal = JobJournal(FileOperationStore(self.root / "operations" / snapshot.id), self.backend)
        context = RunContext("local", snapshot.id, snapshot.current_run_id or "idle",
                             Path(snapshot.workspace), self.policy, journal)
        context.cancelled = snapshot.status == "cancelled"
        entry = _Entry(context)
        expected = self._configuration_identity()
        if snapshot.configuration_id != expected:
            raise ValueError("Service configuration changed; reopen with the original configuration.")
        if snapshot.status != "cancelled":
            if self.model_factory:
                model = self.model_factory(snapshot.id)
                endpoint = None
            else:
                from chemgraph.models.loader import load_chat_model_prepared
                model, prepared = load_chat_model_prepared(model_name=self.model_name, base_url=self.base_url)
                endpoint = prepared.endpoint_descriptor
            tools = ToolRegistry().resolve([*sorted(self.policy.tools), "ask_human"])
            graph = construct_main_agent_graph(
                model, main_tools=tools, tool_registry=ToolRegistry([]),
                agent_registry=AgentRegistry([]), backend=None, discover_skills=False,
                interrupt_on=None, scoped=True, checkpointer=self.saver,
            )
            config = MainAgentGraphConfig(
                model_name=self.model_name, model_endpoint=endpoint,
                artifact_directory=snapshot.workspace, graph_schema_version=GRAPH_SCHEMA_VERSION,
                approval_mode="scoped", subagent_names=(), discover_skills=False,
                configuration_id=expected, requires_configuration_id=True,
                topology_fingerprint=fingerprint({"profile": expected, "workspace": snapshot.workspace,
                                                 "schema": GRAPH_SCHEMA_VERSION,
                                                 "endpoint": endpoint.model_dump() if endpoint else None}),
            )
            metadata = MainAgentSessionMetadata(graph_config=config, checkpoint_backend="AsyncSqliteSaver",
                                                checkpoint_db=str(self.root / "checkpoints.db"))
            if self.history.get_session(snapshot.id) is None:
                self.history.create_session(snapshot.id, self.model_name, "main_agent",
                                            log_dir=snapshot.workspace, status="new",
                                            session_metadata=metadata)
            entry.session = MainAgentSession(graph, thread_id=snapshot.id, session_store=self.history,
                                             session_metadata=metadata, on_event=self.on_event)
        self._entries[snapshot.id] = entry
        return entry

    def _configuration_identity(self):
        return fingerprint({"service": 1, "backend": self.backend.identity, "policy": self.policy.identity,
                            "model": self.model_name, "base_url": self.base_url,
                            "caller": self.configuration_id})

    async def create_session(self):
        self._open()
        if self._closing:
            raise RuntimeError("RunService is closing.")
        identifier = uuid.uuid4().hex
        snapshot = Session(id=identifier, workspace=str(self.root / "workspaces" / identifier),
                           configuration_id=self._configuration_identity())
        await self._entry(snapshot)
        self.store.put("session", snapshot)
        return snapshot

    async def start_run(self, session_id, message):
        if not isinstance(message, str) or not message.strip():
            raise ValueError("A run requires a nonempty message.")
        snapshot = self.get_session(session_id)
        entry = self._entries[session_id]
        async with entry.lock:
            snapshot = self.get_session(session_id)
            if self._closing or snapshot.status == "cancelled":
                raise RuntimeError("This session is terminal; create a new session.")
            if snapshot.current_run_id and self.get_run(snapshot.current_run_id).status != "completed":
                raise RuntimeError("Finish, retry or cancel the current run before starting another.")
            run = Run(id=uuid.uuid4().hex, session_id=session_id, message=message)
            entry.context.task_id = run.id
            self.store.put("run", run)
            self.store.put("session", snapshot.model_copy(update={"current_run_id": run.id, "status": "active"}))
            self._launch(run, "run", message)
            return run

    def _launch(self, run, method, value=None):
        entry = self._entries[run.session_id]
        snapshot = self.get_session(run.session_id)
        if snapshot.status == "cancelled" or snapshot.current_run_id != run.id:
            raise RuntimeError("This run cannot advance.")
        if entry.task is not None and not entry.task.done():
            raise RuntimeError("The session already has an active graph operation.")
        entry.session.config["metadata"]["chemgraph_service_run_id"] = run.id
        self._update_run(self.get_run(run.id), status="queued", interrupts=())
        entry.task = asyncio.create_task(self._drive(run.id, method, value))

    async def _drive(self, run_id, method, value):
        run = self.get_run(run_id)
        entry = self._entries[run.session_id]
        async with entry.lock:
            if self.get_session(run.session_id).status == "cancelled":
                return
            self._update_run(self.get_run(run_id), status="running", error=None)
            try:
                with run_context(entry.context):
                    operation = getattr(entry.session, method)
                    result = await (operation() if method == "retry" else operation(value))
                if self.get_session(run.session_id).status != "cancelled":
                    await self._record_result(self.get_run(run_id), result)
            except asyncio.CancelledError:
                await entry.context.journal.mark_interrupted(entry.context.thread_id)
                raise
            except Exception as exc:
                if self.get_session(run.session_id).status == "cancelled":
                    return
                records = await entry.context.journal.store.records(thread_id=run.session_id)
                status = "recovery_required" if any(r["state"] == UNCERTAIN for r in records) else "failed"
                self._update_run(self.get_run(run_id), status=status, error=str(exc), interrupts=())

    async def _record_result(self, run, result):
        pending = tuple(Interrupt(id=i.id, kind=(i.payload.get("kind", "clarification")
                        if isinstance(i.payload, dict) else "clarification"),
                        operation_id=i.payload.get("operation_id") if isinstance(i.payload, dict) else None,
                        payload=i.payload if isinstance(i.payload, dict) else {"question": str(i.payload)})
                        for i in result.interrupts)
        status = result.status
        if pending:
            status = "waiting_for_user" if any(i.kind != "job_wait" for i in pending) else "waiting_for_job"
        responses = {k: v for k, v in run.responses.items() if k in {p.response_key for p in pending}}
        self._update_run(run, status=status, assistant_response=result.assistant_response,
                         interrupts=pending, responses=responses, error=None)
        if status == "completed":
            snapshot = self.get_session(run.session_id)
            self.store.put("session", snapshot.model_copy(update={"status": "ready"}))

    async def respond(self, run_id, interrupt_id, response):
        run, entry = self._active(run_id)
        async with entry.lock:
            run, _ = self._active(run_id)
            if run.status not in {"waiting_for_user", "waiting_for_job"}:
                raise ValueError("This run is not waiting for a response.")
            pending = next((i for i in run.interrupts if i.id == interrupt_id and i.kind == "clarification"), None)
            if not pending or not isinstance(response, str) or not response.strip():
                raise ValueError("A response must match a current clarification and be nonempty.")
            self._update_run(run, responses={**run.responses, pending.response_key: response})
            await self._maybe_resume(self.get_run(run_id), entry)
        return self.get_run(run_id)

    async def decide(self, run_id, operation_id, *, digest, approve, message=None):
        run, entry = self._active(run_id)
        async with entry.lock:
            run, _ = self._active(run_id)
            if run.status not in {"waiting_for_user", "waiting_for_job"}:
                raise ValueError("This run is not waiting for a decision.")
            if not any(i.operation_id == operation_id and i.kind == "approval" for i in run.interrupts):
                raise ValueError("The operation is not a current approval request.")
            await entry.context.journal.decide(operation_id, digest=digest, approve=approve,
                                                decided_by="local", message=message)
            await self._maybe_resume(self.get_run(run_id), entry)
        return self.get_run(run_id)

    async def retry_run(self, run_id):
        run, entry = self._active(run_id)
        async with entry.lock:
            run, _ = self._active(run_id)
            if run.status != "failed":
                raise ValueError("Only a failed graph operation can be retried.")
            await entry.context.journal.advance(entry.context)
            records = await entry.context.journal.store.records(thread_id=run.session_id)
            if any(r["state"] == UNCERTAIN for r in records):
                raise ValueError("Reconcile uncertain operations before retry.")
            self._launch(run, "retry")
        return self.get_run(run_id)

    async def reconcile(self, run_id, operation_id, *, action, evidence):
        run, entry = self._active(run_id)
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Reconciliation requires explicit evidence.")
        async with entry.lock:
            run, _ = self._active(run_id)
            record = await entry.context.journal.get(operation_id)
            if not record or record["task_id"] != run_id:
                raise ValueError("The operation does not belong to this run.")
            if action == "confirm_nonexecution":
                await entry.context.journal.confirm_nonexecution(
                    operation_id, decided_by="local", evidence=evidence)
            elif action == "abandon":
                await entry.context.journal.abandon(operation_id, decided_by="local", reason=evidence)
            else:
                raise ValueError("Use confirm_nonexecution or abandon.")
            await entry.context.journal.advance(entry.context)
            if entry.session.failed:
                self._launch(run, "retry")
            else:
                self._update_run(run, status="waiting_for_job", error=None)
        return self.get_run(run_id)

    async def cancel_run(self, run_id):
        run = self.get_run(run_id)
        entry = self._entries[run.session_id]
        snapshot = self.get_session(run.session_id)
        if snapshot.current_run_id != run_id:
            raise ValueError("Only the current run can be cancelled.")
        if snapshot.status == "cancelled":
            return run
        async with entry.context.lock:
            snapshot = self.get_session(run.session_id)
            if snapshot.current_run_id != run_id:
                raise ValueError("Only the current run can be cancelled.")
            self.store.put("session", snapshot.model_copy(update={"status": "cancelled"}))
            entry.context.cancelled = True
            self._update_run(self.get_run(run_id), status="cancelled")
        if entry.task and not entry.task.done():
            entry.task.cancel()
            await asyncio.gather(entry.task, return_exceptions=True)
        async with entry.lock:
            await entry.context.journal.mark_interrupted(run.session_id)
            for record in await entry.context.journal.store.records(thread_id=run.session_id):
                if record.get("job_id") and await self.backend.lookup(record["job_id"]) is not None:
                    job = await self.backend.status(record["job_id"])
                    if not job["published"] and job["status"] in {"queued", "dispatching", "running"}:
                        await self.backend.cancel(record["job_id"])
            self.history.update_session_status(run.session_id, "cancelled")
        return self.get_run(run_id)

    async def list_jobs(self, run_id):
        run = self.get_run(run_id)
        result = []
        for record in await self.backend.records():
            if record["spec"]["task_id"] == run_id:
                job = await self.backend.status(record["job_id"])
                result.append(Job(id=job["job_id"], operation_id=record["spec"]["operation_id"], run_id=run_id,
                                  status=job["status"], cancel_requested=job["cancel_requested"],
                                  promoted=job.get("promoted")))
        known = {job.id for job in result}
        journal = self._entries[run.session_id].context.journal
        for record in await journal.store.records(thread_id=run.session_id):
            if record["task_id"] == run_id and record.get("job_id") and record["job_id"] not in known:
                result.append(Job(id=record["job_id"], operation_id=record["id"],
                                  run_id=run_id, status="unknown"))
        return tuple(result)

    async def list_artifacts(self, run_id):
        run = self.get_run(run_id)
        records = await self._entries[run.session_id].context.journal.store.records(thread_id=run.session_id)
        return tuple(Artifact(id=a["path"], run_id=run_id, promoted=r["result"].get("promoted", True),
                              **{k: a[k] for k in ("source_path", "path", "media_type", "sha256", "size_bytes")})
                     for r in records if r["task_id"] == run_id and r["state"] == FINALIZED
                     for a in r["result"]["artifacts"])

    async def resolve_artifact(self, run_id, artifact_id):
        artifact = next((a for a in await self.list_artifacts(run_id) if a.id == artifact_id), None)
        if artifact is None:
            raise KeyError(artifact_id)
        run = self.get_run(run_id)
        path = artifact_file(Path(self.get_session(run.session_id).workspace), artifact.model_dump())
        await asyncio.to_thread(verify_file, path, artifact.model_dump())
        return path

    async def wait(self, run_id, *, timeout=30):
        """Wait until a run needs user action, recovery, or has finished."""
        async with asyncio.timeout(timeout):
            while True:
                run = self.get_run(run_id)
                if run.status in _TERMINAL or run.status == "waiting_for_user":
                    return run
                await asyncio.sleep(self.poll_interval)

    async def _maybe_resume(self, run, entry):
        if self._closing or run.status not in {"waiting_for_user", "waiting_for_job"} or not run.interrupts:
            return
        answers = {}
        needs_user = False
        needs_job = False
        for pending in run.interrupts:
            if pending.kind == "clarification":
                if pending.response_key not in run.responses:
                    needs_user = True
                else:
                    answers[pending.id] = run.responses[pending.response_key]
            else:
                record = await entry.context.journal.get(pending.operation_id)
                ready = ({APPROVED, REJECTED} if pending.kind == "approval"
                         else {FINALIZED, ABANDONED, PROPOSED})
                if not record or record["state"] not in ready:
                    if pending.kind == "approval":
                        needs_user = True
                    else:
                        needs_job = True
                else:
                    answers[pending.id] = "journal-ready"
        if needs_user or needs_job:
            status = "waiting_for_user" if needs_user else "waiting_for_job"
            if run.status != status:
                self._update_run(run, status=status)
            return
        self._launch(run, "resume", answers)

    async def _tick(self, *, dispatch=True):
        for session_id, entry in list(self._entries.items()):
            if entry.lock.locked() or (entry.task and not entry.task.done()):
                continue
            async with entry.lock:
                snapshot = self.get_session(session_id)
                if not snapshot.current_run_id:
                    continue
                run = self.get_run(snapshot.current_run_id)
                entry.context.task_id = run.id
                try:
                    await entry.context.journal.advance(entry.context, cancelled=snapshot.status == "cancelled")
                    snapshot = self.get_session(session_id)
                    run = self.get_run(snapshot.current_run_id)
                    if snapshot.status != "cancelled":
                        records = await entry.context.journal.store.records(thread_id=session_id)
                        if any(r["state"] == UNCERTAIN for r in records):
                            self._update_run(run, status="recovery_required")
                        else:
                            if run.status == "recovery_required" and not entry.session.failed:
                                run = self._update_run(run, status="waiting_for_job", error=None)
                            elif run.status == "recovery_required" and entry.session.failed:
                                run = self._update_run(run, status="failed",
                                                       error="The job was recovered; retry the failed graph checkpoint.")
                            await self._maybe_resume(run, entry)
                except Exception as exc:
                    run = self.get_run(snapshot.current_run_id)
                    if run.status not in {"completed", "cancelled"}:
                        self._update_run(run, status="recovery_required", error=str(exc))
                    logger.debug("Job recovery needs attention", exc_info=True)
        if dispatch and not self._closing:
            for job in await self.backend.records():
                if job["status"] != "queued":
                    continue
                entry = self._entries[job["spec"]["thread_id"]]
                if self.get_session(entry.context.thread_id).status == "cancelled":
                    await self.backend.cancel(job["job_id"])
                    continue
                async with entry.context.lock:
                    if self.get_session(entry.context.thread_id).status == "cancelled":
                        await self.backend.cancel(job["job_id"])
                        continue
                    record = await entry.context.journal.get(job["spec"]["operation_id"])
                    if not record or record.get("job_id") != job["job_id"]:
                        await self.backend.cancel(job["job_id"])
                        continue
                    if record and await entry.context.journal.dispatchable(entry.context, record):
                        await self.backend.dispatch(job["job_id"])
                        break

    async def _monitor_loop(self):
        while not self._closing:
            try:
                await self._tick()
            except Exception as exc:
                logger.exception("Job monitoring failed; preserving work for recovery")
                for snapshot in self.list_sessions():
                    if snapshot.status == "active" and snapshot.current_run_id:
                        run = self.get_run(snapshot.current_run_id)
                        self._update_run(run, status="recovery_required", error=str(exc))
            await asyncio.sleep(self.poll_interval)

    async def close(self):
        if self._stack is None:
            return
        self._closing = True
        if self._monitor:
            await asyncio.gather(self._monitor, return_exceptions=True)
        await asyncio.gather(*(e.task for e in self._entries.values() if e.task), return_exceptions=True)
        await self.backend.close()
        await self._tick(dispatch=False)
        await self._stack.aclose()
        self._stack = None
        self._entries.clear()
