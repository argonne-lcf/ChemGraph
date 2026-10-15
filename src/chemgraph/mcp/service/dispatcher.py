"""Fair admission and independent, supervised calculation processes."""

import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time

from chemgraph.runtime import OperationJournal
from chemgraph.runtime.journal import APPROVED, FINALIZED, INVOKING, RETURNED
from chemgraph.runtime.policy import PreparedAction
from chemgraph.mcp.service.operations import call_from_record
from chemgraph.mcp.service.worker import execute, policy_dict, write_report

logger = logging.getLogger(__name__)


@dataclass
class Running:
    owner: str
    task: asyncio.Task | None = None
    process: asyncio.subprocess.Process | None = None
    stop_reason: str | None = None


class Dispatcher:
    def __init__(self, service):
        self.service, self.config = service, service.config
        self.active = {}
        self.wake = asyncio.Event()
        self.last_owner = None
        self.loop_task = None
        self.lock_file = None

    async def start(self):
        if os.name != "posix":
            raise RuntimeError("The supervised service currently requires a POSIX host.")
        import fcntl

        self.lock_file = (self.config.data_dir / ".service.lock").open("a+")
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            await OperationJournal(self.service.store).recover()
            for record in await self.service.store.records():
                if record["state"] in {RETURNED, FINALIZED}:
                    async with self.service.locks[record["owner"]]:
                        await self.service.finalize(record)
            self.loop_task = asyncio.create_task(self.run())
            self.service.ready = True
            self.wake.set()
        except BaseException:
            self.lock_file.close()
            self.lock_file = None
            raise

    async def run(self):
        try:
            while not self.service.stopping:
                self.wake.clear()
                pending = await self.service.store.records(state=APPROVED)
                grouped = defaultdict(list)
                for record in pending:
                    if record["id"] not in self.active:
                        grouped[record["owner"]].append(record)
                owners = list(grouped)
                if self.last_owner in owners:
                    index = owners.index(self.last_owner) + 1
                    owners = owners[index:] + owners[:index]
                counts = Counter(item.owner for item in self.active.values())
                while owners and len(self.active) < self.config.max_workers:
                    owner = owners.pop(0)
                    if counts[owner] >= self.config.per_user_running:
                        continue
                    record = grouped[owner].pop(0)
                    running = Running(owner)
                    self.active[record["id"]] = running
                    running.task = asyncio.create_task(self.execute(record, running))
                    counts[owner] += 1
                    self.last_owner = owner
                    if grouped[owner]:
                        owners.append(owner)
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=1)
                except TimeoutError:
                    pass
        except Exception:
            self.service.ready = False
            logger.exception("service_dispatcher_stopped")
            raise

    async def execute(self, record, running):
        owner, identifier = record["owner"], record["id"]
        context = self.service.record_context(record)
        call = call_from_record(record)
        try:
            async with self.service.locks[owner]:
                record = await context.journal.get(identifier)
                if self.service.stopping or record["state"] != APPROVED:
                    return
                record = await context.journal.begin(context, call, record)
                if record["state"] != INVOKING:
                    return
                if record["tool"] != "run_ase":
                    report = await asyncio.to_thread(execute, PreparedAction.from_dict(record["prepared"]), context.scope)
                    await self.service.returned(context, call, record, report)
                    return
                running.process = await self.spawn(context, call, record)
            logger.info("job_started owner=%s job=%s pid=%s", owner, identifier, running.process.pid)
            report = await self.watch(running, context)
            async with self.service.locks[owner]:
                if report is None:
                    await context.journal.uncertain(identifier, reason=running.stop_reason or "Worker exited without a valid report.")
                else:
                    await self.service.returned(context, call, record, report)
            logger.info("job_finished owner=%s job=%s", owner, identifier)
        except Exception:
            # A failure after RETURNED must preserve the saved result for recovery.
            current = await context.journal.get(identifier)
            if current is not None and current["state"] == INVOKING:
                await context.journal.uncertain(identifier, reason="Service lost the invocation outcome.")
            logger.exception("job_collection_failed owner=%s job=%s", owner, identifier)
        finally:
            if running.process is not None and running.process.returncode is None:
                await self.terminate(running.process)
            self.active.pop(identifier, None)
            self.wake.set()

    async def spawn(self, context, call, record):
        root = context.workspace
        payload = {"owner": context.owner, "thread_id": context.thread_id, "task_id": context.task_id,
                   "call": call, "prepared": record["prepared"], "policy": policy_dict(context.policy),
                   "max_file_bytes": self.config.max_storage_bytes}
        request = root / ".service-request.json"
        await asyncio.to_thread(write_report, request, payload)
        home, temporary = root / ".service-home", root / ".service-tmp"
        home.mkdir(exist_ok=True)
        temporary.mkdir(exist_ok=True)
        # Explicit allowlist: bearer/LLM/Globus credentials are not inherited.
        env = {key: os.environ[key] for key in (
            "PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS") if key in os.environ}
        env.update(HOME=str(home), TMPDIR=str(temporary), PYTHONNOUSERSITE="1",
                   PYTHONPATH=str(Path(__file__).resolve().parents[3]))
        with (root / ".service-stdout.log").open("wb") as stdout, (root / ".service-stderr.log").open("wb") as stderr:
            return await asyncio.create_subprocess_exec(
                sys.executable, "-m", "chemgraph.mcp.service.worker", str(request),
                cwd=root, env=env, stdin=asyncio.subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                start_new_session=True,
            )

    async def watch(self, running, context):
        process = running.process
        started = time.monotonic()
        waiter = asyncio.create_task(process.wait())
        try:
            while not waiter.done():
                await asyncio.wait({waiter}, timeout=0.5)
                if time.monotonic() - started > self.config.max_job_seconds:
                    running.stop_reason = "Calculation time limit exceeded."
                try:
                    await asyncio.to_thread(self.service.files.check_quota, context.owner)
                except ValueError:
                    running.stop_reason = "Workspace storage quota exceeded."
                if running.stop_reason:
                    await self.terminate(process)
                    break
            await waiter
        finally:
            if not waiter.done():
                await self.terminate(process)
                await waiter
        if running.stop_reason or process.returncode != 0:
            return None
        path = context.workspace / ".service-return.json"
        try:
            if path.stat().st_size > self.config.max_upload_bytes:
                return None
            report = await asyncio.to_thread(lambda: json.loads(path.read_text()))
            if not isinstance(report, dict) or not isinstance(report.get("registered"), list):
                return None
            if "result" not in report or type(report.get("error")) is not bool:
                return None
            return report
        except (OSError, ValueError):
            return None

    @staticmethod
    async def terminate(process):
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    async def cancel(self, owner, identifier):
        journal = self.service.journal(owner)
        task = None
        async with self.service.locks[owner]:
            record = await journal.get(identifier)
            if record is None:
                return {"error": "Unknown job."}
            if record["state"] == APPROVED:
                await journal.abandon(identifier, decided_by=owner, reason="Cancelled before dispatch.")
                self.wake.set()
            elif record["state"] == INVOKING:
                running = self.active.get(identifier)
                if running and running.process and running.process.returncode is None:
                    running.stop_reason = "Cancelled during invocation."
                    await self.terminate(running.process)
                    task = running.task
                else:
                    return {**self.service.summary(record), "cancel_requested": False}
        if task:
            await asyncio.shield(task)
        return await self.service.status(owner, identifier)

    async def stop(self):
        self.service.ready = False
        self.service.stopping = True
        self.wake.set()
        try:
            if self.loop_task:
                # A failed dispatcher still owns children that must be stopped.
                await asyncio.gather(self.loop_task, return_exceptions=True)
            tasks = [item.task for item in self.active.values()] + list(self.service.inline_tasks)
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=self.config.shutdown_seconds)
                if pending:
                    for running in list(self.active.values()):
                        if running.process and running.process.returncode is None:
                            running.stop_reason = "Service stopped during invocation."
                            await self.terminate(running.process)
                    await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if self.lock_file:
                self.lock_file.close()
                self.lock_file = None
