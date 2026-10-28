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
    cleanup_task: asyncio.Task | None = None
    cleanup_failed: bool = False


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
            cataloged = await asyncio.to_thread(self.service.files.cataloged_operations)
            for record in await self.service.store.records():
                if record["state"] == RETURNED or (
                        record["state"] == FINALIZED and record["result"]["artifacts"]
                        and record["id"] not in cataloged):
                    try:
                        async with self.service.locks[record["owner"]]:
                            await self.service.finalize(record)
                    except Exception:
                        # One job's storage or quota failure must not stop the service; it
                        # stays returned and get_job_results retries its publication.
                        logger.exception("job_publication_deferred owner=%s job=%s",
                                         record["owner"], record["id"])
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
                try:
                    await self.dispatch()
                except Exception:
                    # Retry transient failures while refusing new submissions.
                    self.service.ready = False
                    logger.exception("service_dispatch_iteration_failed")
                else:
                    self.service.ready = not self.service.stopping and not self.cleanup_failed
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=1)
                except TimeoutError:
                    pass
        finally:
            self.service.ready = False

    @property
    def cleanup_failed(self):
        return any(item.cleanup_failed for item in self.active.values())

    async def dispatch(self):
        pending = await self.service.store.records(state=APPROVED)
        if self.cleanup_failed:
            return
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
            if running.cleanup_failed:
                reason = "Worker cleanup failed; surviving processes possible."
                if running.stop_reason:
                    reason += " " + running.stop_reason
                try:
                    await context.journal.uncertain(identifier, reason=reason)
                except Exception:
                    logger.exception("job_reconciliation_deferred owner=%s job=%s", owner, identifier)
            else:
                # A failure after RETURNED must preserve the saved result for recovery.
                await self.service.collection_failed(context, identifier)
        finally:
            try:
                await self.cleanup(running)
            except Exception:
                pass  # Cleanup logged the failure; retain this slot and refuse new work.
            else:
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
                    break
        finally:
            # A dead leader can leave children writing outputs. Stop the group
            # before collecting its report or making the dispatch slot available.
            try:
                await self.cleanup(running)
            finally:
                if not waiter.done():
                    waiter.cancel()
                await asyncio.gather(waiter, return_exceptions=True)
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

    async def cleanup(self, running):
        if running.process is None:
            return
        if running.cleanup_task is None:
            running.cleanup_task = asyncio.create_task(self.terminate(running.process))
        # Cancellation and the watcher can reach cleanup concurrently.
        try:
            await asyncio.shield(running.cleanup_task)
        except Exception:
            if not running.cleanup_failed:
                identifier = next((key for key, item in self.active.items() if item is running), "unknown")
                logger.exception("job_cleanup_failed owner=%s job=%s", running.owner, identifier)
            running.cleanup_failed = True
            self.service.ready = False
            self.wake.set()
            raise

    @staticmethod
    def signal_group(process, sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return False
        except PermissionError:
            # macOS refuses a group whose leader exited unreaped; signalling the
            # leader itself succeeds then and still fails for a live, denied child.
            if process.returncode is not None:
                raise
            try:
                process.send_signal(sig)
            except ProcessLookupError:
                pass  # The child watcher may have reaped the leader meanwhile.
        return True

    @classmethod
    async def terminate(cls, process):
        # Recheck after signalling: leader exit alone cannot confirm child exit.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            deadline, sent = time.monotonic() + 2, False
            while True:
                try:
                    exists = cls.signal_group(process, 0 if sent and sig == signal.SIGTERM else sig)
                except PermissionError:
                    # Allow the event loop to reap a just-exited macOS leader.
                    if time.monotonic() >= deadline:
                        raise
                else:
                    sent = True
                    if not exists:
                        await asyncio.wait_for(process.wait(), timeout=2)
                        return
                    if time.monotonic() >= deadline:
                        break
                await asyncio.sleep(0.05)
        raise RuntimeError("Worker process group did not exit after SIGKILL.")

    async def cancel(self, owner, identifier):
        journal = self.service.journal(owner)
        task = None
        cleanup_failed = False
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
                    task = running.task
                    try:
                        await self.cleanup(running)
                    except Exception:
                        cleanup_failed = True
                else:
                    return {**self.service.summary(record), "cancel_requested": False}
        if task:
            await asyncio.shield(task)
        result = await self.service.status(owner, identifier)
        if cleanup_failed:
            result.update(cancel_requested=True, error="Worker cleanup failed; ask the operator to restart the service.")
        return result

    async def stop(self):
        self.service.ready = False
        self.service.stopping = True
        self.wake.set()
        if self.loop_task:
            # A failed dispatcher still owns children that must be stopped.
            await asyncio.gather(self.loop_task, return_exceptions=True)
        tasks = [item.task for item in self.active.values()] + list(self.service.inline_tasks)
        if tasks:
            await asyncio.wait(tasks, timeout=self.config.shutdown_seconds)
        running = list(self.active.values())
        for item in running:
            if item.process and item.process.returncode is None:
                item.stop_reason = "Service stopped during invocation."
        # One denied signal must not prevent cleanup attempts for other workers.
        results = await asyncio.gather(*(self.cleanup(item) for item in running), return_exceptions=True)
        await asyncio.gather(*tasks, return_exceptions=True)
        if any(isinstance(result, BaseException) for result in results):
            # Keep the instance lock until process exit; supervisor cleanup is required.
            raise RuntimeError("Worker cleanup failed; service restart requires supervisor cleanup.")
        if self.lock_file:
            self.lock_file.close()
            self.lock_file = None
