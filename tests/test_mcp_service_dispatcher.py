"""Exercise durable boundaries and real process death without chemistry/network."""

import asyncio
import json
import os
import signal
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid

import httpx
from langchain_core.messages import ToolMessage
import pytest

from chemgraph.mcp.service import dispatcher as dispatcher_module
from chemgraph.mcp.service.app import create_app
from chemgraph.mcp.service.dispatcher import Dispatcher
from chemgraph.mcp.service.files import QuotaExceeded
from chemgraph.mcp.service.operations import Service
from chemgraph.runtime.context import ScopedExecutionError
from chemgraph.runtime.journal import APPROVED, INVOKING, RETURNED, OperationJournal
from tests.mcp_service_helpers import WATER, config

posix_only = pytest.mark.skipif(os.name != "posix", reason="The supervised service requires a POSIX host")


async def job(service, owner, *, state=APPROVED):
    context = service.context(owner, uuid.uuid4().hex, "run_ase")
    (context.workspace / "water.xyz").write_text(WATER)
    call = {"id": context.task_id, "name": "run_ase", "args": {"params": {
        "input_structure_file": "water.xyz", "output_results_file": "out.json", "driver": "energy",
        "calculator": {"calculator_type": "emt"}}}}
    record = await context.journal.propose(context, call)
    record = await context.journal.decide(record["id"], digest=record["prepared"]["digest"],
                                          approve=True, decided_by="test-policy")
    if state in {INVOKING, RETURNED}:
        record = await context.journal.begin(context, call, record)
    if state == RETURNED:
        (context.workspace / "out.json").write_text('{"energy": 1}')
        record = await context.journal.record_return(record, ToolMessage(
            content='{"energy": 1}', tool_call_id=call["id"]), ["out.json"])
    return record, context


async def wait_for(predicate, *, timeout=15):
    async with asyncio.timeout(timeout):
        while not await predicate():
            await asyncio.sleep(0.02)


def fake_workers(dispatcher, monkeypatch, *, blocked=False):
    launched = []

    async def spawn(context, call, record):
        launched.append(record["id"])
        code = """
import json, sys, time
from pathlib import Path
root = Path(sys.argv[1])
if sys.argv[2] == 'blocked':
    while not (root / 'release').exists():
        time.sleep(0.02)
(root / '.service-return.json').write_text(json.dumps({
    'result': {'ok': True}, 'error': False, 'registered': []}))
"""
        return await asyncio.create_subprocess_exec(sys.executable, "-c", code, str(context.workspace),
                                                    "blocked" if blocked else "free", start_new_session=True)
    monkeypatch.setattr(dispatcher, "spawn", spawn)
    return launched


@posix_only
@pytest.mark.asyncio
async def test_crash_only_affects_its_job_and_other_users_and_queue_continue(tmp_path, monkeypatch):
    service = Service(config(tmp_path, max_workers=2))
    dispatcher = service.dispatcher = Dispatcher(service)
    first, c1 = await job(service, "alice")
    queued, c2 = await job(service, "alice")
    other, cb = await job(service, "bob")
    launched = fake_workers(dispatcher, monkeypatch, blocked=True)
    await dispatcher.start()
    try:
        async def both_running():
            return all(identifier in dispatcher.active and dispatcher.active[identifier].process
                       for identifier in (first["id"], other["id"]))
        await wait_for(both_running)
        assert set(launched) == {first["id"], other["id"]}
        os.kill(dispatcher.active[first["id"]].process.pid, signal.SIGKILL)

        async def queued_started():
            return queued["id"] in launched
        await wait_for(queued_started)
        assert (await service.status("alice", first["id"]))["status"] == "uncertain"
        assert (await service.status("bob", other["id"]))["status"] == "invoking"
        (c2.workspace / "release").touch()
        (cb.workspace / "release").touch()

        async def finished():
            states = [await service.status(owner, rec["id"]) for owner, rec in (("alice", queued), ("bob", other))]
            return all(state["status"] == "finalized" for state in states)
        await wait_for(finished)
        assert launched.count(queued["id"]) == 1
    finally:
        await dispatcher.stop()


@posix_only
@pytest.mark.asyncio
async def test_restart_dispatches_only_approved_and_finalizes_returned(tmp_path, monkeypatch):
    service = Service(config(tmp_path))
    queued, _ = await job(service, "alice")
    invoking, _ = await job(service, "bob", state=INVOKING)
    returned, _ = await job(service, "carol", state=RETURNED)
    # Fresh objects against the same durable database, without live futures.
    service = Service(config(tmp_path))
    dispatcher = service.dispatcher = Dispatcher(service)
    launched = fake_workers(dispatcher, monkeypatch)
    await dispatcher.start()
    try:
        async def done():
            return (await service.status("alice", queued["id"]))["status"] == "finalized"
        await wait_for(done)
        assert launched == [queued["id"]]
        assert (await service.status("bob", invoking["id"]))["status"] == "uncertain"
        saved = await service.results("carol", returned["id"])
        assert saved["result"] == {"energy": 1} and len(saved["artifacts"]) == 1
        artifact = service.files.read("carol", saved["artifacts"][0]["artifact_id"])
        assert json.loads(artifact["content"]) == {"energy": 1}
    finally:
        await dispatcher.stop()


@posix_only
@pytest.mark.asyncio
async def test_startup_publication_failure_affects_only_that_job(tmp_path, monkeypatch):
    service = Service(config(tmp_path))
    broken, context = await job(service, "alice", state=RETURNED)
    (context.workspace / "out.json").unlink()
    queued, _ = await job(service, "bob")
    service = Service(config(tmp_path))
    dispatcher = service.dispatcher = Dispatcher(service)
    launched = fake_workers(dispatcher, monkeypatch)
    await dispatcher.start()
    try:
        assert service.ready

        async def done():
            return (await service.status("bob", queued["id"]))["status"] == "finalized"
        await wait_for(done)
        assert launched == [queued["id"]]
        assert (await service.status("alice", broken["id"]))["status"] == "returned"
        with pytest.raises(ScopedExecutionError, match="will not be repeated") as error:
            await service.results("alice", broken["id"])
        assert str(tmp_path) not in str(error.value)
        (context.workspace / "out.json").write_text('{"energy": 1}')
        saved = await service.results("alice", broken["id"])
        assert saved["status"] == "finalized" and len(saved["artifacts"]) == 1
    finally:
        await dispatcher.stop()


@posix_only
@pytest.mark.asyncio
async def test_queued_cancellation_and_changed_snapshot_never_launch(tmp_path, monkeypatch):
    service = Service(config(tmp_path))
    dispatcher = service.dispatcher = Dispatcher(service)
    cancel, _ = await job(service, "alice")
    tampered, context = await job(service, "bob")
    snapshot = context.workspace / tampered["prepared"]["inputs"][0]["path"]
    snapshot.write_text("changed")
    assert (await dispatcher.cancel("alice", cancel["id"]))["status"] == "abandoned"
    assert await dispatcher.cancel("bob", cancel["id"]) == {"error": "Unknown job."}
    launched = fake_workers(dispatcher, monkeypatch)
    await dispatcher.start()
    try:
        async def blocked():
            return (await service.status("bob", tampered["id"]))["status"] == "blocked"
        await wait_for(blocked)
        assert not launched
    finally:
        await dispatcher.stop()


@pytest.mark.asyncio
async def test_queue_limit_and_disconnected_caller(tmp_path):
    service = Service(config(tmp_path, per_user_queued=1))
    service.dispatcher = Dispatcher(service)
    service.ready = True  # Deliberately keep dispatch paused to fill the queue.
    await service.put_file("alice", "water.xyz", WATER)
    args = {"params": {"input_structure_file": "water.xyz", "driver": "energy", "calculator": {"calculator_type": "emt"}}}
    await service.call("alice", "run_ase", args)
    with pytest.raises(ValueError, match="queue limit"):
        await service.call("alice", "run_ase", args)
    started, release = asyncio.Event(), asyncio.Event()
    completed = []

    async def accepted_operation():
        started.set()
        await release.wait()
        completed.append(True)
    request = asyncio.create_task(service.track(accepted_operation()))
    await started.wait()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    release.set()
    await asyncio.gather(*service.inline_tasks)
    assert completed == [True]


@posix_only
@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["cancel", "timeout", "shutdown"])
async def test_running_process_is_reaped_and_never_requeued(tmp_path, monkeypatch, stop):
    service = Service(config(tmp_path, max_job_seconds=1, shutdown_seconds=1))
    dispatcher = service.dispatcher = Dispatcher(service)
    record, _ = await job(service, "alice")
    launched = fake_workers(dispatcher, monkeypatch, blocked=True)
    await dispatcher.start()
    try:
        async def started():
            return record["id"] in dispatcher.active and dispatcher.active[record["id"]].process
        await wait_for(started)
        process = dispatcher.active[record["id"]].process
        if stop == "cancel":
            await dispatcher.cancel("alice", record["id"])
        elif stop == "shutdown":
            # Even a dispatcher-loop failure must not skip child cleanup.
            dispatcher.loop_task.cancel()
            await dispatcher.stop()
        async def uncertain():
            return (await service.status("alice", record["id"]))["status"] == "uncertain"
        await wait_for(uncertain)
        assert process.returncode is not None
        assert launched == [record["id"]]
    finally:
        await dispatcher.stop()


@posix_only
@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["cancel", "timeout", "shutdown", "normal", "crash"])
async def test_worker_children_stop_before_publication_and_next_job(tmp_path, monkeypatch, stop):
    service = Service(config(tmp_path, max_workers=1, shutdown_seconds=1,
                             max_job_seconds=1 if stop == "timeout" else 60))
    dispatcher = service.dispatcher = Dispatcher(service)
    record, context = await job(service, "alice")
    heartbeat = context.workspace / "heartbeat"
    dispatchers = [dispatcher]
    processes, observations = [], {}
    child = """
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(sys.argv[1], 'ab', buffering=0) as stream:
    while True:
        stream.write(b'x')
        time.sleep(0.01)
"""
    worker = """
import json, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1])
if sys.argv[2]:
    child = subprocess.Popen([sys.executable, '-c', sys.argv[2], str(root / 'heartbeat')])
    (root / 'child.pid').write_text(str(child.pid))
    while not (root / 'release').exists():
        time.sleep(0.01)
(root / '.service-return.json').write_text(json.dumps({
    'result': {'ok': True}, 'error': False, 'registered': []}))
sys.exit(1 if sys.argv[3] == 'crash' else 0)
"""

    async def stopped_writing():
        before = heartbeat.stat().st_size
        await asyncio.sleep(0.15)
        return heartbeat.stat().st_size == before

    async def spawn(ctx, call, rec):
        first = rec["id"] == record["id"]
        if not first:
            observations["next_job"] = await stopped_writing()
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", worker, str(ctx.workspace), child if first else "",
            stop if first else "normal", start_new_session=True,
            stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        processes.append(process)
        return process

    returned = service.returned
    async def collect(ctx, call, rec, report):
        if rec["id"] == record["id"]:
            observations["publication"] = await stopped_writing()
        return await returned(ctx, call, rec, report)

    monkeypatch.setattr(dispatcher, "spawn", spawn)
    monkeypatch.setattr(service, "returned", collect)
    await dispatcher.start()
    try:
        async def child_started():
            return heartbeat.exists() and heartbeat.stat().st_size > 0
        await wait_for(child_started)
        queued, _ = await job(service, "alice")
        if stop == "cancel":
            await dispatcher.cancel("alice", record["id"])
        elif stop == "shutdown":
            await dispatcher.stop()
        elif stop in {"normal", "crash"}:
            (context.workspace / "release").touch()

        async def finished():
            return record["id"] not in dispatcher.active
        await wait_for(finished)
        assert await stopped_writing(), "The worker exited but its child is still writing"
        expected = "finalized" if stop == "normal" else "uncertain"
        assert (await service.status("alice", record["id"]))["status"] == expected
        if stop == "normal":
            assert observations["publication"]
            assert (await service.results("alice", record["id"]))["outcome"] == "success"
        if stop == "shutdown":
            assert (await service.status("alice", queued["id"]))["status"] == APPROVED
            service = Service(service.config)
            dispatcher = service.dispatcher = Dispatcher(service)
            dispatchers.append(dispatcher)
            monkeypatch.setattr(dispatcher, "spawn", spawn)
            await dispatcher.start()
        async def next_finished():
            return (await service.status("alice", queued["id"]))["status"] == "finalized"
        await wait_for(next_finished)
        assert observations["next_job"]
        assert len(processes) == 2  # The first operation was never retried.
    finally:
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        child_pid = context.workspace / "child.pid"
        if child_pid.exists():
            try:
                os.kill(int(child_pid.read_text()), signal.SIGKILL)
            except OSError:
                pass
        for process in processes:
            await process.wait()
        await asyncio.gather(*(item.stop() for item in dispatchers), return_exceptions=True)
        for item in dispatchers:
            if item.lock_file:
                item.lock_file.close()


@posix_only
@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["descendant", "transient_denial", "denial", "stuck", "leader"])
async def test_termination_confirms_group_exit_and_bounds_failures(monkeypatch, case):
    clock, signals = [0], []
    process = SimpleNamespace(pid=123456, returncode=0, wait=AsyncMock(return_value=0))
    if case == "leader":
        async def never_reaped():
            await asyncio.Future()
        process.wait = AsyncMock(side_effect=never_reaped)

    def killpg(pid, sig):
        assert pid == process.pid
        signals.append(sig)
        if case == "denial" or (case == "transient_denial" and len(signals) < 3):
            raise PermissionError("signal denied")
        if (case in {"leader", "transient_denial"}
                or (case == "descendant" and signals.count(signal.SIGKILL) == 3)):
            raise ProcessLookupError

    async def tick(seconds):
        clock[0] += 1
    monkeypatch.setattr(dispatcher_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(dispatcher_module, "asyncio", SimpleNamespace(sleep=tick, wait_for=asyncio.wait_for))
    monkeypatch.setattr(dispatcher_module.os, "killpg", killpg, raising=False)
    error = {"denial": PermissionError, "stuck": RuntimeError, "leader": TimeoutError}.get(case)
    if error:
        with pytest.raises(error):
            await Dispatcher.terminate(process)
    else:
        await Dispatcher.terminate(process)
        process.wait.assert_awaited_once()
    if case == "descendant":
        assert signals.count(signal.SIGKILL) == 3
    elif case == "transient_denial":
        assert len(signals) == 3


@posix_only
@pytest.mark.asyncio
async def test_cleanup_denial_stays_unready_and_shutdown_cleans_other_jobs(tmp_path, monkeypatch, caplog):
    app = create_app(config(tmp_path, max_workers=3, shutdown_seconds=1))
    service, dispatcher = app.state.service, app.state.service.dispatcher
    first, _ = await job(service, "alice")
    other, _ = await job(service, "bob")
    launched = fake_workers(dispatcher, monkeypatch, blocked=True)
    killpg = os.killpg
    processes = []
    await dispatcher.start()
    try:
        async def started():
            return all(identifier in dispatcher.active and dispatcher.active[identifier].process
                       for identifier in (first["id"], other["id"]))
        await wait_for(started)
        processes = [dispatcher.active[item["id"]].process for item in (first, other)]
        def denied(pid, sig):
            if pid == processes[0].pid:
                raise PermissionError("private operator diagnostic")
            return killpg(pid, sig)
        monkeypatch.setattr(os, "killpg", denied)
        result = await dispatcher.cancel("alice", first["id"])
        assert result["status"] == "uncertain" and result["cancel_requested"]
        assert result["error"] == "Worker cleanup failed; ask the operator to restart the service."
        saved = await service.journal("alice").get(first["id"])
        assert saved["error"] == (
            "Worker cleanup failed; surviving processes possible. Cancelled during invocation."
        )
        running = dispatcher.active[first["id"]]
        assert running.cleanup_failed and running.task.done() and running.task.exception() is None
        assert "job_cleanup_failed owner=alice job=" + first["id"] in caplog.text
        assert "job_collection_failed owner=alice" not in caplog.text

        # Simulate an operator approval after failure, with spare global capacity.
        queued, _ = await job(service, "carol")
        passed, dispatch = asyncio.Event(), dispatcher.dispatch
        async def next_pass():
            await dispatch()
            passed.set()
        monkeypatch.setattr(dispatcher, "dispatch", next_pass)
        dispatcher.wake.set()
        await asyncio.wait_for(passed.wait(), timeout=2)
        assert not service.ready and queued["id"] not in launched
        assert queued["id"] not in dispatcher.active
        assert (await service.status("carol", queued["id"]))["status"] == APPROVED
        with pytest.raises(ValueError, match="not ready"):
            await service.call("carol", "run_ase", {})
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            assert (await client.get("/readyz")).status_code == 503
        with pytest.raises(RuntimeError, match="supervisor cleanup"):
            await dispatcher.stop()
        assert processes[1].returncode is not None
        assert (await service.status("bob", other["id"]))["status"] == "uncertain"
        assert dispatcher.lock_file is not None and not dispatcher.lock_file.closed
        with pytest.raises(BlockingIOError):
            await Dispatcher(Service(service.config)).start()
    finally:
        monkeypatch.setattr(os, "killpg", killpg)
        for process in processes:
            try:
                killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            await process.wait()
        await asyncio.gather(dispatcher.stop(), return_exceptions=True)
        if dispatcher.lock_file:
            dispatcher.lock_file.close()


@pytest.mark.asyncio
async def test_catalog_failure_retries_without_repeating_calculation(tmp_path, monkeypatch):
    service = Service(config(tmp_path))
    record, _ = await job(service, "alice", state=RETURNED)
    publish = service.files.publish
    def unavailable(*args, **kwargs):
        raise OSError("storage unavailable")
    monkeypatch.setattr(service.files, "publish", unavailable)
    with pytest.raises(OSError, match="storage unavailable"):
        await service.finalize(record)
    assert (await service.status("alice", record["id"]))["status"] == "finalized"
    monkeypatch.setattr(service.files, "publish", publish)
    result = await service.results("alice", record["id"])
    assert result["result"] == {"energy": 1} and len(result["artifacts"]) == 1


@posix_only
@pytest.mark.asyncio
async def test_failed_dispatch_iteration_keeps_dispatching(tmp_path, monkeypatch):
    service = Service(config(tmp_path))
    dispatcher = service.dispatcher = Dispatcher(service)
    record, _ = await job(service, "alice")
    launched = fake_workers(dispatcher, monkeypatch)
    records, failures = service.store.records, []
    async def flaky(**kwargs):
        if kwargs.get("state") == APPROVED and not failures:
            failures.append(True)
            raise sqlite3.OperationalError("database is locked")
        return await records(**kwargs)
    monkeypatch.setattr(service.store, "records", flaky)
    await dispatcher.start()
    try:
        async def not_ready():
            return not service.ready
        await wait_for(not_ready)
        assert (await service.status("alice", record["id"]))["status"] == APPROVED
        with pytest.raises(ValueError, match="not ready"):
            await service.put_file("alice", "later.xyz", WATER)
        async def done():
            return (await service.status("alice", record["id"]))["status"] == "finalized"
        await wait_for(done)
        assert failures and launched == [record["id"]] and service.ready
    finally:
        await dispatcher.stop()
    assert not service.ready


@pytest.mark.asyncio
async def test_deferred_publication_never_adopts_replaced_outputs(tmp_path):
    record, context = await job(Service(config(tmp_path)), "alice", state=RETURNED)
    full = Service(config(tmp_path, max_storage_bytes=Service(config(tmp_path)).files.usage("alice")))
    with pytest.raises(QuotaExceeded):
        await full.finalize(record)
    (context.workspace / "out.json").write_text('{"energy": 2}')
    service = Service(config(tmp_path))
    with pytest.raises(ScopedExecutionError):
        await service.results("alice", record["id"])
    assert (await service.status("alice", record["id"]))["status"] == "returned"
    assert not service.files.list("alice")


@pytest.mark.asyncio
@pytest.mark.parametrize("number", ["NaN", "Infinity", "1e999"])
async def test_nonfinite_inline_result_is_a_saved_error(tmp_path, number):
    service = Service(config(tmp_path))
    service.ready = True
    await service.put_file("alice", "result.json", '{"energy": ' + number + '}')
    result = await service.call("alice", "extract_output_json", {"json_file": "result.json"})
    assert result["status"] == "finalized" and result["outcome"] == "error"
    assert result["result"]["error_type"] == "ValueError"
    assert await service.results("alice", result["batch_id"]) == result
    json.dumps(result, allow_nan=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase,recovery_fails,state", [
    ("record_return", False, "uncertain"),
    ("record_return", True, "invoking"),
    ("bind_outputs", False, "returned"),
    ("publish", False, "finalized"),
])
async def test_inline_collection_preserves_durable_state_and_original_error(
    tmp_path, monkeypatch, phase, recovery_fails, state,
):
    service = Service(config(tmp_path))
    service.ready = True
    await service.put_file("alice", "result.json", '{"energy": 1}')
    original = OSError("collection unavailable")

    async def unavailable(*args, **kwargs):
        raise original
    def publish_unavailable(*args, **kwargs):
        raise original
    async def reconciliation_unavailable(*args, **kwargs):
        raise OSError("reconciliation unavailable")
    if phase == "publish":
        monkeypatch.setattr(service.files, phase, publish_unavailable)
    else:
        monkeypatch.setattr(OperationJournal, phase, unavailable)
    if recovery_fails:
        monkeypatch.setattr(OperationJournal, "uncertain", reconciliation_unavailable)
    with pytest.raises(OSError) as error:
        await service.call("alice", "extract_output_json", {"json_file": "result.json"})
    assert error.value is original
    records = await service.store.records()
    assert len(records) == 1 and records[0]["state"] == state


@pytest.mark.asyncio
async def test_output_binding_requires_a_returned_operation(tmp_path):
    record, context = await job(Service(config(tmp_path)), "alice")
    with pytest.raises(ScopedExecutionError, match="Only returned"):
        await context.journal.bind_outputs(context, record)
    assert (await context.journal.get(record["id"]))["state"] == APPROVED
