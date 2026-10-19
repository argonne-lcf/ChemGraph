"""Exercise durable boundaries and real process death without chemistry/network."""

import asyncio
import json
import os
import signal
import sqlite3
import sys
import uuid

from langchain_core.messages import ToolMessage
import pytest

from chemgraph.mcp.service.dispatcher import Dispatcher
from chemgraph.mcp.service.files import QuotaExceeded
from chemgraph.mcp.service.operations import Service
from chemgraph.runtime.context import ScopedExecutionError
from chemgraph.runtime.journal import APPROVED, INVOKING, RETURNED
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
        async def done():
            return (await service.status("alice", record["id"]))["status"] == "finalized"
        await wait_for(done)
        assert failures and launched == [record["id"]] and service.ready
    finally:
        await dispatcher.stop()


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
