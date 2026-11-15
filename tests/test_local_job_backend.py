"""Durable backend contracts, independent of the service and job journal."""

import asyncio
from dataclasses import asdict

import pytest

from chemgraph.execution.jobs import LocalJobBackend, atomic_json
from chemgraph.runtime.artifacts import file_identity
from chemgraph.runtime.context import run_context, scoped_path
from chemgraph.runtime.policy import PreparedAction, prepare
from tests.runtime_helpers import make_context, run_ase_call


@pytest.mark.asyncio
async def test_backend_dispatch_marker_survives_reopen(tmp_path):
    backend = LocalJobBackend(tmp_path)
    record = await backend.submit("a-0", {"action": {"digest": "fingerprint"}})
    record.update(status="dispatching", dispatching_at=1)
    atomic_json(tmp_path / "a-0" / "job.json", record)
    restored = LocalJobBackend(tmp_path)
    assert (await restored.status("a-0"))["status"] == "unknown"
    assert (await restored.dispatch("a-0"))["status"] == "unknown"


def test_v1_preparations_roundtrip_and_job_tree_is_reserved(tmp_path):
    context = make_context(tmp_path)
    action = prepare(context, run_ase_call())
    legacy = action.to_dict()
    assert legacy["version"] == 1 and "execution" not in legacy
    assert PreparedAction.from_dict(legacy).to_dict() == legacy
    with run_context(context), pytest.raises(PermissionError, match="Internal"):
        scoped_path(".chemgraph-jobs/foreign/output.json")
    (context.workspace / "escape").symlink_to(tmp_path)
    with run_context(context), pytest.raises(PermissionError, match="outside"):
        scoped_path("escape/output.json")


def worker_spec(context, call, action, job_id="a-0"):
    policy = asdict(context.policy)
    for key in ("tools", "calculators", "helpers"):
        policy[key] = sorted(policy[key])
    policy["model_roots"] = list(policy["model_roots"])
    return {"job_id": job_id, "operation_id": action.operation_id,
            "owner": context.owner, "thread_id": context.thread_id, "task_id": context.task_id,
            "workspace": str(context.workspace), "policy": policy,
            "action": action.to_dict(), "call": call}


@pytest.mark.asyncio
async def test_idempotent_submission_and_queued_cancel_survive_reopen(tmp_path):
    backend = LocalJobBackend(tmp_path)
    spec = {"action": {"digest": "first"}}
    first = await backend.submit("a-0", spec)
    assert await backend.submit("a-0", spec) == first
    with pytest.raises(ValueError, match="different inputs"):
        await backend.submit("a-0", {"action": {"digest": "changed"}})
    await backend.submit("b-0", spec)
    assert [r["job_id"] for r in await backend.records()] == ["a-0", "b-0"]
    await backend.cancel("a-0")
    restored = LocalJobBackend(tmp_path)
    assert (await restored.dispatch("a-0"))["status"] == "cancelled"
    assert not (tmp_path / "a-0/started.json").exists()
    assert (await restored.lookup("b-0"))["status"] == "queued"


@pytest.mark.asyncio
async def test_acknowledged_queue_record_cannot_dispatch(tmp_path):
    backend = LocalJobBackend(tmp_path)
    await backend.submit("a-0", {"action": {"digest": "first"}})
    await backend.acknowledge("a-0", promoted=False)
    await backend.dispatch("a-0")
    assert not (tmp_path / "a-0/started.json").exists()
    assert "dispatching_at" not in await backend.lookup("a-0")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_real_worker_writes_recoverable_completion_envelope(tmp_path, failure):
    context = make_context(tmp_path)
    if failure:
        (context.workspace / "h2.xyz").write_text("1\nUnsupported EMT element\nHe 0 0 0\n")
    call = run_ase_call()
    action = prepare(context, call)
    spec = worker_spec(context, call, action)
    backend = LocalJobBackend(tmp_path / "backend")
    try:
        await backend.submit("a-0", spec)
        await backend.dispatch("a-0")
        async with asyncio.timeout(30):
            while (await backend.status("a-0"))["status"] not in {"succeeded", "failed", "unknown"}:
                await asyncio.sleep(0.01)
        assert (await backend.status("a-0"))["status"] == ("failed" if failure else "succeeded")
    finally:
        await backend.close()
    # A new backend has no Future, but the durable envelope still proves the result.
    restored = LocalJobBackend(tmp_path / "backend")
    envelope = await restored.fetch("a-0")
    assert envelope["result"]["status"] == ("failure" if failure else "success")
    assert (tmp_path / "backend/a-0/started.json").exists()
    assert (tmp_path / "backend/a-0/exited.json").exists()
    if not failure:
        output = envelope["outputs"][0]
        assert file_identity(context.workspace / output["source_path"]) == {
            k: output[k] for k in ("sha256", "size_bytes")}
        (context.workspace / output["source_path"]).write_text("corrupted")
        with pytest.raises(ValueError, match="changed"):
            await restored.fetch("a-0")
