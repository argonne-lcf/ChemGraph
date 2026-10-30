"""The operation state machine at its durable boundaries, with real disk restarts."""

import asyncio
import copy
import shutil
import time

import pytest
from langchain_core.messages import ToolMessage

from chemgraph.runtime import (
    ApprovalRequired, FileOperationStore, OperationConflict, OperationJournal,
    ScopedExecutionError,
)
from chemgraph.runtime import artifacts as artifact_store
from chemgraph.runtime import journal as journal_module
from chemgraph.runtime.artifact_audit import audit_records
from chemgraph.runtime.context import invocation, register_artifact, run_context
from chemgraph.runtime.journal import ARTIFACT_KEY
from tests.runtime_helpers import approve, make_context, run_ase_call


def writer(context, calls, content='{"energy": 1.0}', *, fail=None):
    """An invoke stage that writes and registers one output, like a native tool."""

    async def invoke(action, outputs):
        calls.append(action.digest)
        with run_context(context), invocation(action, outputs):
            (context.workspace / "energy.json").write_text(content)
            register_artifact("energy.json")
        if fail is not None:
            raise fail
        return ToolMessage(content="done", tool_call_id=action.operation_id)

    return invoke


async def approved(context, call=None):
    call = call or run_ase_call()
    record = await context.journal.propose(context, call)
    await approve(context, record)
    return call, record


def reopen(context, tmp_path):
    """Simulate a process restart: a new journal over the same durable store."""
    context.journal = OperationJournal(FileOperationStore(tmp_path / "journal"))
    return context.journal


@pytest.mark.asyncio
async def test_finalized_operation_replays_without_invocation(tmp_path):
    context = make_context(tmp_path)
    call, record = await approved(context)
    calls = []
    first = await context.journal.execute(context, call, writer(context, calls))
    artifact = first.additional_kwargs[ARTIFACT_KEY][0]
    assert artifact["path"].startswith(".chemgraph-artifacts/")
    assert artifact["source_path"] == "energy.json"
    (context.workspace / "energy.json").write_text("overwritten later")
    replay = await reopen(context, tmp_path).execute(context, call, writer(context, calls))
    assert replay == first and len(calls) == 1
    assert artifact_store.artifact_file(context.workspace, artifact).read_text() == '{"energy": 1.0}'


@pytest.mark.asyncio
async def test_replay_survives_workspace_relocation(tmp_path):
    context = make_context(tmp_path)
    call, _ = await approved(context)
    calls = []
    first = await context.journal.execute(context, call, writer(context, calls))
    moved = tmp_path / "restored"
    shutil.copytree(context.workspace, moved)
    shutil.rmtree(context.workspace)
    context.workspace = moved
    assert await reopen(context, tmp_path).execute(context, call, writer(context, calls)) == first
    artifact = first.additional_kwargs[ARTIFACT_KEY][0]
    artifact_store.artifact_file(moved, artifact).write_text("corrupt")
    with pytest.raises(ScopedExecutionError, match="restore it from backup"):
        await context.journal.execute(context, call, writer(context, calls))
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_tool_exception_is_a_returned_outcome(tmp_path):
    context = make_context(tmp_path)
    call, record = await approved(context)
    calls = []
    result = await context.journal.execute(
        context, call, writer(context, calls, fail=RuntimeError("SCF diverged")))
    assert result.status == "error" and "SCF diverged" in result.content
    assert (await context.journal.get(record["id"]))["state"] == "finalized"
    # Outputs registered before the failure are still published.
    assert [a["source_path"] for a in result.additional_kwargs[ARTIFACT_KEY]] == ["energy.json"]
    audit = audit_records(await context.journal.store.records(), workspace=context.workspace,
                         thread_id=context.thread_id, include_inputs=True)
    assert audit["outcome"] == "verified" and audit["counts"]["checked"] == 2
    assert audit["records"][0]["execution_outcome"] == "unknown"
    assert audit["records"][0]["scientific_acceptance"] == "unassessed" and len(calls) == 1
    # The workspace is not blocked by an observed failure.
    other, _ = await approved(context, run_ase_call("second"))
    assert (await context.journal.execute(context, other, writer(context, calls))).status != "error"


@pytest.mark.asyncio
async def test_crash_after_intent_is_uncertain_until_reconciled(tmp_path):
    class Crash(BaseException):
        pass

    context = make_context(tmp_path)
    call, record = await approved(context)
    calls = []
    with pytest.raises(Crash):
        await context.journal.execute(context, call, writer(context, calls, fail=Crash()))
    journal = reopen(context, tmp_path)
    assert [r["id"] for r in await journal.recover()] == [record["id"]]
    with pytest.raises(ScopedExecutionError, match="uncertain"):
        await journal.execute(context, call, writer(context, calls))
    other, _ = await approved(context, run_ase_call("second"))
    with pytest.raises(ScopedExecutionError, match="reconcile it first"):
        await journal.execute(context, other, writer(context, calls))
    assert len(calls) == 1

    reconciled = await journal.confirm_nonexecution(
        record["id"], decided_by="operator", evidence="process log shows no start")
    assert reconciled["state"] == "proposed" and not journal.authorized(reconciled)
    with pytest.raises(ScopedExecutionError, match="no valid approval"):
        await journal.execute(context, call, writer(context, calls))
    await approve(context, reconciled)
    await journal.execute(context, call, writer(context, calls))
    assert len(calls) == 2
    events = [item["event"] for item in (await journal.get(record["id"]))["history"]]
    assert events == ["decision", "restart", "nonexecution", "decision"]


@pytest.mark.asyncio
async def test_abandoned_operation_is_not_repeated(tmp_path):
    context = make_context(tmp_path)
    call, record = await approved(context)
    calls = []
    await context.journal.abandon(record["id"], decided_by="operator", reason="node lost")
    result = await context.journal.execute(context, call, writer(context, calls))
    assert result.status == "error" and "abandoned" in result.content
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("module, target", [
    (journal_module, "output_manifest"),
    (artifact_store, "publish_file"),
    (artifact_store, "sync_directory"),
])
async def test_finalization_retry_uses_saved_manifest(tmp_path, monkeypatch, module, target):
    context = make_context(tmp_path)
    call, record = await approved(context)
    calls = []

    def unavailable(*args, **kwargs):
        raise OSError("artifact volume unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(module, target, unavailable)
        with pytest.raises(ScopedExecutionError, match="artifacts failed"):
            await context.journal.execute(context, call, writer(context, calls))
    journal = reopen(context, tmp_path)
    saved = await journal.get(record["id"])
    assert saved["state"] == "returned" and saved["returned"]["registered"] == ["energy.json"]
    other, _ = await approved(context, run_ase_call("second"))
    with pytest.raises(ScopedExecutionError, match="reconcile it first"):
        await journal.execute(context, other, writer(context, calls))
    # Retrying never rediscovers files absent from the persisted manifest.
    (context.workspace / "unrelated.json").write_text("{}")
    result = await journal.execute(context, call, writer(context, calls))
    artifacts = result.additional_kwargs[ARTIFACT_KEY]
    assert result.content == "done" and [a["source_path"] for a in artifacts] == ["energy.json"]
    assert artifact_store.artifact_file(context.workspace, artifacts[0]).read_text() == '{"energy": 1.0}'
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_changed_input_after_approval_blocks_without_invocation(tmp_path):
    context = make_context(tmp_path)
    call, record = await approved(context)
    (context.workspace / record["prepared"]["inputs"][0]["path"]).write_text("tampered")
    calls = []
    result = await context.journal.execute(context, call, writer(context, calls))
    assert result.status == "error" and "input changed" in result.content
    assert calls == [] and (await context.journal.get(record["id"]))["state"] == "blocked"


@pytest.mark.asyncio
async def test_decisions_bind_digest_and_expiry(tmp_path):
    context = make_context(tmp_path)
    call = run_ase_call()
    record = await context.journal.propose(context, call)
    assert await context.journal.propose(context, call) == record
    with pytest.raises(ValueError, match="digest"):
        await context.journal.decide(record["id"], digest="0" * 64, approve=True, decided_by="op")
    expired = await approve(context, record, expires_at=time.time() - 1)
    assert not context.journal.authorized(expired)
    renewed = await approve(context, expired, expires_at=time.time() + 60)
    assert context.journal.authorized(renewed)
    changed = run_ase_call(output="other.json")
    with pytest.raises(ScopedExecutionError, match="reused with different arguments"):
        await context.journal.propose(context, changed)


@pytest.mark.asyncio
async def test_rejection_during_validation_prevents_invocation(tmp_path, monkeypatch):
    context = make_context(tmp_path)
    call, record = await approved(context)
    validate = journal_module.validate

    def reject_meanwhile(*args):
        validate(*args)
        # Another process records a rejection while the preparation is re-checked.
        operator = OperationJournal(FileOperationStore(tmp_path / "journal"))
        asyncio.run(operator.decide(record["id"], digest=record["prepared"]["digest"],
                                    approve=False, decided_by="operator", message="Wrong output."))

    monkeypatch.setattr(journal_module, "validate", reject_meanwhile)
    calls = []
    with pytest.raises(ApprovalRequired):
        await context.journal.execute(context, call, writer(context, calls))
    saved = await context.journal.get(record["id"])
    assert calls == [] and saved["state"] == "rejected"
    assert [event["approve"] for event in saved["history"]] == [True, False]


@pytest.mark.asyncio
async def test_expiry_during_validation_prevents_invocation(tmp_path, monkeypatch):
    context = make_context(tmp_path)
    call = run_ase_call()
    record = await context.journal.propose(context, call)
    await approve(context, record, expires_at=time.time() + 60)
    validate, clock = journal_module.validate, time.time

    def expire_meanwhile(*args):
        validate(*args)
        monkeypatch.setattr(journal_module.time, "time", lambda: clock() + 120)

    monkeypatch.setattr(journal_module, "validate", expire_meanwhile)
    calls = []
    with pytest.raises(ApprovalRequired, match="expired"):
        await context.journal.execute(context, call, writer(context, calls))
    assert calls == [] and (await context.journal.get(record["id"]))["state"] == "approved"


@pytest.mark.asyncio
async def test_saves_are_compare_and_set(tmp_path, monkeypatch):
    context = make_context(tmp_path)
    record = await context.journal.propose(context, run_ase_call())
    stale = copy.deepcopy(record)
    await approve(context, record)
    store = context.journal.store
    with pytest.raises(OperationConflict):
        await store.put({**stale, "state": "invoking"}, expected_revision=stale["revision"])

    async def stale_get(identifier):
        return copy.deepcopy(stale)

    monkeypatch.setattr(store, "get", stale_get)
    with pytest.raises(ValueError, match="reload"):
        await context.journal.decide(record["id"], digest=stale["prepared"]["digest"],
                                     approve=False, decided_by="operator")
