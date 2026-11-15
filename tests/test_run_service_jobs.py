"""Service integration with real workers, scientific outputs, and injected faults."""

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from chemgraph.execution.jobs import run_local_job
from chemgraph.runtime.artifacts import file_identity
from chemgraph.runtime.context import run_context
from chemgraph.runtime.journal import decode_result
from tests.runtime_helpers import H2, ai, run_ase_call
from tests.test_run_service import approve_all, service, until


def controlled_job(spec, receipt_directory):
    """A real worker with a deterministic test-controlled start boundary."""
    directory = Path(receipt_directory)
    (directory / "ready").write_text("ready")
    deadline = time.monotonic() + 15
    while not (directory / "release").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    run_local_job(spec, receipt_directory)


@pytest.mark.asyncio
async def test_dispatch_intent_is_unknown_and_new_attempt_uses_original_snapshot(tmp_path):
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="Recovered")]) as api:
        actual_dispatch = api.backend.dispatch

        async def no_dispatch(*args):
            return None

        api.backend.dispatch = no_dispatch
        session = await api.create_session()
        original = Path(session.workspace) / "h2.xyz"
        original.write_text(H2)
        run = await api.start_run(session.id, "Energy")
        await approve_all(api, run.id)
        await until(lambda: api.get_run(run.id).status == "waiting_for_job")
        job = (await api.list_jobs(run.id))[0]
        record = await api.backend.lookup(job.id)
        record.update(status="dispatching", dispatching_at=time.time())
        await api.backend._save(record)
        await until(lambda: api.get_run(run.id).status == "recovery_required")
        assert (await api.list_jobs(run.id))[0].status == "unknown"
        await asyncio.sleep(0.05)
        assert not (api.backend._directory(job.id) / "started.json").exists()
        original.write_text("1\nChanged logical input\nHe 0 0 0\n")
        await api.reconcile(run.id, job.operation_id, action="confirm_nonexecution",
                            evidence="Injected dispatch marker; no pool submission occurred.")
        paused = await api.wait(run.id)
        assert paused.status == "waiting_for_user", paused
        new_action = paused.interrupts[0].payload["prepared"]
        assert new_action["execution"]["attempt"] == 1
        assert new_action["inputs"][0]["sha256"] != file_identity(original)["sha256"]
        api.backend.dispatch = actual_dispatch
        await approve_all(api, run.id)
        assert (await api.wait(run.id)).status == "completed"
        jobs = await api.list_jobs(run.id)
        assert len(jobs) == 2
        assert [j.status for j in jobs] == ["unknown", "succeeded"]
        artifact = (await api.list_artifacts(run.id))[0]
        result = json.loads((await api.resolve_artifact(run.id, artifact.id)).read_text())
        assert result["potential_energy"] == pytest.approx(1.5461139951321488)


@pytest.mark.asyncio
async def test_running_cancel_archives_without_promoting(tmp_path, monkeypatch):
    monkeypatch.setattr("chemgraph.execution.jobs.run_local_job", controlled_job)
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="Must not resume")]) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        await approve_all(api, run.id)
        await until(lambda: bool(list((tmp_path / "jobs").glob("*/ready"))))
        job = (await api.list_jobs(run.id))[0]
        directory = api.backend._directory(job.id)
        try:
            await api.cancel_run(run.id)
            assert api.get_run(run.id).status == "cancelled"
        finally:
            (directory / "release").write_text("go")
        await until(lambda: (directory / "result.json").exists())
        async with asyncio.timeout(30):
            while not await api.list_artifacts(run.id):
                await asyncio.sleep(0.01)
        artifacts = await api.list_artifacts(run.id)
        assert artifacts and all(not a.promoted for a in artifacts)
        assert not (Path(session.workspace) / "energy.json").exists()
        assert api.get_run(run.id).status == "cancelled"
        assert (await api.list_jobs(run.id))[0].promoted is False
        assert (await api.resolve_artifact(run.id, artifacts[0].id)).is_file()


@pytest.mark.asyncio
async def test_queued_cancel_never_starts_worker(tmp_path):
    async with service(tmp_path, [ai(run_ase_call())]) as api:
        async def no_dispatch(*args):
            return None
        api.backend.dispatch = no_dispatch
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        await approve_all(api, run.id)
        await until(lambda: api.get_run(run.id).status == "waiting_for_job")
        await api.cancel_run(run.id)
        assert (await api.list_jobs(run.id))[0].status == "cancelled"
        assert not list((tmp_path / "jobs").glob("*/started.json"))


@pytest.mark.asyncio
async def test_publication_retry_never_recalculates(tmp_path, monkeypatch):
    from chemgraph.runtime import jobs
    actual = jobs.promote_outputs
    failures = []

    def fail_once(*args):
        if not failures:
            failures.append(True)
            raise OSError("Injected promotion interruption")
        return actual(*args)

    monkeypatch.setattr(jobs, "promote_outputs", fail_once)
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="Done")]) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        await approve_all(api, run.id)
        # Recovery may be exposed briefly; the monitor retries only publication.
        await until(lambda: api.get_run(run.id).status == "completed")
        assert failures == [True]
        assert len(await api.list_jobs(run.id)) == 1
        assert len(list((tmp_path / "jobs").glob("*/started.json"))) == 1


@pytest.mark.asyncio
async def test_opt_extract_then_vib_uses_logical_paths(tmp_path):
    extract = {"id": "read", "name": "extract_output_json", "args": {"json_file": "energy.json"}}
    responses = [ai(run_ase_call("opt", driver="opt", steps=20)), ai(extract),
                 ai(run_ase_call("vib", driver="vib", structure="h2_opt.traj", output="vib.json")),
                 AIMessage(content="Done")]
    async with service(tmp_path, responses) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Optimize then vibrate")
        for _ in range(3):
            await approve_all(api, run.id)
        completed = await api.wait(run.id)
        assert completed.status == "completed", completed
        artifacts = await api.list_artifacts(run.id)
        names = {a.source_path for a in artifacts}
        assert {"energy.json", "h2_opt.traj", "vib.json", "frequencies_h2_opt.csv"} <= names
        assert any(name.endswith(".traj") and "_vib." in name for name in names)
        # Compare against the existing synchronous producer on the same molecule.
        from chemgraph.schemas.ase_input import ASEInputSchema
        from chemgraph.tools.ase_core import run_ase_core
        from chemgraph.utils.artifacts import artifact_context
        synchronous = tmp_path / "synchronous"
        synchronous.mkdir()
        (synchronous / "h2.xyz").write_text(H2)
        with artifact_context(str(synchronous)):
            result = await asyncio.to_thread(
                run_ase_core, ASEInputSchema.model_validate(run_ase_call(driver="opt", steps=20)["args"]["params"]))
        assert result["status"] == "success"
        expected = json.loads((synchronous / "energy.json").read_text())
        actual = json.loads((Path(session.workspace) / "energy.json").read_text())
        import numpy as np
        from ase.io import read
        assert actual["potential_energy"] == pytest.approx(expected["potential_energy"])
        np.testing.assert_allclose(read(Path(session.workspace) / "h2_opt.traj").positions,
                                   read(synchronous / "h2_opt.traj").positions)


@pytest.mark.asyncio
async def test_vib_then_single_atom_removes_stale_logical_outputs(tmp_path):
    async with service(tmp_path, [ai(run_ase_call("vib", driver="vib")), AIMessage(content="First"),
                                 ai(run_ase_call("atom", driver="vib")), AIMessage(content="Second")]) as api:
        session = await api.create_session()
        structure = Path(session.workspace) / "h2.xyz"
        structure.write_text(H2)
        first = await api.start_run(session.id, "Vibrate H2")
        await approve_all(api, first.id)
        assert (await api.wait(first.id)).status == "completed"
        first_artifacts = await api.list_artifacts(first.id)
        assert (Path(session.workspace) / "frequencies_h2.csv").exists()
        assert list(Path(session.workspace).glob("h2_vib.*.traj"))
        structure.write_text("1\none atom\nH 0 0 0\n")
        second = await api.start_run(session.id, "Vibrate one atom")
        await approve_all(api, second.id)
        assert (await api.wait(second.id)).status == "completed"
        assert not (Path(session.workspace) / "frequencies_h2.csv").exists()
        assert not list(Path(session.workspace).glob("h2_vib.*.traj"))
        for artifact in first_artifacts:
            assert (await api.resolve_artifact(first.id, artifact.id)).exists()


@pytest.mark.asyncio
async def test_real_graph_premature_wake_cannot_supply_a_result(tmp_path):
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="Done")]) as api:
        actual_dispatch = api.backend.dispatch

        async def no_dispatch(*args):
            return None
        api.backend.dispatch = no_dispatch
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        await approve_all(api, run.id)
        await until(lambda: api.get_run(run.id).status == "waiting_for_job")
        entry = api._entries[session.id]
        before = await api.list_jobs(run.id)
        async with entry.lock:
            with run_context(entry.context):
                result = await entry.session.resume({i.id: {"potential_energy": -999}
                                                    for i in api.get_run(run.id).interrupts})
            await api._record_result(api.get_run(run.id), result)
        assert api.get_run(run.id).status == "waiting_for_job"
        assert [j.id for j in await api.list_jobs(run.id)] == [j.id for j in before]
        api.backend.dispatch = actual_dispatch
        assert (await api.wait(run.id)).status == "completed"


@pytest.mark.asyncio
async def test_expired_queued_approval_never_dispatches(tmp_path):
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="Blocked")]) as api:
        actual_dispatch = api.backend.dispatch

        async def no_dispatch(*args):
            return None
        api.backend.dispatch = no_dispatch
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        await approve_all(api, run.id)
        await until(lambda: api.get_run(run.id).status == "waiting_for_job")
        job = (await api.list_jobs(run.id))[0]
        journal = api._entries[session.id].context.journal
        record = await journal.get(job.operation_id)
        record["decision"]["expires_at"] = 0
        await journal._put(record, record["state"])
        api.backend.dispatch = actual_dispatch
        assert (await api.wait(run.id)).status == "completed"
        assert (await api.list_jobs(run.id))[0].status == "cancelled"
        assert not list((tmp_path / "jobs").glob("*/started.json"))


@pytest.mark.asyncio
async def test_unknown_job_guards_only_its_own_session(tmp_path):
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="Done")]) as api:
        actual_dispatch = api.backend.dispatch

        async def no_dispatch(*args):
            return None
        api.backend.dispatch = no_dispatch
        first_session = await api.create_session()
        (Path(first_session.workspace) / "h2.xyz").write_text(H2)
        first = await api.start_run(first_session.id, "Energy")
        await approve_all(api, first.id)
        await until(lambda: api.get_run(first.id).status == "waiting_for_job")
        job = (await api.list_jobs(first.id))[0]
        record = await api.backend.lookup(job.id)
        record["status"] = "dispatching"
        await api.backend._save(record)
        await until(lambda: api.get_run(first.id).status == "recovery_required")
        api.backend.dispatch = actual_dispatch
        second_session = await api.create_session()
        (Path(second_session.workspace) / "h2.xyz").write_text(H2)
        second = await api.start_run(second_session.id, "Energy")
        await approve_all(api, second.id)
        assert (await api.wait(second.id)).status == "completed"
        assert api.get_run(first.id).status == "recovery_required"


@pytest.mark.asyncio
async def test_calculator_failure_is_a_failed_job(tmp_path):
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="EMT cannot handle He")]) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text("1\nUnsupported EMT element\nHe 0 0 0\n")
        run = await api.start_run(session.id, "Energy")
        await approve_all(api, run.id)
        assert (await api.wait(run.id)).status == "completed"
        assert (await api.list_jobs(run.id))[0].status == "failed"
        record = (await api._entries[session.id].context.journal.store.records())[0]
        assert decode_result(record["result"]["message"]).status == "error"


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["returned", "finalized", "unacknowledged"])
async def test_restart_after_completion_does_not_repeat_calculation(tmp_path, boundary):
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="Unused")]) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        await approve_all(api, run.id)
        await until(lambda: api.get_run(run.id).status == "waiting_for_job")
        journal = api._entries[session.id].context.journal
        if boundary == "returned":
            async def stop_publication(*args, **kwargs):
                raise OSError("Injected unavailable publication storage")
            journal.finalize_job = stop_publication
        else:
            async def no_wake(*args):
                return None
            api._maybe_resume = no_wake
            if boundary == "unacknowledged":
                async def no_ack(*args, **kwargs):
                    return None
                api.backend.acknowledge = no_ack
        async with asyncio.timeout(30):
            while (await journal.store.records())[0]["state"] != (
                    "finalized" if boundary == "unacknowledged" else boundary):
                await asyncio.sleep(0.01)
        jobs = await api.list_jobs(run.id)
    async with service(tmp_path, [AIMessage(content="Restored")]) as api:
        assert (await api.wait(run.id)).status == "completed"
        assert [j.id for j in await api.list_jobs(run.id)] == [j.id for j in jobs]
        assert len(list((tmp_path / "jobs").glob("*/started.json"))) == 1
        history = api.history.get_session(session.id).messages
        assert len([m for m in history if m.role == "human"]) == 1
        assert len([m for m in history if m.role == "tool"]) == 1
        assert (await api.backend.lookup(jobs[0].id))["published"] is True


@pytest.mark.asyncio
async def test_submission_intent_without_backend_record_can_be_reconciled(tmp_path):
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="Recovered")]) as api:
        actual_submit = api.backend.submit

        async def fail_before_ack(*args):
            raise OSError("Injected crash before backend acknowledgment")
        api.backend.submit = fail_before_ack
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        await approve_all(api, run.id)
        assert (await api.wait(run.id)).status == "recovery_required"
        job = (await api.list_jobs(run.id))[0]
        assert job.status == "unknown"
        api.backend.submit = actual_submit
        await api.reconcile(run.id, job.operation_id, action="confirm_nonexecution",
                            evidence="Test backend never received the submission.")
        await approve_all(api, run.id)
        assert (await api.wait(run.id)).status == "completed"
        assert len(list((tmp_path / "jobs").glob("*/started.json"))) == 1


@pytest.mark.asyncio
async def test_cancel_during_publication_staging_never_promotes(tmp_path, monkeypatch):
    from chemgraph.runtime import jobs
    actual = jobs.stage_promotions
    entered, release = threading.Event(), threading.Event()

    def staged_then_wait(*args):
        result = actual(*args)
        entered.set()
        if not release.wait(15):
            raise TimeoutError("Test did not release staging")
        return result

    monkeypatch.setattr(jobs, "stage_promotions", staged_then_wait)
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="Must not resume")]) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        await approve_all(api, run.id)
        await until(entered.is_set)
        cancellation = asyncio.create_task(api.cancel_run(run.id))
        try:
            await until(lambda: api.get_session(session.id).status == "cancelled")
        finally:
            release.set()
        await cancellation
        assert api.get_run(run.id).status == "cancelled"
        assert not (Path(session.workspace) / "energy.json").exists()
        assert all(not a.promoted for a in await api.list_artifacts(run.id))


@pytest.mark.asyncio
async def test_reconciled_queued_attempt_cannot_dispatch_again(tmp_path):
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="Recovered")]) as api:
        actual_submit, actual_dispatch = api.backend.submit, api.backend.dispatch

        async def persist_then_fail(*args):
            await actual_submit(*args)
            raise OSError("Injected fsync failure after queue record rename")
        api.backend.submit = persist_then_fail
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        await approve_all(api, run.id)
        assert (await api.wait(run.id)).status == "recovery_required"
        old = (await api.list_jobs(run.id))[0]
        assert old.status == "queued"
        api.backend.submit = actual_submit
        recovered = await api.reconcile(run.id, old.operation_id, action="confirm_nonexecution",
                                        evidence="Queue was written; dispatch was never attempted.")
        assert recovered.id == run.id
        assert (await api.backend.status(old.id))["status"] == "cancelled"

        async def no_dispatch(*args):
            return None
        api.backend.dispatch = no_dispatch
        await approve_all(api, run.id)
        await until(lambda: api.get_run(run.id).status == "waiting_for_job")
        # Also exercise the service's attempt-identity guard on stale queue data.
        stale = await api.backend.lookup(old.id)
        stale.update(status="queued", published=False, cancel_requested=False)
        await api.backend._save(stale)
        api.backend.dispatch = actual_dispatch
        assert (await api.wait(run.id)).status == "completed"
        assert not (api.backend._directory(old.id) / "started.json").exists()
        jobs = await api.list_jobs(run.id)
        assert [job.status for job in jobs] == ["cancelled", "succeeded"]
