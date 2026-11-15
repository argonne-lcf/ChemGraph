"""Exercise the Python API against real graphs, SQLite and spawned EMT workers."""

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from chemgraph.service import RunService
from tests.runtime_helpers import H2, ai, run_ase_call
from tests.test_main_agent import _ScriptedChatModel


def service(root, responses, **kwargs):
    return RunService(root, model_factory=lambda _: _ScriptedChatModel(responses=list(responses)),
                      configuration_id="scripted-v1", poll_interval=0.01, **kwargs)


async def until(predicate, timeout=30):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


async def approve_all(api, run_id):
    run = await api.wait(run_id)
    assert run.status == "waiting_for_user", run
    for item in run.interrupts:
        if item.kind == "approval":
            await api.decide(run_id, item.operation_id, digest=item.payload["digest"], approve=True)
    return run


@pytest.mark.asyncio
async def test_generated_structure_job_and_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path / "ambient"))
    write = {"id": "structure", "name": "save_atomsdata_to_file", "args": {
        "atomsdata": {"numbers": [1, 1], "positions": [[0, 0, 0], [0, 0, 0.9]]}, "fname": "h2.xyz"}}
    responses = [ai(write), ai(run_ase_call()), AIMessage(content="Energy computed"),
                 AIMessage(content="Follow-up")]
    async with service(tmp_path / "data", responses) as api:
        session = await api.create_session()
        run = await api.start_run(session.id, "Build H2 and calculate its energy.")
        await approve_all(api, run.id)
        await approve_all(api, run.id)
        completed = await api.wait(run.id)
        assert completed.status == "completed", completed
        assert completed.assistant_response == "Energy computed"
        jobs = await api.list_jobs(run.id)
        assert len(jobs) == 1 and jobs[0].status == "succeeded" and jobs[0].promoted
        artifacts = await api.list_artifacts(run.id)
        assert {a.source_path for a in artifacts} == {"energy.json", "h2.xyz"}
        energy = next(a for a in artifacts if a.source_path == "energy.json")
        path = await api.resolve_artifact(run.id, energy.id)
        result = json.loads(path.read_text())
        assert result["potential_energy"] == pytest.approx(1.5461139951321488)
        assert (Path(session.workspace) / "energy.json").is_file()
        followup = await api.start_run(session.id, "Continue")
        assert (await api.wait(followup.id)).assistant_response == "Follow-up"
        path.write_text("corrupted")
        with pytest.raises(ValueError, match="changed"):
            await api.resolve_artifact(run.id, energy.id)
    assert not (tmp_path / "ambient").exists()


@pytest.mark.asyncio
async def test_parallel_jobs_same_output_have_distinct_immutable_results(tmp_path):
    async with service(tmp_path, [ai(run_ase_call("first"), run_ase_call("second")),
                                 AIMessage(content="Done")]) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Twice")
        await approve_all(api, run.id)
        completed = await api.wait(run.id)
        assert completed.status == "completed", completed
        jobs = await api.list_jobs(run.id)
        assert len(jobs) == 2 and all(j.status == "succeeded" for j in jobs)
        artifacts = await api.list_artifacts(run.id)
        assert len(artifacts) == 2 and len({a.path for a in artifacts}) == 2
        for artifact in artifacts:
            assert (await api.resolve_artifact(run.id, artifact.id)).is_file()


@pytest.mark.asyncio
async def test_clarification_is_not_an_approval_or_job_result(tmp_path):
    question = {"id": "ask", "name": "ask_human", "args": {"question": "Which molecule?"}}
    async with service(tmp_path, [ai(question), ai(run_ase_call()), AIMessage(content="Done")]) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Calculate")
        paused = await api.wait(run.id)
        assert paused.interrupts[0].kind == "clarification"
        with pytest.raises(ValueError):
            await api.respond(run.id, "stale", "H2")
        await api.respond(run.id, paused.interrupts[0].id, "H2")
        approved = await api.wait(run.id)
        with pytest.raises(ValueError):
            await api.respond(run.id, approved.interrupts[0].id, '{"potential_energy": -999}')
        await approve_all(api, run.id)
        assert (await api.wait(run.id)).status == "completed"


@pytest.mark.asyncio
async def test_queued_job_survives_restart_with_same_turn_identity(tmp_path):
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="Unused")]) as api:
        async def no_dispatch(*args):
            return None
        api.backend.dispatch = no_dispatch
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        await approve_all(api, run.id)
        await until(lambda: api.get_run(run.id).status == "waiting_for_job")
        before = await api.list_jobs(run.id)
        assert before[0].status == "queued"
    async with service(tmp_path, [AIMessage(content="Restored")]) as api:
        completed = await api.wait(run.id)
        assert completed.status == "completed", completed
        after = await api.list_jobs(run.id)
        assert [j.id for j in after] == [j.id for j in before]
        assert all(j.run_id == run.id for j in after)
        saved = api.history.get_session(session.id)
        assert len([m for m in saved.messages if m.role == "human"]) == 1


@pytest.mark.asyncio
async def test_cancelled_session_stays_terminal_after_reopen(tmp_path):
    async with service(tmp_path, [ai(run_ase_call())]) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        await api.wait(run.id)
        await api.cancel_run(run.id)
        assert api.history.get_session(session.id).status == "cancelled"
        with pytest.raises(RuntimeError):
            await api.start_run(session.id, "Continue")
    async with service(tmp_path, [AIMessage(content="New session works")]) as api:
        assert api.get_run(run.id).status == "cancelled"
        assert api.history.get_session(session.id).status == "cancelled"
        assert api.history.list_sessions()[0].status == "cancelled"
        with pytest.raises(RuntimeError):
            await api.retry_run(run.id)
        fresh = await api.create_session()
        new_run = await api.start_run(fresh.id, "Hello")
        assert (await api.wait(new_run.id)).status == "completed"


@pytest.mark.asyncio
async def test_exclusive_root_and_configuration_identity(tmp_path):
    async with service(tmp_path, [AIMessage(content="Done")]) as api:
        session = await api.create_session()
        run = await api.start_run(session.id, "Hello")
        await api.wait(run.id)
        with pytest.raises(RuntimeError, match="owns"):
            async with service(tmp_path, []):
                pass
    with pytest.raises(ValueError, match="configuration changed"):
        async with RunService(tmp_path, model_factory=lambda _: _ScriptedChatModel(responses=[]),
                              configuration_id="changed"):
            pass


@pytest.mark.asyncio
async def test_new_turn_intent_before_first_checkpoint_survives_restart(tmp_path):
    async with service(tmp_path, [AIMessage(content="First")]) as api:
        session = await api.create_session()
        first = await api.start_run(session.id, "First message")
        assert (await api.wait(first.id)).status == "completed"
        # Simulate losing the process after turn intent, before graph invocation.
        api._launch = lambda *args: None
        second = await api.start_run(session.id, "Second message")
    async with service(tmp_path, [AIMessage(content="Second")]) as api:
        restored = await api.wait(second.id)
        assert restored.assistant_response == "Second"
        assert [m.content for m in api.history.get_session(session.id).messages
                if m.role == "human"] == ["First message", "Second message"]


@pytest.mark.asyncio
async def test_mixed_review_and_clarification_retain_answer_across_restart(tmp_path):
    question = {"id": "ask", "name": "ask_human", "args": {"question": "Continue?"}}
    async with service(tmp_path, [ai(run_ase_call(), question)]) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        paused = await api.wait(run.id)
        assert {i.kind for i in paused.interrupts} == {"approval", "clarification"}
        clarification = next(i for i in paused.interrupts if i.kind == "clarification")
        await api.respond(run.id, clarification.id, "Yes, H2")
        assert not await api.list_jobs(run.id)
    async with service(tmp_path, [AIMessage(content="Done")]) as api:
        await approve_all(api, run.id)
        assert (await api.wait(run.id)).status == "completed"
        with pytest.raises(ValueError):
            await api.respond(run.id, clarification.id, "Stale answer")
        assert len(await api.list_jobs(run.id)) == 1


@pytest.mark.asyncio
async def test_cancel_before_graph_starts_is_readable_and_terminal(tmp_path):
    async with service(tmp_path, [AIMessage(content="Must not run")]) as api:
        session = await api.create_session()
        run = await api.start_run(session.id, "Hello")
        await api.cancel_run(run.id)
        assert api.get_run(run.id).status == "cancelled"
        assert api.history.get_session(session.id).status == "cancelled"
        assert not api.history.get_session(session.id).messages
    async with service(tmp_path, []) as api:
        assert api.get_run(run.id).status == "cancelled"
        assert api.history.get_session(session.id).status == "cancelled"


@pytest.mark.asyncio
async def test_concurrent_decision_and_cancel_never_reopen_session(tmp_path):
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="Done")]) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        paused = await api.wait(run.id)
        pending = paused.interrupts[0]
        await asyncio.gather(
            api.decide(run.id, pending.operation_id, digest=pending.payload["digest"], approve=True),
            api.cancel_run(run.id), return_exceptions=True,
        )
        assert api.get_run(run.id).status == "cancelled"
        with pytest.raises(RuntimeError):
            await api.start_run(session.id, "Continue")
        assert not list((tmp_path / "jobs").glob("*/started.json"))


@pytest.mark.asyncio
async def test_reconciliation_waiting_for_lock_rechecks_cancellation(tmp_path):
    async with service(tmp_path, [ai(run_ase_call())]) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        paused = await api.wait(run.id)
        entry = api._entries[session.id]
        operation = paused.interrupts[0].operation_id
        async with entry.lock:
            reconciliation = asyncio.create_task(api.reconcile(
                run.id, operation, action="abandon", evidence="Operator stops waiting."))
            await asyncio.sleep(0)  # Reconciliation has checked status and waits on the lock.
            cancellation = asyncio.create_task(api.cancel_run(run.id))
            await until(lambda: api.get_session(session.id).status == "cancelled")
        with pytest.raises(RuntimeError, match="cannot advance"):
            await reconciliation
        await cancellation
        assert (await entry.context.journal.get(operation))["state"] == "proposed"


@pytest.mark.asyncio
async def test_completed_run_survives_transient_monitor_failure(tmp_path):
    async with service(tmp_path, [AIMessage(content="First"), AIMessage(content="Second")]) as api:
        session = await api.create_session()
        first = await api.start_run(session.id, "Hello")
        assert (await api.wait(first.id)).status == "completed"
        journal = api._entries[session.id].context.journal
        actual = journal.advance

        async def fail_once(*args, **kwargs):
            journal.advance = actual
            raise OSError("Transient completion acknowledgment failure")
        journal.advance = fail_once
        await api._tick(dispatch=False)
        assert api.get_run(first.id).status == "completed"
        await api._tick(dispatch=False)
        second = await api.start_run(session.id, "Follow up")
        assert (await api.wait(second.id)).assistant_response == "Second"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["clarification", "approval"])
async def test_queued_duplicate_reply_is_rejected(tmp_path, kind):
    call = (run_ase_call() if kind == "approval" else
            {"id": "ask", "name": "ask_human", "args": {"question": "Continue?"}})
    async with service(tmp_path, [ai(call), AIMessage(content="Done")]) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        run = await api.start_run(session.id, "Energy")
        pending = (await api.wait(run.id)).interrupts[0]

        def reply(first):
            if kind == "approval":
                return api.decide(run.id, pending.operation_id, digest=pending.payload["digest"], approve=first)
            return api.respond(run.id, pending.id, "First answer" if first else "Stale answer")

        async with api._entries[session.id].lock:
            first = asyncio.create_task(reply(True))
            second = asyncio.create_task(reply(False))
            await asyncio.sleep(0)
        await first
        with pytest.raises(ValueError, match="not waiting"):
            await second
        assert (await api.wait(run.id)).status == "completed"
        if kind == "approval":
            assert (await api.list_jobs(run.id))[0].status == "succeeded"


@pytest.mark.asyncio
async def test_failed_resume_consumes_old_interrupts(tmp_path):
    question = {"id": "ask", "name": "ask_human", "args": {"question": "Continue?"}}
    async with service(tmp_path, [ai(question)]) as api:
        session = await api.create_session()
        run = await api.start_run(session.id, "Hello")
        pending = (await api.wait(run.id)).interrupts[0]
        await api.respond(run.id, pending.id, "Yes")
        failed = await api.wait(run.id)  # Scripted model has no next response.
        assert failed.status == "failed"
        assert not failed.interrupts
        with pytest.raises(ValueError, match="not waiting"):
            await api.respond(run.id, pending.id, "A stale answer")


@pytest.mark.asyncio
async def test_cancellation_does_not_mark_previously_finished_jobs(tmp_path):
    question = {"id": "ask", "name": "ask_human", "args": {"question": "Next molecule?"}}
    async with service(tmp_path, [ai(run_ase_call()), AIMessage(content="Done"), ai(question)]) as api:
        session = await api.create_session()
        (Path(session.workspace) / "h2.xyz").write_text(H2)
        first = await api.start_run(session.id, "Energy")
        await approve_all(api, first.id)
        assert (await api.wait(first.id)).status == "completed"
        second = await api.start_run(session.id, "Next")
        await api.wait(second.id)
        await api.cancel_run(second.id)
        job = (await api.list_jobs(first.id))[0]
        assert job.status == "succeeded" and job.promoted and not job.cancel_requested


@pytest.mark.asyncio
async def test_startup_validates_all_sessions_before_launching(tmp_path, monkeypatch):
    async with service(tmp_path, []) as api:
        api._launch = lambda *args: None
        first, second = await api.create_session(), await api.create_session()
        await api.start_run(first.id, "Not checkpointed yet")
        api.store.put("session", second.model_copy(update={"configuration_id": "incompatible"}))
    launches = []
    monkeypatch.setattr(RunService, "_launch", lambda *args: launches.append(args))
    restored = service(tmp_path, [])
    with pytest.raises(ValueError, match="configuration changed"):
        await restored.__aenter__()
    assert not launches
    assert restored._stack is None and not restored._entries


@pytest.mark.asyncio
async def test_startup_failure_cancels_tasks_before_closing_storage(tmp_path, monkeypatch):
    async with service(tmp_path, []) as api:
        api._launch = lambda *args: None
        for _ in range(2):
            session = await api.create_session()
            await api.start_run(session.id, "Not checkpointed yet")
    actual, tasks = RunService._launch, []

    def fail_second(self, run, *args):
        if tasks:
            raise OSError("Injected startup storage failure")
        actual(self, run, *args)
        tasks.append(self._entries[run.session_id].task)

    monkeypatch.setattr(RunService, "_launch", fail_second)
    restored = service(tmp_path, [])
    with pytest.raises(OSError, match="startup storage"):
        await restored.__aenter__()
    assert tasks and all(task.cancelled() for task in tasks)
    assert restored._stack is None
