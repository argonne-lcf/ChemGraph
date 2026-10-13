import pytest
pytest.importorskip("a2a")

import asyncio
import time

import httpx
from a2a.types import a2a_pb2 as a2a
from tests.service.conftest import TOKENS, running

from chemgraph.service.adapters import Outcome
from tests.service.fixtures import FixtureAdapter
from chemgraph.service.config import HPC, Calculation, Profile


class FakeBackend:
    def __init__(self):
        self.submissions = 0
        self.done = False
        self.unavailable = False
        self.ambiguous = False

    async def submit(self, operation):
        self.submissions += 1
        if self.ambiguous:
            raise TimeoutError("accepted remotely, response lost")
        return {"batch_id": "batch-1", "globus_task_ids": ["globus-1"]}

    async def status(self, batch_id):
        if self.unavailable:
            raise ConnectionError()
        return {"status": "completed" if self.done else "running", "total_tasks": 1}

    async def results(self, batch_id):
        return {"results": [{"potential_energy": -1, "units": "eV", "converged": False}]}

    async def cancel(self, batch_id):
        return {"cancelled": 0, "already_done": 1, "could_not_cancel": 0}


@pytest.fixture
def hpc_settings(settings):
    hpc = HPC(
        server_url="https://mcp.example/mcp",
        endpoint_id="test-endpoint",
        tool="run_ase",
        calculator="mace_mp",
        model="installed-model",
        scheduler_limits="10 min / one node",
        calculations=(Calculation(name="conformer-1", arguments={"params": {"driver": "opt"}}),),
    )
    return settings.model_copy(update={"profile": Profile(hpc=hpc)})


class RemoteAdapter(FixtureAdapter):
    def __init__(self, settings):
        super().__init__(settings)
        self.backend = FakeBackend()

    async def run(self, record):
        if record["text"].startswith("The tracked remote"):
            return Outcome(text="Remote job finished; nonconvergence makes its energy unvalidated.")
        operation = await self.operations.prepare(record, "tool-call-1", "conformer-1")
        operation = await self.operations.execute(operation["id"])
        if operation["state"] in {"approval", "uncertain"}:
            return Outcome(
                state="input_required",
                reason="approval",
                text="Operator approval required",
                interrupts=[{"id": "interrupt-1", "payload": {"operation_id": operation["id"]}}],
            )
        return Outcome(text="Computation submitted")


async def approve(service, app, task_id, operation_id=None):
    operation = (
        await service.operations.get(operation_id)
        if operation_id
        else (await service.operations.list(task_id))[0]
    )
    async with httpx.AsyncClient(
        base_url="http://test",
        transport=httpx.ASGITransport(app),
        headers={"Authorization": f"Bearer {TOKENS['operator']}"},
    ) as operator:
        response = await operator.post(
            f"/admin/operations/{operation['id']}/approve",
            json={
                "digest": operation["digest"],
                "interrupt_id": operation["interrupt_id"],
                "expires_in": 600,
            },
        )
        assert response.status_code == 200, response.text
    return operation


async def wait_backend(service, task_id):
    last_state = None
    try:
        async with asyncio.timeout(30):
            while True:
                record = await service.store.get(task_id)
                assert record is not None, f"Task {task_id} disappeared before reaching backend"
                last_state = {
                    "phase": record["phase"],
                    "reason": record.get("reason"),
                    "status": record["snapshot"]["status"],
                }
                if record["phase"] == "backend":
                    return
                if record["phase"] in {"waiting", "done"}:
                    pytest.fail(f"Task {task_id} stopped before reaching backend: {last_state}")
                await asyncio.sleep(0.01)
    except TimeoutError:
        pytest.fail(f"Timed out waiting for task {task_id} to reach backend: {last_state}")


async def test_approval_and_remote_restart_without_resubmission(hpc_settings):
    backend = FakeBackend()
    async with running(hpc_settings, RemoteAdapter(hpc_settings)) as (service, client, app):
        service.operations.backend = backend
        task = (await client.send("remote", blocking=True)).task
        assert task.status.state == a2a.TASK_STATE_INPUT_REQUIRED
        assert backend.submissions == 0
        operation = (await service.operations.list(task.id))[0]
        assert (
            await client.http.post(f"/admin/operations/{operation['id']}/approve", json={})
        ).status_code == 403
        with pytest.raises(Exception, match="operator"):
            await client.send("APPROVED", task_id=task.id)
    async with running(hpc_settings, RemoteAdapter(hpc_settings)) as (service, client, app):
        service.operations.backend = backend
        await approve(service, app, task.id)
        await wait_backend(service, task.id)
        assert (await client.get(task.id)).status.state == a2a.TASK_STATE_WORKING
        assert backend.submissions == 1
    backend.done = True
    async with running(hpc_settings, RemoteAdapter(hpc_settings)) as (service, client, _):
        service.operations.backend = backend
        operation = (await service.operations.list(task.id))[0]
        assert operation["receipt"]["globus_task_ids"] == ["globus-1"]
        operation["next_poll"] = 0
        async with service.lock:
            async with service.store.transaction():
                await service.operations.put(operation)
        await service.reconcile_jobs()
        finished = await service.wait(task.id, "alice")
        assert finished.status.state == a2a.TASK_STATE_COMPLETED
        assert "nonconvergence" in finished.status.message.parts[0].text
        assert backend.submissions == 1


async def test_ambiguous_submission_never_repeated(hpc_settings):
    backend = FakeBackend()
    backend.ambiguous = True
    async with running(hpc_settings, RemoteAdapter(hpc_settings)) as (service, client, app):
        service.operations.backend = backend
        task = (await client.send("remote", blocking=True)).task
        operation = await approve(service, app, task.id)
        task = await service.wait(task.id, "alice")
        assert task.status.state == a2a.TASK_STATE_INPUT_REQUIRED
        assert "unknown" in task.status.message.parts[0].text
        assert backend.submissions == 1
    async with running(hpc_settings, RemoteAdapter(hpc_settings)) as (service, _, _):
        service.operations.backend = backend
        assert (await service.operations.execute(operation["id"]))["state"] == "uncertain"
        assert backend.submissions == 1


async def test_expired_or_changed_action_approval(hpc_settings):
    async with running(hpc_settings, RemoteAdapter(hpc_settings)) as (service, client, _):
        service.operations.backend = FakeBackend()
        task = (await client.send("remote", blocking=True)).task
        operation = (await service.operations.list(task.id))[0]
        with pytest.raises(ValueError, match="digest"):
            await service.operations.approve(
                operation["id"], {"digest": "changed", "interrupt_id": "interrupt-1"}, "operator"
            )
        operation["approval"] = {"digest": operation["digest"], "expires": time.time() - 1}
        async with service.lock:
            async with service.store.transaction():
                await service.operations.put(operation)
        assert (await service.operations.execute(operation["id"]))["state"] == "approval"
        assert service.operations.backend.submissions == 0


async def test_crash_between_acceptance_and_identifier_record(hpc_settings):
    async with running(hpc_settings, RemoteAdapter(hpc_settings)) as (service, client, _):
        task = (await client.send("remote", blocking=True)).task
        operation = (await service.operations.list(task.id))[0]
        operation["state"] = "submitting"
        async with service.lock:
            async with service.store.transaction():
                await service.operations.put(operation)
    async with running(hpc_settings, RemoteAdapter(hpc_settings)) as (service, _, _):
        assert (await service.operations.get(operation["id"]))["state"] == "uncertain"
        assert (await service.store.get(task.id))["reason"] == "reconciliation"


async def test_operator_can_associate_recovered_job_without_submitting(hpc_settings):
    backend = FakeBackend()
    backend.ambiguous = True
    async with running(hpc_settings, RemoteAdapter(hpc_settings)) as (service, client, app):
        service.operations.backend = backend
        task = (await client.send("remote", blocking=True)).task
        operation = await approve(service, app, task.id)
        await service.wait(task.id, "alice")
        await service.operations.reconcile(
            operation["id"], "recovered-batch", "Reviewed persistent gateway tracker", "operator"
        )
        await wait_backend(service, task.id)
        assert (await service.operations.get(operation["id"]))["batch_id"] == "recovered-batch"
        assert backend.submissions == 1


async def test_submission_limit_and_unconfigured_calculation(hpc_settings):
    hpc_settings = hpc_settings.model_copy(
        update={"profile": hpc_settings.profile.model_copy(update={"max_submissions": 1})}
    )
    async with running(hpc_settings, RemoteAdapter(hpc_settings)) as (service, client, _):
        task = (await client.send("remote", blocking=True)).task
        record = await service.store.get(task.id)
        with pytest.raises(ValueError, match="budget"):
            await service.operations.prepare(record, "other-call", "conformer-1")
        with pytest.raises(ValueError, match="template"):
            await service.operations.prepare(record, "other-call", "unconfigured")


async def test_backend_unavailable_preserves_job_and_cancel_not_falsely_confirmed(hpc_settings):
    backend = FakeBackend()
    async with running(hpc_settings, RemoteAdapter(hpc_settings)) as (service, client, app):
        service.operations.backend = backend
        task = (await client.send("remote", blocking=True)).task
        await approve(service, app, task.id)
        await wait_backend(service, task.id)
        backend.unavailable = True
        await service.operations.poll()
        operation = (await service.operations.list(task.id))[0]
        assert operation["state"] == "running"
        with pytest.raises(Exception, match="tracking continues"):
            await client.cancel(task.id)


pytestmark = pytest.mark.asyncio
