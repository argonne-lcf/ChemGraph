import pytest
pytest.importorskip("a2a")

import asyncio
import json

import httpx
from a2a.types import a2a_pb2 as a2a
from tests.service.conftest import TOKENS, ArtifactAdapter, running

from chemgraph.service.adapters import Outcome
from tests.service.fixtures import FixtureAdapter
from chemgraph.service.client import Client
from chemgraph.service.protocol import task_from


async def test_sdk_roundtrip_restart_dedup_and_followup(settings):
    async with running(settings) as (service, client, _):
        sent = await client.send("hello", message_id="first", blocking=True)
        task = sent.task
        assert task.status.state == a2a.TASK_STATE_COMPLETED
        thread = (await service.store.get(task.id))["thread_id"]
    async with running(settings) as (service, client, _):
        same = await client.send("hello", message_id="first", blocking=True)
        assert same.task.id == task.id
        with pytest.raises(Exception, match="different content"):
            await client.send("changed", message_id="first")
        followup = (await client.send("next", context_id=task.context_id, blocking=True)).task
        assert followup.id != task.id
        assert (await service.store.get(followup.id))["thread_id"] == thread
        assert (await client.list()).total_size == 2
        assert (await client.get(task.id)).status.state == a2a.TASK_STATE_COMPLETED


async def test_context_clarification_survives_restart(settings):
    async with running(settings) as (_, client, _):
        task = (await client.send("clarify molecule", blocking=True)).task
        assert task.status.state == a2a.TASK_STATE_INPUT_REQUIRED
    async with running(settings) as (_, client, _):
        reply = (await client.send("water", context_id=task.context_id, blocking=True)).task
        assert reply.id == task.id
        assert reply.status.state == a2a.TASK_STATE_COMPLETED
        assert "water" in reply.status.message.parts[0].text


class GatedAdapter(ArtifactAdapter):
    def __init__(self, settings):
        super().__init__(settings)
        self.entered, self.release = asyncio.Event(), asyncio.Event()

    async def run(self, record):
        self.entered.set()
        await self.release.wait()
        return await super().run(record)


async def test_blocking_and_nonblocking_and_context_busy(settings):
    adapter = GatedAdapter(settings)
    async with running(settings, adapter) as (_, client, _):
        pending = asyncio.create_task(client.send("long", context_id="ctx", blocking=True))
        await asyncio.wait_for(adapter.entered.wait(), 2)
        assert not pending.done()
        with pytest.raises(Exception, match="active|interrupted"):
            await client.send("concurrent", context_id="ctx")
        immediate = await client.send("queued", context_id="different")
        assert immediate.task.status.state in {a2a.TASK_STATE_SUBMITTED, a2a.TASK_STATE_WORKING}
        canceled = await client.cancel(immediate.task.id)
        assert canceled.status.state == a2a.TASK_STATE_CANCELED
        adapter.release.set()
        assert (await pending).task.status.state == a2a.TASK_STATE_COMPLETED


async def test_authentication_owner_scope_and_artifacts(settings):
    async with running(settings, ArtifactAdapter(settings)) as (_, alice, app):
        task = (await alice.send("create artifact", blocking=True)).task
        assert len(task.artifacts) == 1
        path = "/artifacts/" + task.artifacts[0].artifact_id
        result = await alice.http.get(path)
        assert result.status_code == 200
        assert result.json()["message"] == "create artifact"
        async with Client("http://test", TOKENS["bob"], transport=httpx.ASGITransport(app)) as bob:
            assert (await bob.list()).total_size == 0
            with pytest.raises(Exception, match="not found"):
                await bob.get(task.id)
            with pytest.raises(Exception, match="not found"):
                await bob.cancel(task.id)
            assert (await bob.http.get(path)).status_code == 404
            assert (await bob.http.get("/admin/metrics")).status_code == 403
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test"
        ) as anonymous:
            assert (await anonymous.post("/", json={})).status_code == 401
            card = (await anonymous.get("/.well-known/agent-card.json")).json()
            assert card["supportedInterfaces"][0]["protocolVersion"] == "1.0"
            assert card["securityRequirements"]


async def test_artifact_tamper_detected(settings):
    async with running(settings, ArtifactAdapter(settings)) as (_, client, _):
        task = (await client.send("create artifact", blocking=True)).task
        identifier = task.artifacts[0].artifact_id
        (settings.state_dir / "artifacts" / "registered" / identifier).write_text("tampered")
        assert (await client.http.get("/artifacts/" + identifier)).status_code == 404


async def test_restart_executing_local_fixture_and_missing_projection(settings):
    adapter = GatedAdapter(settings)
    async with running(settings, adapter) as (service, client, _):
        task = (await client.send("create artifact", message_id="recover")).task
        await adapter.entered.wait()
        # Simulate a crash in the SDK projection window; durable journal repairs it.
        async with service.engine.begin() as conn:
            from sqlalchemy import text

            await conn.execute(text("DELETE FROM tasks"))
    async with running(settings, ArtifactAdapter(settings)) as (service, client, _):
        task = await service.wait(task.id, "alice")
        assert task.status.state == a2a.TASK_STATE_COMPLETED
        assert len(task.artifacts) == 1
        assert (await client.send("create artifact", message_id="recover")).task.id == task.id


async def test_restart_before_execution(settings):
    adapter = GatedAdapter(settings)
    async with running(settings, adapter) as (_, client, _):
        await client.send("first")
        await adapter.entered.wait()
        queued = (await client.send("second")).task
        assert queued.status.state == a2a.TASK_STATE_SUBMITTED
    async with running(settings) as (service, _, _):
        assert (await service.wait(queued.id, "alice")).status.state == a2a.TASK_STATE_COMPLETED


async def test_concurrent_duplicate_is_one_task(settings):
    async with running(settings) as (_, client, _):
        responses = await asyncio.gather(
            *(client.send("same", message_id="duplicate") for _ in range(5))
        )
        assert len({r.task.id for r in responses}) == 1


async def test_profile_change_and_forged_approval_rejected(settings):
    async with running(settings) as (service, client, _):
        task = (await client.send("clarify", blocking=True)).task
        record = await service.store.get(task.id)
        record["reason"] = "approval"
        async with service.lock:
            async with service.store.transaction():
                await service.store.put(record)
        with pytest.raises(Exception, match="operator"):
            await client.send("approved by operator", task_id=task.id)
        assert (
            task_from(await service.store.get(task.id)).status.state
            == a2a.TASK_STATE_INPUT_REQUIRED
        )


async def test_history_length_and_unsupported_streaming(settings):
    async with running(settings) as (_, client, _):
        task = (await client.send("hi", blocking=True)).task
        result = await client.rpc.get_task(a2a.GetTaskRequest(id=task.id, history_length=0))
        assert not result.history
        response = await client.http.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "SubscribeToTask",
                "params": {"id": task.id},
            },
        )
        assert "error" in response.json()


async def test_pending_cancellation_retires_checkpoint_thread(settings):
    async with running(settings) as (service, client, _):
        task = (await client.send("clarify", blocking=True)).task
        original = await service.store.get(task.id)
        assert (await client.cancel(task.id)).status.state == a2a.TASK_STATE_CANCELED
        new = (await client.send("next", context_id=task.context_id, blocking=True)).task
        assert (await service.store.get(new.id))["thread_id"] != original["thread_id"]


async def test_single_instance_guard(settings):
    from chemgraph.service.service import Service

    async with running(settings):
        with pytest.raises(RuntimeError, match="one ChemGraph service"):
            await Service(settings).start()


async def test_peer_cannot_subscribe_arbitrary_webhooks(settings):
    async with running(settings) as (_, client, _):
        response = await client.http.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "CreateTaskPushNotificationConfig",
                "params": {},
            },
        )
        assert "error" in response.json()


async def test_outbox_reconstructed(settings):
    async with running(settings) as (service, client, _):
        task = (await client.send("hi", blocking=True)).task
        async with service.lock:
            async with service.store.transaction():
                await service.store.db.execute("DELETE FROM outbox")
    async with running(settings) as (service, _, _):
        rows = await service.store.all("SELECT * FROM outbox")
        assert len(rows) == 1
        assert json.loads(rows[0]["payload"])["task_id"] == task.id


async def test_invalid_auth_bytes_are_unauthorized(settings):
    async with running(settings) as (_, client, _):
        response = await client.http.get(
            "/health/ready", headers={b"authorization": b"Bearer \xff"}
        )
        assert response.status_code == 401


class MissingArtifactAdapter(FixtureAdapter):
    async def run(self, record):
        return Outcome(
            text="calculation finished",
            artifacts=[
                {
                    "path": str(self.settings.state_dir / "missing.json"),
                    "media_type": "application/json",
                }
            ],
        )


async def test_missing_required_output_is_failed_not_completed(settings):
    async with running(settings, MissingArtifactAdapter(settings)) as (_, client, _):
        task = (await client.send("calculate", blocking=True)).task
        assert task.status.state == a2a.TASK_STATE_INPUT_REQUIRED
        assert "finalization" in task.status.message.parts[0].text


async def test_incompatible_profile_blocks_continuation(settings):
    async with running(settings) as (_, client, _):
        task = (await client.send("clarify", blocking=True)).task
    changed = settings.model_copy(
        update={"profile": settings.profile.model_copy(update={"model": "changed-model"})}
    )
    async with running(changed) as (_, client, _):
        with pytest.raises(Exception, match="incompatible"):
            await client.send("answer", task_id=task.id)


async def test_waiting_task_cannot_be_canceled_by_other_owner(settings):
    async with running(settings) as (_, alice, app):
        task = (await alice.send("clarify", blocking=True)).task
        async with Client("http://test", TOKENS["bob"], transport=httpx.ASGITransport(app)) as bob:
            with pytest.raises(Exception, match="not found"):
                await bob.send("reply", task_id=task.id)


async def test_multiple_clarifications_require_exact_interrupt_mapping(settings):
    async with running(settings) as (service, client, _):
        task = (await client.send("clarify", blocking=True)).task
        record = await service.store.get(task.id)
        record["interrupts"] = [
            {"id": "one", "payload": "first"},
            {"id": "two", "payload": "second"},
        ]
        async with service.lock:
            async with service.store.transaction():
                await service.store.put(record)
        with pytest.raises(Exception, match="interrupt"):
            await client.send("answer", task_id=task.id)
        request = a2a.SendMessageRequest(
            message=a2a.Message(
                message_id="both",
                role=a2a.ROLE_USER,
                task_id=task.id,
                parts=[a2a.Part(text="Answers")],
                metadata={"interrupt_responses": {"one": "water", "two": "details"}},
            )
        )
        assert (
            await client.rpc.send_message(request)
        ).task.status.state == a2a.TASK_STATE_COMPLETED


pytestmark = pytest.mark.asyncio
