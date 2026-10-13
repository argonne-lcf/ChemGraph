import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from chemgraph.distributed.protocol import PeerEnvelope, PeerTurnResult
from chemgraph.distributed.runtime import PeerRuntime
from chemgraph.distributed.runner import ChemGraphRunner


def message(content="task", **kwargs):
    return PeerEnvelope(run_id="run", launch_id="launch", sender="other", recipient="worker", content=content, **kwargs)


class Runner:
    interrupts = ()
    def __init__(self):
        self.messages = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.result = PeerTurnResult("answer")
    async def run(self, envelope):
        self.messages.append(envelope)
        self.started.set()
        await self.release.wait()
        return self.result
    async def resume(self, response):
        self.response = response
        return PeerTurnResult("resumed")
    async def retry(self):
        return PeerTurnResult("retried")


def runtime(runner, trace=None):
    return PeerRuntime(name="worker", run_id="run", launch_id="launch",
                       allowed_senders=("other",), runner=runner, trace=trace)


@pytest.mark.asyncio
async def test_delivery_acknowledges_queue_and_serializes_turns():
    runner = Runner()
    peer = runtime(runner)
    await peer.start()
    try:
        first, second = message("one"), message("two")
        assert await peer.receive(first) == {"status": "accepted", "message_id": first.message_id}
        await runner.started.wait()
        await peer.receive(second)
        assert runner.messages == [first]
        runner.release.set()
        await asyncio.wait_for(peer.inbox.join(), 2)
        assert runner.messages == [first, second]
        assert peer.status()["state"] == "idle"
        await peer.receive(message("three"))
        await asyncio.wait_for(peer.inbox.join(), 2)
        assert len(runner.messages) == 3
    finally:
        await peer.close(timeout_s=0.05)


@pytest.mark.asyncio
async def test_incoming_messages_do_not_resume_approval():
    runner = Runner()
    runner.release.set()
    runner.result = PeerTurnResult(status="waiting_for_user", interrupts=({"id": "approval"},))
    peer = runtime(runner)
    await peer.start()
    try:
        await peer.receive(message())
        await runner.started.wait()
        await asyncio.sleep(0)
        await peer.receive(message("approve"))
        assert peer.status()["state"] == "waiting_for_user"
        assert len(runner.messages) == 1
        runner.result = PeerTurnResult("next")
        await peer.resume({"decisions": [{"type": "reject"}]})
        await asyncio.wait_for(peer.inbox.join(), 2)
        assert len(runner.messages) == 2
        assert runner.response == {"decisions": [{"type": "reject"}]}
    finally:
        await peer.close(timeout_s=0.05)


@pytest.mark.asyncio
async def test_wrong_launch_recipient_and_sender_rejected():
    runner = Runner()
    peer = runtime(runner)
    await peer.start()
    try:
        for changes in ({"launch_id": "old"}, {"recipient": "wrong"}, {"sender": "unknown"}):
            with pytest.raises(ValueError):
                await peer.receive(message().model_copy(update=changes))
        assert peer.inbox.empty()
    finally:
        await peer.close()


@pytest.mark.asyncio
async def test_shutdown_drains_queued_work_and_rejects_new_messages():
    runner = Runner()
    peer = runtime(runner)
    await peer.start()
    await peer.receive(message())
    closing = asyncio.create_task(peer.close())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="not accepting"):
        await peer.receive(message())
    runner.release.set()
    await closing
    assert peer.state == "stopped"
    assert len(runner.messages) == 1


@pytest.mark.asyncio
async def test_blocked_shutdown_reports_abandoned_work():
    runner = Runner()
    events = []
    peer = runtime(runner, lambda event, payload: events.append((event, payload)))
    await peer.start()
    await peer.receive(message())
    await runner.started.wait()
    await peer.close(timeout_s=0.01)
    assert peer.state == "stopped"
    assert any(payload.get("error") == "shutdown_timeout" for _, payload in events)


@pytest.mark.asyncio
async def test_duplicate_messages_execute_once_and_conflicting_ids_are_rejected():
    runner = Runner()
    peer = runtime(runner)
    await peer.start()
    try:
        first = message()
        acknowledgements = await asyncio.gather(*(peer.receive(first) for _ in range(10)))
        assert acknowledgements == [{"status": "accepted", "message_id": first.message_id}] * 10
        await runner.started.wait()
        with pytest.raises(ValueError, match="different content"):
            await peer.receive(first.model_copy(update={"content": "another task"}))
        runner.release.set()
        await asyncio.wait_for(peer.inbox.join(), 2)
        await peer.receive(first)
        await asyncio.wait_for(peer.inbox.join(), 2)
        assert runner.messages == [first]
    finally:
        await peer.close(timeout_s=0.01)


@pytest.mark.asyncio
async def test_failed_legacy_peer_rejects_new_work(tmp_path):
    runner = ChemGraphRunner(SimpleNamespace(log_dir=str(tmp_path), run=AsyncMock(
        side_effect=RuntimeError("model unavailable"))), thread_id="legacy")
    peer = runtime(runner)
    await peer.start()
    try:
        first = message()
        await peer.receive(first)
        async with asyncio.timeout(2):
            while peer.state != "failed":
                await asyncio.sleep(0)
        assert (await peer.receive(first))["status"] == "accepted"
        with pytest.raises(RuntimeError, match="restart the failed peer"):
            await peer.receive(message("next"))
        with pytest.raises(RuntimeError, match="restart other failed peers"):
            await peer.retry()
        assert peer.inbox.empty()
        runner.agent.run.assert_awaited_once()
    finally:
        await peer.close(timeout_s=0.01)


@pytest.mark.asyncio
async def test_failed_retryable_peer_accepts_work_after_recovery():
    runner = Runner()
    runner.supports_retry = True
    runner.run = AsyncMock(side_effect=RuntimeError("temporary failure"))
    peer = runtime(runner)
    await peer.start()
    try:
        await peer.receive(message())
        async with asyncio.timeout(2):
            while peer.state != "failed":
                await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="retry the failed turn"):
            await peer.receive(message("next"))
        await peer.retry()
        runner.run.side_effect = None
        runner.run.return_value = PeerTurnResult("next answer")
        await peer.receive(message("next"))
        await asyncio.wait_for(peer.inbox.join(), 2)
        assert peer.output == "next answer"
    finally:
        await peer.close(timeout_s=0.01)


@pytest.mark.asyncio
async def test_identical_legacy_questions_have_distinct_control_tokens(tmp_path):
    answers = []
    class Agent:
        log_dir = str(tmp_path)
        async def run(self, prompt, config):
            for _ in range(2):
                answers.append(await self.human_input_handler("Approve?", {"question": "same"}))
            return "done"

    peer = runtime(ChemGraphRunner(Agent(), thread_id="legacy"))
    await peer.start()
    try:
        await peer.receive(message())
        async with asyncio.timeout(2):
            while peer.status()["state"] != "waiting_for_user":
                await asyncio.sleep(0)
        first = peer.status()
        runner_token = peer.runner.interrupt_token
        await peer.resume("one", control_token=first["control_token"])
        async with asyncio.timeout(2):
            while peer.runner.interrupt_token == runner_token:
                await asyncio.sleep(0)
        second = peer.status()
        assert first["interrupts"] == second["interrupts"]
        assert first["control_token"] != second["control_token"]
        with pytest.raises(RuntimeError, match="Peer state changed"):
            await peer.resume("stale", control_token=first["control_token"])
        await peer.resume("two", control_token=second["control_token"])
        await asyncio.wait_for(peer.inbox.join(), 2)
        assert answers == ["one", "two"]
        assert peer.status()["control_token"] is None
    finally:
        await peer.close(timeout_s=0.01)


@pytest.mark.asyncio
async def test_failed_retry_attempt_invalidates_old_control_token():
    runner = Runner()
    runner.run = AsyncMock(side_effect=RuntimeError("failed"))
    runner.retry = AsyncMock(side_effect=[RuntimeError("still failed"), PeerTurnResult("recovered")])
    peer = runtime(runner)
    await peer.start()
    try:
        await peer.receive(message())
        async with asyncio.timeout(2):
            while peer.state != "failed":
                await asyncio.sleep(0)
        first = peer.status()["control_token"]
        with pytest.raises(RuntimeError, match="still failed"):
            await peer.retry(control_token=first)
        second = peer.status()["control_token"]
        assert first != second
        with pytest.raises(RuntimeError, match="Peer state changed"):
            await peer.retry(control_token=first)
        await peer.retry(control_token=second)
        await asyncio.wait_for(peer.inbox.join(), 2)
        assert runner.retry.await_count == 2
    finally:
        await peer.close(timeout_s=0.01)
