import asyncio

import pytest

from chemgraph.distributed.protocol import PeerEnvelope, PeerTurnResult
from chemgraph.distributed.runtime import PeerRuntime


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
