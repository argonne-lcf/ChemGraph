import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

pytest.importorskip("academy")

from academy.exchange import HttpExchangeFactory, LocalExchangeFactory
from academy.exchange.cloud.app import create_app
from academy.exchange.cloud.client import HttpAgentRegistration
from academy.handle import Handle
from academy.manager import Manager
from academy.runtime import Runtime, RuntimeConfig
from academy.serialize import SerializationStrategy
from aiohttp.test_utils import TestServer
from langchain_core.messages import AIMessage

from chemgraph.academy import ChemGraphPeer, PeerMessage, make_peer_message_tool
from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.models.endpoints import PreparedModel
from examples.academy.common import SessionHandler, emt_energy
from tests.test_main_agent import _ScriptedChatModel


@pytest_asyncio.fixture(params=["local", "http"])
async def factory(request):
    if request.param == "local":
        yield LocalExchangeFactory()
    else:
        # A real loopback HTTP exchange, with no external service or credentials.
        async with TestServer(create_app()) as server:
            yield HttpExchangeFactory(str(server.make_url("")), auth_method=None)


async def send(handle, content="task", *, message_id=None, sender="operator"):
    message = PeerMessage(sender=sender, content=content, **({"message_id": message_id} if message_id else {}))
    async with asyncio.timeout(3):
        ack = await handle.action("receive_message", message.model_dump(mode="json"))
    assert ack == {"status": "accepted", "message_id": message.message_id}
    return message


@pytest.mark.asyncio
async def test_prompt_acceptance_sequential_turns_duplicates_and_shutdown(factory):
    started, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    received = []
    errors = AsyncMock()

    async def turn(message):
        received.append(message.content)
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async with await Manager.from_exchange_factory(factory) as manager:
        peer = ChemGraphPeer(turn, on_error=errors)
        handle = await manager.launch(peer)
        await send(handle, "first", message_id="one")
        await asyncio.wait_for(started.wait(), 3)
        await send(handle, "first", message_id="one")
        await send(handle, "second", message_id="two")
        assert received == ["first"]
        release.set()
        await asyncio.wait_for(peer._inbox.join(), 3)
        assert received == ["first", "second"]
        await send(handle, "first", message_id="one")
        await asyncio.wait_for(peer._inbox.join(), 3)
        assert received == ["first", "second"]
        release.clear()
        started.clear()
        await send(handle, "third")
        await asyncio.wait_for(started.wait(), 3)
        await send(handle, "discard at shutdown")
        await manager.shutdown(handle, timeout=3)
        assert cancelled.is_set()
        assert peer._inbox.empty()
        assert received == ["first", "second", "third"]
        errors.assert_not_awaited()
        with pytest.raises(RuntimeError, match="shut down"):
            await peer.receive_message(PeerMessage(sender="operator", content="late").model_dump())


@pytest.mark.asyncio
async def test_failure_handler_preserves_mailbox_and_holds_later_messages(factory):
    repairing, repaired = asyncio.Event(), asyncio.Event()
    received = []

    async def broken(message):
        raise ValueError("workflow failed")

    async def replacement(message):
        received.append(message.content)

    async def recover(message, error):
        assert message.content == "bad"
        assert str(error) == "workflow failed"
        repairing.set()
        await repaired.wait()
        peer.on_turn = replacement

    async with await Manager.from_exchange_factory(factory) as manager:
        peer = ChemGraphPeer(broken, on_error=recover)
        handle = await manager.launch(peer)
        await send(handle, "bad", message_id="bad")
        await asyncio.wait_for(repairing.wait(), 3)
        await send(handle, "later")
        await send(handle, "bad", message_id="bad")
        assert not received
        assert peer._inbox.qsize() == 1
        repaired.set()
        await asyncio.wait_for(peer._inbox.join(), 3)
        assert received == ["later"]
        await send(handle, "subsequent")
        await asyncio.wait_for(peer._inbox.join(), 3)
        assert received == ["later", "subsequent"]


@pytest.mark.asyncio
async def test_native_http_registration_round_trip_and_runtime():
    received = asyncio.Queue()
    async with TestServer(create_app()) as server:
        factory = HttpExchangeFactory(str(server.make_url("")), auth_method=None)
        client = await factory.create_user_client()
        try:
            original = await client.register_agent(ChemGraphPeer)
            registration = HttpAgentRegistration.model_validate_json(original.model_dump_json())
            peer = ChemGraphPeer(received.put, on_error=AsyncMock())
            async with Runtime(peer, exchange_factory=factory, registration=registration,
                               config=RuntimeConfig(default_serializer=SerializationStrategy.JSON,
                                                    allowed_deserializers={SerializationStrategy.JSON})) as runtime:
                handle = Handle(registration.agent_id, exchange=client,
                                request_serializer=SerializationStrategy.JSON,
                                result_serializer=SerializationStrategy.JSON)
                await send(handle)
                assert (await asyncio.wait_for(received.get(), 3)).content == "task"
                await handle.shutdown()
                await runtime.wait_shutdown(timeout=3)
        finally:
            await client.close()


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def scripted_models(monkeypatch, responses):
    models = {name: _ScriptedChatModel(responses=items) for name, items in responses.items()}

    def load(**kwargs):
        name = kwargs["model_name"]
        return models[name], PreparedModel(endpoint_name=name, protocol="openai_compatible", client_kwargs={"model": name})

    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", load)
    return models


@pytest.mark.asyncio
async def test_real_workflows_delegate_emt_reply_and_accept_subsequent_work(factory, tmp_path, monkeypatch):
    models = scripted_models(monkeypatch, {
        "planner": [item for i in range(2) for item in (
            call("send_peer_message", {"recipient": "executor", "content": f"energy {i}"}, f"request-{i}"),
            AIMessage(content="delegated"), AIMessage(content=f"summary {i}"),
        )],
        "executor": [item for i in range(2) for item in (
            call("emt_energy", {"distance": 0.74 + i * 0.16}, f"emt-{i}"),
            call("send_peer_message", {"recipient": "planner", "content": f"result {i}"}, f"reply-{i}"),
            AIMessage(content="sent"),
        )],
    })
    handlers, peers = {}, {}
    async with await Manager.from_exchange_factory(factory) as manager:
        registrations = {name: await manager.exchange_client.register_agent(ChemGraphPeer) for name in models}
        handles = {name: Handle(reg.agent_id, exchange=manager.exchange_client) for name, reg in registrations.items()}
        for name in models:
            handler = SessionHandler(name, {other: handle for other, handle in handles.items() if other != name},
                                     model=name, work_dir=tmp_path, console_lock=asyncio.Lock())
            handlers[name] = handler
            peers[name] = ChemGraphPeer(handler, on_error=AsyncMock(side_effect=AssertionError("unexpected failure")))
            await manager.launch(peers[name], registration=registrations[name])
        for i in range(2):
            await send(handles["planner"], f"request {i}")
            async with asyncio.timeout(10):
                while (await handlers["planner"].completed.get()).sender != "executor":
                    pass
                for peer in peers.values():
                    await peer._inbox.join()
            snapshot = await handlers["planner"].agent.workflow.aget_state(handlers["planner"].session.config)
            assert snapshot.values["messages"][-1].content == f"summary {i}"
        for name, count in (("planner", 4), ("executor", 2)):
            handler = handlers[name]
            assert handler.agent.approval_mode == "review"
            snapshot = await handler.agent.workflow.aget_state(handler.session.config)
            assert sum(m.type == "human" for m in snapshot.values["messages"]) == count
        results = [json.loads(m.content) for m in snapshot.values["messages"] if m.type == "tool" and m.name == "emt_energy"]
        assert len(results) == 2
        assert results[0] == emt_energy.invoke({"distance": 0.74})
        assert results[0]["potential_energy"] != results[1]["potential_energy"]


@pytest.mark.asyncio
async def test_real_approval_queues_messages_until_session_resume(factory, tmp_path, monkeypatch):
    scripted_models(monkeypatch, {"worker": [
        call("write_file", {"file_path": "approved.txt", "content": "approved"}, "write"),
        AIMessage(content="first complete"), AIMessage(content="second complete"),
    ]})
    reviewing, approve = asyncio.Event(), asyncio.Event()

    async def review(function, payload):
        assert payload["action_requests"][0]["name"] == "write_file"
        reviewing.set()
        await approve.wait()
        return {"decisions": [{"type": "approve"}]}

    monkeypatch.setattr("examples.academy.common.console_call", review)
    async with await Manager.from_exchange_factory(factory) as manager:
        handler = SessionHandler("worker", {}, model="worker", work_dir=tmp_path, console_lock=asyncio.Lock())
        peer = ChemGraphPeer(handler, on_error=AsyncMock(side_effect=AssertionError("unexpected failure")))
        handle = await manager.launch(peer)
        await send(handle, "first")
        await asyncio.wait_for(reviewing.wait(), 10)
        await send(handle, "second")
        assert handler.session.pending_interrupts
        assert peer._inbox.qsize() == 1
        assert handler.completed.empty()
        approve.set()
        await asyncio.wait_for(peer._inbox.join(), 10)
        assert [handler.completed.get_nowait().content for _ in range(2)] == ["first", "second"]
        assert not handler.session.pending_interrupts


@pytest.mark.asyncio
async def test_tool_ids_are_stable_per_instance_and_failures_propagate():
    messages = []

    async def receive(action, payload):
        assert action == "receive_message"
        messages.append(payload)
        return {"status": "accepted", "message_id": payload["message_id"]}

    handle = SimpleNamespace(action=AsyncMock(side_effect=receive))
    tool = make_peer_message_tool(sender="planner", peers={"executor": handle})
    assert "tool_call_id" not in tool.tool_call_schema.model_json_schema()["properties"]
    invocation = {"type": "tool_call", "name": tool.name, "id": "call", "args": {"recipient": "executor", "content": "task"}}
    for _ in range(2):
        await tool.ainvoke(invocation)
    await tool.ainvoke({**invocation, "id": "next"})
    await make_peer_message_tool(sender="planner", peers={"executor": handle}).ainvoke(invocation)
    assert messages[0] == messages[1]
    assert len({m["message_id"] for m in messages}) == 3
    result = await tool.ainvoke({**invocation, "args": {"recipient": "missing", "content": "task"}})
    assert result.status == "error"
    handle.action.side_effect = TimeoutError("uncertain delivery")
    with pytest.raises(TimeoutError, match="uncertain delivery"):
        await tool.ainvoke(invocation)
    assert handle.action.await_count == 5


@pytest.mark.asyncio
async def test_main_session_failure_waits_for_application_retry(tmp_path, monkeypatch):
    scripted_models(monkeypatch, {"worker": [ValueError("model failed"),
                                           AIMessage(content="recovered"), AIMessage(content="later")]})
    failed, retry = asyncio.Event(), asyncio.Event()

    async def operator_retry(*_args):
        failed.set()
        await retry.wait()

    monkeypatch.setattr("examples.academy.common.console_call", operator_retry)
    async with await Manager.from_exchange_factory(LocalExchangeFactory()) as manager:
        handler = SessionHandler("worker", {}, model="worker", work_dir=tmp_path, console_lock=asyncio.Lock())
        peer = ChemGraphPeer(handler, on_error=handler.on_error)
        handle = await manager.launch(peer)
        await send(handle, "first")
        await asyncio.wait_for(failed.wait(), 5)
        assert handler.session.failed
        await send(handle, "second")
        assert peer._inbox.qsize() == 1
        retry.set()
        await asyncio.wait_for(peer._inbox.join(), 10)
        assert not handler.session.failed
        snapshot = await handler.agent.workflow.aget_state(handler.session.config)
        assert sum(m.type == "human" for m in snapshot.values["messages"]) == 2
        assert snapshot.values["messages"][-1].content == "later"


@pytest.mark.asyncio
async def test_caller_owned_single_agent_run_retains_thread(tmp_path, monkeypatch):
    scripted_models(monkeypatch, {"worker": [
        call("emt_energy", {"distance": 0.74}, "emt"), AIMessage(content="energy"), AIMessage(content="follow-up"),
    ]})
    graph = ChemGraph(model_name="worker", workflow_type="single_agent", tools=[emt_energy],
                      enable_memory=False, log_dir=str(tmp_path))
    results = []

    async def turn(message):
        results.append(await graph.run(message.content, config={"thread_id": "caller-owned"}))

    async with await Manager.from_exchange_factory(LocalExchangeFactory()) as manager:
        peer = ChemGraphPeer(turn, on_error=AsyncMock(side_effect=AssertionError("unexpected failure")))
        handle = await manager.launch(peer)
        await send(handle, "calculate")
        await send(handle, "follow up")
        await asyncio.wait_for(peer._inbox.join(), 10)
        assert len(results) == 2
        snapshot = await graph.workflow.aget_state({"configurable": {"thread_id": "caller-owned"}})
        assert sum(m.type == "human" for m in snapshot.values["messages"]) == 2


@pytest.mark.asyncio
async def test_http_example_in_separate_processes(tmp_path):
    # Run the documented HTTP entry point with scripted models in each child.
    # Registrations are passed verbatim, exactly as in the example commands.
    code = '''
import asyncio
import sys
from types import SimpleNamespace
from langchain_core.messages import AIMessage
import chemgraph.agent.llm_agent as agent_module
from chemgraph.models.endpoints import PreparedModel
from tests.test_main_agent import _ScriptedChatModel
from tests.test_academy_peer import call
from examples.academy.http import main
name, url, directory = sys.argv[1:]
if name == "planner":
    responses = [item for i in range(2) for item in (
        call("send_peer_message", {"recipient": "executor", "content": f"energy {i}"}, f"request-{i}"),
        AIMessage(content="delegated"), AIMessage(content=f"summary {i}"),
    )]
else:
    responses = [item for i in range(2) for item in (
        call("emt_energy", {"distance": 0.74 + i * 0.16}, f"emt-{i}"),
        call("send_peer_message", {"recipient": "planner", "content": f"energy {i}"}, f"reply-{i}"),
        AIMessage(content="sent"),
    )]
model = _ScriptedChatModel(responses=responses)
agent_module.load_chat_model_prepared = lambda **kwargs: (
    model, PreparedModel(endpoint_name=name, protocol="openai_compatible", client_kwargs={"model": name}))
asyncio.run(main(SimpleNamespace(command="run", name=name, url=url, work_dir=directory, model=name)))
'''

    async def wait_line(process, expected):
        async with asyncio.timeout(30):
            while True:
                line = await process.stdout.readline()
                assert line, f"Process exited before {expected!r}"
                if expected in line.decode():
                    return

    processes = {}
    async with TestServer(create_app()) as server:
        url = str(server.make_url(""))
        client = await HttpExchangeFactory(url, auth_method=None).create_user_client()
        try:
            registrations = {name: await client.register_agent(ChemGraphPeer) for name in ("planner", "executor")}
            env = {**os.environ, **{f"{name.upper()}_REGISTRATION": reg.model_dump_json()
                                   for name, reg in registrations.items()}}
            for name in registrations:
                processes[name] = await asyncio.create_subprocess_exec(
                    sys.executable, "-c", code, name, url, str(tmp_path), env=env,
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                )
                await wait_line(processes[name], f"{name} ready")
            handles = {name: Handle(reg.agent_id, exchange=client,
                                    request_serializer=SerializationStrategy.JSON,
                                    result_serializer=SerializationStrategy.JSON)
                       for name, reg in registrations.items()}
            for i in range(2):
                await send(handles["planner"], f"request {i}")
                await wait_line(processes["planner"], f"summary {i}")
            for name, handle in handles.items():
                await handle.shutdown()
                output, _ = await asyncio.wait_for(processes[name].communicate(), 10)
                assert processes[name].returncode == 0, output.decode()
        finally:
            for process in processes.values():
                if process.returncode is None:
                    process.kill()
                    await process.communicate()
            await client.close()
