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
from academy.handle import Handle
from academy.manager import Manager
from academy.runtime import Runtime
from academy.serialize import SerializationStrategy
from aiohttp.test_utils import TestServer
from ase.calculators.emt import EMT
from ase.io import read
from langchain_core.messages import AIMessage

from chemgraph.academy import ChemGraphPeer, PeerMessage, make_peer_message_tool, send_peer_message
from chemgraph.memory.store import SessionStore
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
        ack = await send_peer_message(handle, message)
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
        # A relaunch of the same instance (e.g. a Manager retry) fails at startup.
        registration = await manager.exchange_client.register_agent(ChemGraphPeer)
        with pytest.raises(RuntimeError, match="not restartable"):
            async with Runtime(peer, exchange_factory=factory, registration=registration):
                pass


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


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def energy_call(structure, results, call_id):
    return call("run_ase", {"params": {
        "input_structure_file": str(structure), "output_results_file": str(results),
        "driver": "energy", "calculator": {"calculator_type": "emt"},
    }}, call_id)


def peer_responses(name):
    if name == "planner":
        return [item for i in range(2) for item in (
            call("send_peer_message", {"recipient": "executor", "content": f"energy {i}"}, f"request-{i}"),
            AIMessage(content="delegated"), AIMessage(content=f"summary {i}"),
        )]
    return [item for i, distance in enumerate((0.74, 0.90)) for item in (
        call("search_tools", {"query": "save_atomsdata_to_file run_ase"}, f"search-{i}"),
        call("load_tools", {"names": ["save_atomsdata_to_file", "run_ase"]}, f"load-{i}"),
        call("save_atomsdata_to_file", {
            "atomsdata": {"numbers": [1, 1], "positions": [[0, 0, 0], [0, 0, distance]]},
            "fname": f"h2-{distance:.2f}.xyz",
        }, f"structure-{i}"),
        energy_call(f"h2-{distance:.2f}.xyz", f"h2-{distance:.2f}.json", f"emt-{i}"),
        call("send_peer_message", {"recipient": "planner", "content": ""}, f"reply-{i}"),
        AIMessage(content="sent"),
    )]


class _PeerChatModel(_ScriptedChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for tool_call in result.generations[0].message.tool_calls:
            if tool_call["name"] == "send_peer_message" and tool_call["args"]["recipient"] == "planner":
                # Reply with the real tool output, including its file path.
                energy = next(m for m in reversed(messages) if m.type == "tool" and m.name == "run_ase")
                request = next(m for m in reversed(messages) if m.type == "human")
                tool_call["args"].update(content=energy.content,
                                         reply_to=json.loads(request.content.split("\n", 1)[1])["message_id"])
        return result


async def approve_native_tools(_function, payload):
    actions = payload["action_requests"]
    assert actions and all(action["name"] in {"save_atomsdata_to_file", "run_ase"} for action in actions)
    return {"decisions": [{"type": "approve"} for _ in actions]}


def assert_peer_histories(directory):
    sessions = {}
    for name, count in (("planner", 4), ("executor", 2)):
        database = directory / name / "sessions.db"
        assert database.is_file()
        store = SessionStore(str(database))
        summaries = store.list_sessions()
        assert len(summaries) == 1
        session = store.get_session(summaries[0].session_id)
        assert session.status == "completed"
        assert session.model_name == name
        assert sum(m.role == "human" for m in session.messages) == count
        assert session.messages[-1].content == ("summary 1" if name == "planner" else "sent")
        sessions[name] = session
    assert sessions["planner"].session_id != sessions["executor"].session_id
    results = [json.loads(m.content) for m in sessions["executor"].messages if m.tool_name == "run_ase"]
    assert len(results) == 2
    requests = [json.loads(m.content.split("\n", 1)[1]) for m in sessions["executor"].messages if m.role == "human"]
    replies = [json.loads(m.content.split("\n", 1)[1]) for m in sessions["planner"].messages
               if m.role == "human" and json.loads(m.content.split("\n", 1)[1])["sender"] == "executor"]
    assert [json.loads(reply["content"]) for reply in replies] == results
    assert [reply["reply_to"] for reply in replies] == [request["message_id"] for request in requests]
    for result, distance in zip(results, (0.74, 0.90), strict=True):
        structure = directory / "executor" / f"h2-{distance:.2f}.xyz"
        atoms = read(structure)
        assert atoms.get_chemical_formula() == "H2"
        assert atoms.get_distance(0, 1) == pytest.approx(distance)
        atoms.calc = EMT()
        assert result["status"] == "success"
        assert result["driver"] == "energy"
        assert result["energy_unit"] == "eV"
        assert result["potential_energy"] == pytest.approx(atoms.get_potential_energy())
        assert result["results_file"] == str(directory / "executor" / f"h2-{distance:.2f}.json")
        saved = json.loads(Path(result["results_file"]).read_text())
        assert saved["success"] is True
        assert saved["energy_unit"] == "eV"
        assert saved["potential_energy"] == result["potential_energy"]
    assert results[0]["potential_energy"] != results[1]["potential_energy"]
    assert not list((directory / "planner").glob("h2-*"))


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
    handle.action.side_effect = None
    handle.action.return_value = {"status": "accepted", "message_id": "other"}
    with pytest.raises(RuntimeError, match="Invalid peer acknowledgement"):
        await send_peer_message(handle, PeerMessage(sender="dispatcher", content="task"))


@pytest.mark.asyncio
async def test_http_example_in_separate_processes(tmp_path):
    # Run the documented HTTP entry point with scripted models in each child.
    # Registrations are passed verbatim, exactly as in the example commands.
    code = '''
import asyncio
import sys
from types import SimpleNamespace
import chemgraph.agent.llm_agent as agent_module
from chemgraph.models.endpoints import PreparedModel
from tests.test_academy_peer import _PeerChatModel, approve_native_tools, peer_responses
import examples.academy.common as common
from examples.academy.http_peers import main
name, url, directory = sys.argv[1:]
common.console_call = approve_native_tools
model = _PeerChatModel(responses=peer_responses(name))
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
            assert_peer_histories(tmp_path)
        finally:
            for process in processes.values():
                if process.returncode is None:
                    process.kill()
                    await process.communicate()
            await client.close()
