import asyncio
from contextlib import AsyncExitStack

import pytest

pytest.importorskip("academy")

from academy.exchange.local import LocalExchangeFactory
from langchain_core.messages import AIMessage

from chemgraph.distributed.academy import AcademyTransport
from chemgraph.distributed.config import DistributedConfig
from chemgraph.distributed.peer_tools import build_peer_tool
from chemgraph.distributed.protocol import PeerEnvelope, PeerTurnResult
from chemgraph.distributed.runner import ConfiguredRunner
from chemgraph.distributed.runtime import PeerRuntime
from chemgraph.models.endpoints import PreparedModel
from tests.test_main_agent import _ScriptedChatModel


def envelope(sender, recipient, content="task", launch_id="launch"):
    return PeerEnvelope(run_id="run", launch_id=launch_id, sender=sender, recipient=recipient, content=content)


@pytest.mark.asyncio
async def test_main_agent_peers_exchange_results_and_accept_more_work(tmp_path, monkeypatch):
    def send(recipient, content, call_id):
        return AIMessage(content="", tool_calls=[{"name": "send_peer_message", "id": call_id,
                                                 "args": {"recipient": recipient, "content": content}}])

    models = {
        "planner": _ScriptedChatModel(responses=[item for i in range(2) for item in (
            send("executor", f"task {i}", f"delegate-{i}"), AIMessage(content="delegated"),
            AIMessage(content=f"summary {i}"),
        )]),
        "executor": _ScriptedChatModel(responses=[item for i in range(2) for item in (
            send("planner", f"result {i}", f"reply-{i}"), AIMessage(content="result sent"),
        )]),
    }

    def load(**kwargs):
        name = kwargs["model_name"]
        return models[name], PreparedModel(endpoint_name=name, protocol="openai_compatible", client_kwargs={"model": name})

    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", load)
    settings = DistributedConfig(run_id="run", launch_id="launch", exchange_url="http://unused", run_dir=tmp_path,
                                 peers={name: {"model_name": name, "allowed_peers": [other]}
                                        for name, other in (("planner", "executor"), ("executor", "planner"))})
    factory = LocalExchangeFactory()
    runtimes = {}
    async with AsyncExitStack() as stack:
        for name, spec in settings.peers.items():
            transport = await stack.enter_async_context(AcademyTransport(
                exchange_factory=factory, run_id="run", launch_id="launch", name=name, timeout_s=3,
            ))
            tool = build_peer_tool(transport=transport, run_id="run", launch_id="launch", sender=name,
                                   allowed_peers=spec.allowed_peers)
            runner = await stack.enter_async_context(ConfiguredRunner(settings, name, [tool], lambda *_args: None))
            peer = PeerRuntime(name=name, run_id="run", launch_id="launch", allowed_senders=spec.allowed_peers,
                               runner=runner)
            await peer.start()
            stack.push_async_callback(peer.close, timeout_s=0.05)
            await transport.start(peer.receive, peer.status)
            runtimes[name] = peer

        operator = await stack.enter_async_context(AcademyTransport(
            exchange_factory=factory, run_id="run", launch_id="launch", timeout_s=3,
        ))
        for i in range(2):
            assert (await operator.send(envelope("operator", "planner", f"request {i}")))["status"] == "accepted"
            for name in ("planner", "executor", "planner"):
                await asyncio.wait_for(runtimes[name].inbox.join(), 5)
            status = await operator.peer_status("planner")
            assert status["state"] == "idle"
            assert status["output"] == f"summary {i}"
            assert runtimes["executor"].status()["state"] == "idle"
        for peer in runtimes.values():
            assert peer.runner.agent.workflow_type == "main_agent"
            tool_names = {item.name for item in models[peer.name].bound_tools}
            assert {"send_peer_message", "read_file", "write_file"} <= tool_names
            assert not {"run_ase", "search_tools", "load_tools", "search_agents", "load_agents", "task"} & tool_names
            snapshot = await peer.runner.agent.workflow.aget_state(peer.runner.session.config)
            assert len([m for m in snapshot.values["messages"] if m.type == "human"]) == (4 if peer.name == "planner" else 2)


@pytest.mark.asyncio
async def test_real_exchange_bidirectional_messages_and_persistent_peers():
    factory = LocalExchangeFactory()
    received = {"one": [], "two": []}
    transports = {}
    runtimes = {}
    reply_seen = asyncio.Event()

    class Runner:
        interrupts = ()
        def __init__(self, name):
            self.name = name
        async def run(self, message):
            received[self.name].append(message.content)
            if message.content == "task":
                target = "two" if self.name == "one" else "one"
                await transports[self.name].send(envelope(self.name, target, "reply"))
            else:
                reply_seen.set()
            return PeerTurnResult("done")

    async with AsyncExitStack() as stack:
        for name in received:
            transports[name] = await stack.enter_async_context(AcademyTransport(
                exchange_factory=factory, run_id="run", launch_id="launch", name=name, timeout_s=2,
            ))
            runtime = PeerRuntime(name=name, run_id="run", launch_id="launch", allowed_senders=("one", "two"), runner=Runner(name))
            runtimes[name] = runtime
            await runtime.start()
            await transports[name].start(runtime.receive, runtime.status)
        try:
            await transports["one"].send(envelope("one", "two"))
            await asyncio.wait_for(reply_seen.wait(), 3)
            await asyncio.gather(*(runtime.inbox.join() for runtime in runtimes.values()))
            assert received == {"one": ["reply"], "two": ["task"]}
            assert (await transports["one"].peer_status("two"))["state"] == "idle"
            await transports["one"].send(envelope("one", "two", "another"))
            await asyncio.wait_for(runtimes["two"].inbox.join(), 3)
            assert received["two"] == ["task", "another"]
        finally:
            for runtime in runtimes.values():
                await runtime.close(timeout_s=0.05)


@pytest.mark.asyncio
async def test_missing_peer_and_wrong_launch_do_not_report_success():
    async with AcademyTransport(exchange_factory=LocalExchangeFactory(), run_id="run", launch_id="launch", timeout_s=0.1) as transport:
        with pytest.raises(TimeoutError, match="No acknowledgement"):
            await transport.send(envelope("operator", "missing"))
        with pytest.raises(ValueError, match="different launch"):
            await transport.send(envelope("operator", "missing", launch_id="stale"))


@pytest.mark.asyncio
async def test_duplicate_live_registrations_rejected():
    factory = LocalExchangeFactory()
    async def receive(message):
        return {"status": "accepted", "message_id": message.message_id}
    async with AsyncExitStack() as stack:
        transports = []
        for _ in range(2):
            transport = await stack.enter_async_context(AcademyTransport(
                exchange_factory=factory, run_id="run", launch_id="launch", name="same", timeout_s=2,
            ))
            await transport.start(receive, lambda: {"state": "idle"})
            transports.append(transport)
        with pytest.raises(RuntimeError, match="Multiple live"):
            await transports[0].send(envelope("same", "same"))


@pytest.mark.asyncio
async def test_http_exchange_with_independent_factories():
    from aiohttp.test_utils import TestServer
    from academy.exchange.cloud import HttpExchangeFactory
    from academy.exchange.cloud.app import create_app

    messages = []
    async def receive(message):
        messages.append(message)
        return {"status": "accepted", "message_id": message.message_id}

    async with TestServer(create_app()) as server:
        url = str(server.make_url(""))
        async with AcademyTransport(exchange_factory=HttpExchangeFactory(url=url), run_id="run",
                                    launch_id="launch", name="remote", timeout_s=3) as remote:
            await remote.start(receive, lambda: {"state": "idle"})
            async with AcademyTransport(exchange_factory=HttpExchangeFactory(url=url), run_id="run",
                                        launch_id="launch", timeout_s=3) as client:
                sent = envelope("operator", "remote")
                assert (await client.send(sent))["status"] == "accepted"
                assert messages == [sent]
