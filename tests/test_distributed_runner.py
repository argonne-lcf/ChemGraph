import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from chemgraph.distributed.config import DistributedConfig
from chemgraph.distributed.peer_tools import build_peer_tool
from chemgraph.distributed.protocol import PeerEnvelope
from chemgraph.distributed.runner import ConfiguredRunner, ChemGraphRunner
from chemgraph.distributed.runtime import PeerRuntime
from chemgraph.mcp.config import MCPToolSourceConfig
from chemgraph.models.endpoints import PreparedModel
from chemgraph.observability.event_log import EventLog, read_events
from tests.test_main_agent import _ScriptedChatModel


def config(tmp_path, workflow=None):
    peer = {"model_name": "test", "options": {"enable_memory": False}}
    if workflow is not None:
        peer["workflow_type"] = workflow
    return DistributedConfig(run_id="run", launch_id="launch", exchange_url="http://unused", run_dir=tmp_path,
                             peers={"worker": peer})


def message(content="question"):
    return PeerEnvelope(run_id="run", launch_id="launch", sender="operator", recipient="worker", content=content)


@pytest.fixture
def scripted(monkeypatch, tmp_path):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    responses = []
    def load(**kwargs):
        return (_ScriptedChatModel(responses=list(responses)),
                PreparedModel(endpoint_name="test", protocol="openai_compatible", client_kwargs={"model": "test"}))
    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", load)
    return responses


@pytest.mark.asyncio
async def test_main_agent_turns_and_checkpoint_restore(tmp_path, scripted):
    settings = config(tmp_path)
    path = tmp_path / "events.jsonl"
    trace = lambda event, payload: EventLog(path).emit(event, payload=payload)  # noqa: E731
    scripted[:] = [AIMessage(content="first")]
    async with ConfiguredRunner(settings, "worker", [], trace) as runner:
        assert runner.agent.workflow_type == "main_agent"
        assert not runner.agent.tool_registry.names()
        assert not runner.agent.agent_registry.names()
        assert runner.agent.discover_skills is False
        assert runner.session.thread_id == "run/launch/worker"
        assert (await runner.run(message())).output == "first"
    scripted[:] = [AIMessage(content="second")]
    async with ConfiguredRunner(settings, "worker", [], trace) as runner:
        assert (await runner.run(message("follow-up"))).output == "second"
        snapshot = await runner.agent.workflow.aget_state(runner.session.config)
        assert [m.content for m in snapshot.values["messages"] if m.type == "ai"] == ["first", "second"]
    assert read_events(path)


@pytest.mark.asyncio
async def test_main_agent_approval_survives_restart(tmp_path, scripted):
    settings = config(tmp_path)
    scripted[:] = [AIMessage(content="", tool_calls=[{
        "name": "write_file", "args": {"file_path": "/note.txt", "content": "pending"},
        "id": "write", "type": "tool_call",
    }])]
    async with ConfiguredRunner(settings, "worker", [], lambda *_args: None) as runner:
        result = await runner.run(message("write a note"))
        assert result.status == "waiting_for_user"
        assert runner.interrupts
    scripted[:] = [AIMessage(content="rejected")]
    async with ConfiguredRunner(settings, "worker", [], lambda *_args: None) as runner:
        assert runner.interrupts
        result = await runner.resume({"decisions": [{"type": "reject"}]})
        assert result.status == "completed"
        assert result.output == "rejected"
        assert not runner.interrupts


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["single_agent", "multi_agent"])
async def test_legacy_workflows_write_state_files(tmp_path, scripted, workflow):
    scripted[:] = [AIMessage(content=("answer" if workflow == "single_agent" else
                                     json.dumps({"thought_process": "answer", "next_step": "FINISH"})))]
    settings = config(tmp_path, workflow)
    async with ConfiguredRunner(settings, "worker", [], lambda *_args: None) as runner:
        assert runner.session is None
        assert (await runner.run(message())).output == "answer"
    artifacts = settings.peer_dir("worker") / "artifacts"
    files = list(artifacts.glob("state_thread_*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["thread_id"] == "run%2Flaunch%2Fworker"
    assert not list(artifacts.glob("*/"))


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", ["main_agent", "single_agent"])
async def test_operator_reply_error_recovers_to_final_answer(tmp_path, scripted, workflow):
    transport = SimpleNamespace(send=AsyncMock())
    peer_tool = build_peer_tool(transport=transport, run_id="run", launch_id="launch", sender="worker",
                                allowed_peers=("executor",))
    scripted[:] = [
        AIMessage(content="", tool_calls=[{"name": "send_peer_message", "id": "bad-recipient",
                                          "args": {"recipient": "operator", "content": "energy result"}}]),
        AIMessage(content="energy result"), AIMessage(content="next answer"),
    ]
    async with ConfiguredRunner(config(tmp_path, workflow), "worker", [peer_tool], lambda *_args: None) as runner:
        runtime = PeerRuntime(name="worker", run_id="run", launch_id="launch", allowed_senders=(), runner=runner)
        await runtime.start()
        try:
            await runtime.receive(message())
            await asyncio.wait_for(runtime.inbox.join(), 5)
            assert runtime.status()["state"] == "idle"
            assert runtime.status()["output"] == "energy result"
            snapshot = await runner.agent.workflow.aget_state({"configurable": {"thread_id": runner.thread_id}})
            error = next(m for m in snapshot.values["messages"] if m.type == "tool")
            assert error.status == "error"
            assert "normal final assistant response" in error.content
            await runtime.receive(message("follow-up"))
            await asyncio.wait_for(runtime.inbox.join(), 5)
            assert runtime.status()["output"] == "next answer"
            transport.send.assert_not_called()
        finally:
            await runtime.close(timeout_s=0.05)


@pytest.mark.asyncio
async def test_mcp_allowlist_and_main_agent_approval(tmp_path, scripted, monkeypatch):
    calls = []
    @tool
    def run_ase() -> str:
        """Run a test calculation."""
        calls.append("run_ase")
        return "energy result"
    @tool
    def unwanted() -> str:
        """A tool this peer must not receive."""
        raise AssertionError("Unconfigured tool was executed")

    client = SimpleNamespace(get_tools=AsyncMock(return_value=[run_ase, unwanted]))
    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", lambda *_args: client)
    settings = config(tmp_path)
    settings.peers["worker"].mcp_servers = (MCPToolSourceConfig(
        name="science", url="http://unused/mcp", allowed_tools=("run_ase",),
    ),)
    scripted[:] = [AIMessage(content="", tool_calls=[{"name": "run_ase", "args": {}, "id": "calculate"}]),
                   AIMessage(content="energy result")]
    async with ConfiguredRunner(settings, "worker", [], lambda *_args: None) as runner:
        assert [item.name for item in runner.agent.tools] == ["run_ase"]
        assert not runner.agent.tool_registry.names()
        result = await runner.run(message("calculate"))
        assert result.status == "waiting_for_user"
        assert not calls
        result = await runner.resume({"decisions": [{"type": "approve"}]})
        assert result.output == "energy result"
        assert calls == ["run_ase"]


@pytest.mark.asyncio
async def test_failed_main_agent_checkpoint_can_retry_after_restart(tmp_path, scripted):
    settings = config(tmp_path)
    scripted[:] = [RuntimeError("model unavailable")]
    async with ConfiguredRunner(settings, "worker", [], lambda *_args: None) as runner:
        with pytest.raises(RuntimeError, match="model unavailable"):
            await runner.run(message())
        assert runner.failed
    scripted[:] = [AIMessage(content="recovered")]
    async with ConfiguredRunner(settings, "worker", [], lambda *_args: None) as runner:
        assert runner.failed
        assert (await runner.retry()).output == "recovered"
        snapshot = await runner.agent.workflow.aget_state(runner.session.config)
        assert len([m for m in snapshot.values["messages"] if m.type == "human"]) == 1


@pytest.mark.asyncio
async def test_explicit_specialist_and_skill_options_remain_available(tmp_path, scripted):
    settings = config(tmp_path)
    settings.peers["worker"].options.update(subagent_names=["single_agent"], skills=["/skills"])
    async with ConfiguredRunner(settings, "worker", [], lambda *_args: None) as runner:
        assert runner.agent.agent_registry.names() == ("single_agent",)
        assert runner.agent.skills == ("/skills",)


@pytest.mark.asyncio
async def test_legacy_operator_response_uses_existing_handler(tmp_path):
    class Agent:
        log_dir = str(tmp_path)
        async def run(self, prompt, config):
            answer = await self.human_input_handler("Which calculator?", {"question": "calculator"})
            return SimpleNamespace(content=answer)
    runner = ChemGraphRunner(Agent(), thread_id="test")
    task = asyncio.create_task(runner.run(message()))
    await asyncio.sleep(0)
    assert runner.interrupts
    assert await runner.resume("EMT") is None
    assert (await task).output == "EMT"
    assert not runner.interrupts
