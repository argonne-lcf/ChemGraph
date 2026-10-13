import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from chemgraph.distributed.config import DistributedConfig
from chemgraph.distributed.protocol import PeerEnvelope
from chemgraph.distributed.runner import ConfiguredRunner, ChemGraphRunner
from chemgraph.models.endpoints import PreparedModel
from tests.test_main_agent import _ScriptedChatModel


def config(tmp_path, workflow="main_agent"):
    return DistributedConfig(run_id="run", launch_id="launch", exchange_url="http://unused", run_dir=tmp_path,
                             peers={"worker": {"workflow_type": workflow, "model_name": "test", "options": {
                                 "discover_skills": False, "subagent_names": [], "enable_memory": False,
                             } if workflow == "main_agent" else {"enable_memory": False}}})


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
    scripted[:] = [AIMessage(content="first")]
    async with ConfiguredRunner(settings, "worker", [], lambda *_args: None) as runner:
        assert (await runner.run(message())).output == "first"
    scripted[:] = [AIMessage(content="second")]
    async with ConfiguredRunner(settings, "worker", [], lambda *_args: None) as runner:
        assert (await runner.run(message("follow-up"))).output == "second"
        snapshot = await runner.agent.workflow.aget_state(runner.session.config)
        assert [m.content for m in snapshot.values["messages"] if m.type == "ai"] == ["first", "second"]


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
async def test_legacy_workflows_keep_chemgraph_run_interface(tmp_path, monkeypatch, workflow):
    calls = []
    class Agent:
        def __init__(self, **kwargs):
            self.log_dir = kwargs["log_dir"]
            calls.append(kwargs)
        async def run(self, prompt, config):
            calls.append((prompt, config))
            return SimpleNamespace(content="answer")
    monkeypatch.setattr("chemgraph.agent.llm_agent.ChemGraph", Agent)
    async with ConfiguredRunner(config(tmp_path, workflow), "worker", [], lambda *_args: None) as runner:
        assert (await runner.run(message())).output == "answer"
    assert calls[0]["workflow_type"] == workflow
    assert calls[1][1] == {"thread_id": "run/launch/worker"}


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
