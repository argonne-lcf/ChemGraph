import pytest
pytest.importorskip("a2a")

import asyncio
from typing import Any



from a2a.types import a2a_pb2 as a2a
from chemgraph.models.endpoints import PreparedModel
from tests.service.conftest import running
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from chemgraph.service.adapters import ChemGraphAdapter
from chemgraph.service.config import Profile




class ScriptedModel(BaseChatModel):
    responses: list[Any]
    index: int = 0

    @property
    def _llm_type(self):
        return "connect-scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


def call(name, **args):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": name, "type": "tool_call"}]
    )


@pytest.fixture
def model(monkeypatch):
    responses = []

    def load(**kwargs):
        return ScriptedModel(responses=list(responses)), PreparedModel(
            endpoint_name="test", protocol="openai_compatible", client_kwargs={"model": "test"}
        )

    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", load)
    return responses


async def test_native_catalog_checkpoint_followup(settings, model):
    from tests.service.test_operations import approve

    model[:] = [
        call("search_tools", query="structure"),
        call("load_tools", names=["smiles_to_atomsdata"]),
        call("smiles_to_atomsdata", smiles="O"),
        AIMessage(content="Water."),
    ]
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, app):
        task = (await client.send("Prepare water", message_id="science-1", blocking=True)).task
        assert task.status.state == a2a.TASK_STATE_INPUT_REQUIRED
        await approve(service, app, task.id)
        task = await service.wait(task.id, "alice")
        assert task.status.state == a2a.TASK_STATE_COMPLETED
        assert (await service.operations.list(task.id))[0]["tool"] == "smiles_to_atomsdata"
        thread = (await service.store.get(task.id))["thread_id"]
    model[:] = [AIMessage(content="This is the same scientific conversation.")]
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, _):
        duplicate = await client.send("Prepare water", message_id="science-1")
        assert duplicate.task.id == task.id
        followup = (await client.send("Explain", context_id=task.context_id, blocking=True)).task
        assert followup.status.state == a2a.TASK_STATE_COMPLETED
        record = await service.store.get(followup.id)
        assert record["thread_id"] == thread
        session = await service.adapter.session(record)
        snapshot = await session.workflow.aget_state(session.config)
        assert [m.content for m in snapshot.values["messages"] if isinstance(m, HumanMessage)] == [
            "Prepare water", "Explain",
        ]


async def test_real_interrupt_restart_resume(settings, model):
    settings = settings.model_copy(update={"profile": Profile(adapter="chemgraph")})
    model[:] = [call("ask_human", question="Which molecule?")]
    async with running(settings, ChemGraphAdapter(settings)) as (_, client, _):
        task = (await client.send("Ask me", blocking=True)).task
        assert task.status.state == a2a.TASK_STATE_INPUT_REQUIRED
    model[:] = [AIMessage(content="Understood: water.")]
    async with running(settings, ChemGraphAdapter(settings)) as (_, client, _):
        resumed = (await client.send("water", context_id=task.context_id, blocking=True)).task
        assert resumed.id == task.id
        assert resumed.status.state == a2a.TASK_STATE_COMPLETED


class AfterGraphAdapter(ChemGraphAdapter):
    def __init__(self, settings):
        super().__init__(settings)
        self.finished = asyncio.Event()

    async def run(self, record):
        await super().run(record)
        self.finished.set()
        await asyncio.Event().wait()


async def test_crash_after_checkpoint_before_task_projection(settings, model):
    settings = settings.model_copy(update={"profile": Profile(adapter="chemgraph")})
    model[:] = [AIMessage(content="Durable answer")]
    adapter = AfterGraphAdapter(settings)
    async with running(settings, adapter) as (_, client, _):
        task = (await client.send("once", message_id="once")).task
        await asyncio.wait_for(adapter.finished.wait(), 20)
    model[:] = []  # Recovery must not invoke a model or add another human message.
    async with running(settings, ChemGraphAdapter(settings)) as (_, client, _):
        recovered = await client.get(task.id)
        assert recovered.status.state == a2a.TASK_STATE_COMPLETED
        assert recovered.status.message.parts[0].text == "Durable answer"


@pytest.mark.parametrize("submission_delay", [0, 3.2], ids=["immediate", "delayed"])
async def test_real_guarded_mcp_tool_pauses_and_resumes(settings, model, submission_delay):
    from tests.service.test_operations import FakeBackend, approve, wait_backend

    from chemgraph.service.config import HPC, Calculation

    class DelayedBackend(FakeBackend):
        async def submit(self, operation):
            # Cover valid execution that exceeds the former three-second deadline.
            await asyncio.sleep(submission_delay)
            return await super().submit(operation)

    hpc = HPC(
        server_url="https://mcp.example/mcp",
        endpoint_id="test",
        tool="run_ase",
        calculator="mace_mp",
        model="installed",
        scheduler_limits="10 min",
        calculations=(Calculation(name="five-conformers", arguments={"params": {}}),),
    )
    settings = settings.model_copy(update={"profile": Profile(adapter="chemgraph", hpc=hpc)})
    model[:] = [call("submit_hpc", calculation="five-conformers"), AIMessage(content="Submitted.")]
    adapter = ChemGraphAdapter(settings)
    adapter.backend = DelayedBackend()
    async with running(settings, adapter) as (service, client, app):
        task = (await client.send("submit fixed study", blocking=True)).task
        assert task.status.state == a2a.TASK_STATE_INPUT_REQUIRED
        assert task.metadata["waiting_reason"] == "approval"
        assert adapter.backend.submissions == 0
        await approve(service, app, task.id)
        await wait_backend(service, task.id)
        assert adapter.backend.submissions == 1
        assert (await client.get(task.id)).status.state == a2a.TASK_STATE_WORKING


async def test_native_batch_approvals_and_immutable_artifacts(settings, model):
    from pathlib import Path

    from tests.service.test_operations import approve

    settings = settings.model_copy(update={"profile": Profile(adapter="chemgraph")})
    model[:] = [
        call("load_tools", names=["save_atomsdata_to_file"]),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_atomsdata_to_file",
                    "id": "write-one",
                    "args": {
                        "atomsdata": {"numbers": [1], "positions": [[0, 0, 0]]},
                        "fname": "output.xyz",
                    },
                },
                {
                    "name": "save_atomsdata_to_file",
                    "id": "write-two",
                    "args": {
                        "atomsdata": {"numbers": [2], "positions": [[0, 0, 0]]},
                        "fname": "output.xyz",
                    },
                },
            ],
        ),
        AIMessage(content="Both structures were written."),
    ]
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, app):
        task = (await client.send("Write two structures", blocking=True)).task
        assert task.status.state == a2a.TASK_STATE_INPUT_REQUIRED
        operations = await service.operations.list(task.id)
        assert len(operations) == 2
        record = await service.store.get(task.id)
        workspace = settings.state_dir / "artifacts" / record["thread_id"]
        assert not (workspace / "output.xyz").exists()
        with pytest.raises(Exception, match="operator"):
            await client.send("approved", task_id=task.id)
        await approve(service, app, task.id, operations[0]["id"])
        assert (await client.get(task.id)).status.state == a2a.TASK_STATE_INPUT_REQUIRED
        assert not (workspace / "output.xyz").exists()
        await approve(service, app, task.id, operations[1]["id"])
        task = await service.wait(task.id, "alice")
        assert task.status.state == a2a.TASK_STATE_COMPLETED, task
        assert len(task.artifacts) == 2
        bodies = [
            (await client.http.get("/artifacts/" + a.artifact_id)).text for a in task.artifacts
        ]
        assert any("H " in body for body in bodies)
        assert any("He " in body for body in bodies)
        assert all(op["state"] == "completed" for op in await service.operations.list(task.id))
        assert Path(workspace / "output.xyz").exists()
    model[:] = []
    async with running(settings, ChemGraphAdapter(settings)) as (_, client, _):
        restored = await client.get(task.id)
        assert [a.artifact_id for a in restored.artifacts] == [
            a.artifact_id for a in task.artifacts
        ]


async def test_native_approval_restores_checkpoint(settings, model):
    from tests.service.test_operations import approve

    settings = settings.model_copy(update={"profile": Profile(adapter="chemgraph")})
    model[:] = [
        call("load_tools", names=["save_atomsdata_to_file"]),
        call(
            "save_atomsdata_to_file",
            atomsdata={"numbers": [1], "positions": [[0, 0, 0]]},
            fname="structure.xyz",
        ),
    ]
    async with running(settings, ChemGraphAdapter(settings)) as (_, client, _):
        task = (await client.send("Write a structure", blocking=True)).task
        assert task.status.state == a2a.TASK_STATE_INPUT_REQUIRED
    model[:] = [AIMessage(content="Saved.")]
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, app):
        await approve(service, app, task.id)
        finished = await service.wait(task.id, "alice")
        assert finished.status.state == a2a.TASK_STATE_COMPLETED, finished
        assert len(finished.artifacts) == 1


async def test_unsupported_workspace_mutation_cannot_be_approved(settings, model):
    model[:] = [call("write_file", file_path="/draft.txt", content="notes")]
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, _):
        task = (await client.send("Save notes", blocking=True)).task
        operation = (await service.operations.list(task.id))[0]
        assert operation["state"] == "blocked"
        assert "preparation" in operation["preparation_error"]
        with pytest.raises(ValueError, match="awaiting approval"):
            await service.operations.approve(operation["id"], {}, "operator")


async def test_native_approval_expiry_prevents_resume(settings, model):
    from tests.service.test_operations import approve

    settings = settings.model_copy(update={"profile": Profile(adapter="chemgraph")})
    model[:] = [call("load_tools", names=["smiles_to_coordinate_file"]),
                call("smiles_to_coordinate_file", smiles="O")]
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, app):
        task = (await client.send("Save notes", blocking=True)).task
        # Pause the worker so the expiry check is deterministic.
        service.worker.cancel()
        await asyncio.gather(service.worker, return_exceptions=True)
        operation = await approve(service, app, task.id)
        operation = await service.operations.get(operation["id"])
        operation["approval"]["expires"] = 0
        async with service.lock:
            async with service.store.transaction():
                await service.operations.put(operation)
        record = await service.store.get(task.id)
        outcome = await service.adapter.run(record)
        assert outcome.state == "input_required"
        assert outcome.reason == "approval"
        assert (await service.operations.get(operation["id"]))["state"] == "approval"


pytestmark = pytest.mark.asyncio
