"""Offline Python API example: python examples/run_service_local.py."""

import asyncio
import tempfile

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from chemgraph.service import RunService


class DemoModel(BaseChatModel):
    """A deterministic conversation; replace with your configured chat model."""

    index: int = 0
    responses: list = Field(default_factory=lambda: [
        AIMessage(content="", tool_calls=[{
            "id": "structure", "name": "save_atomsdata_to_file", "args": {
                "atomsdata": {"numbers": [1, 1], "positions": [[0, 0, 0], [0, 0, 0.9]]},
                "fname": "h2.xyz",
            },
        }]),
        AIMessage(content="", tool_calls=[{
            "id": "energy", "name": "run_ase", "args": {"params": {
                "input_structure_file": "h2.xyz", "calculator": {"calculator_type": "emt"},
                "driver": "energy", "output_results_file": "energy.json",
            }},
        }]),
        AIMessage(content="H2 energy calculated using EMT."),
    ])

    @property
    def _llm_type(self):
        return "offline-service-demo"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self.responses[self.index]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=response)])


async def main():
    root = tempfile.mkdtemp(prefix="chemgraph-service-demo-")
    async with RunService(root, model_factory=lambda _: DemoModel(),
                          configuration_id="offline-demo-v1") as api:
        session = await api.create_session()
        run = await api.start_run(session.id, "Build H2 and calculate EMT energy.")
        while True:
            run = await api.wait(run.id)
            if run.status != "waiting_for_user":
                break
            for pending in run.interrupts:
                if pending.kind == "approval":
                    # Only this fixed demonstration auto-approves its scripted calls.
                    print("Review:", pending.payload["prepared"]["tool"])
                    await api.decide(run.id, pending.operation_id,
                                     digest=pending.payload["digest"], approve=True)
        if run.status != "completed":
            raise RuntimeError(f"{run.status}: {run.error}")
        print(run.assistant_response)
        for artifact in await api.list_artifacts(run.id):
            print(artifact.source_path, "->", await api.resolve_artifact(run.id, artifact.id))
    print("Persistent data:", root)


if __name__ == "__main__":
    asyncio.run(main())
