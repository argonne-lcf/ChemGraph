"""Shared fixtures for chemgraph.runtime tests: real EMT, no models or network."""

from langchain_core.messages import AIMessage

from chemgraph.runtime import ExecutionPolicy, FileOperationStore, OperationJournal, RunContext

H2 = "2\nEMT test\nH 0 0 0\nH 0 0 0.9\n"


def make_context(tmp_path, *, owner="alice", thread="thread-1", task="task-1",
                 journal=None, **policy):
    policy.setdefault("calculators", frozenset({"emt"}))
    context = RunContext(
        owner=owner, thread_id=thread, task_id=task,
        workspace=tmp_path / "workspace" / thread,
        policy=ExecutionPolicy(**policy),
        journal=journal or OperationJournal(FileOperationStore(tmp_path / "journal")),
    )
    (context.workspace / "h2.xyz").write_text(H2)
    return context


def energy_args(output="energy.json", structure="h2.xyz", driver="energy", **params):
    return {"params": {
        "input_structure_file": structure, "output_results_file": output,
        "calculator": {"calculator_type": "emt"}, "driver": driver, **params,
    }}


def run_ase_call(identifier="energy", **kwargs):
    return {"name": "run_ase", "args": energy_args(**kwargs), "id": identifier, "type": "tool_call"}


def ai(*calls, content=""):
    return AIMessage(content=content, tool_calls=list(calls))


async def approve(context, record, **kwargs):
    return await context.journal.decide(
        record["id"], digest=record["prepared"]["digest"], approve=True,
        decided_by="operator", **kwargs,
    )
