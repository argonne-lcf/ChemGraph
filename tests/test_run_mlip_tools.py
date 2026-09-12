from chemgraph.schemas.mlip_input import MLIPBatchInputSchema, MLIPInputSchema
from chemgraph.tools import run_mlip_tools


def test_run_mlip_langchain_tool_delegates(monkeypatch):
    params = MLIPInputSchema(
        input_structure_file="input.xyz",
        model={"family": "mace", "checkpoint": "model.pt"},
    )
    monkeypatch.setattr(
        run_mlip_tools,
        "run_mlip_core",
        lambda received: {"status": "single", "params": received},
    )

    result = run_mlip_tools.run_mlip.invoke({"params": params})

    assert result == {"status": "single", "params": params}


def test_run_mlip_batch_langchain_tool_delegates(monkeypatch):
    params = MLIPBatchInputSchema(
        input_structure_files=["input.xyz"],
        model={"family": "mace", "checkpoint": "model.pt"},
    )
    monkeypatch.setattr(
        run_mlip_tools,
        "run_mlip_batch_core",
        lambda received: {"status": "batch", "params": received},
    )

    result = run_mlip_tools.run_mlip_batch.invoke({"params": params})

    assert result == {"status": "batch", "params": params}
