import json

import pytest
from fastmcp import Client

from chemgraph.mcp import run_mlip_mcp
from chemgraph.schemas.mlip_input import MLIPBatchInputSchema, MLIPInputSchema


@pytest.mark.asyncio
async def test_run_mlip_mcp_exposes_standalone_tools(monkeypatch):
    monkeypatch.setattr(
        run_mlip_mcp,
        "run_mlip_core",
        lambda params: {"status": "single", "input": params.input_structure_file},
    )
    monkeypatch.setattr(
        run_mlip_mcp,
        "run_mlip_batch_core",
        lambda params: {"status": "batch", "total": len(params.input_structure_files)},
    )
    single = MLIPInputSchema(
        input_structure_file="input.xyz",
        model={"provider": "mace", "checkpoint": "model.pt"},
    )
    batch = MLIPBatchInputSchema(
        input_structure_files=["one.xyz", "two.xyz"],
        model={"provider": "mace", "checkpoint": "model.pt"},
    )

    async with Client(run_mlip_mcp.mcp) as client:
        tools = await client.list_tools()
        single_result = await client.call_tool(
            "run_mlip", {"params": single.model_dump(mode="json")}
        )
        batch_result = await client.call_tool(
            "run_mlip_batch", {"params": batch.model_dump(mode="json")}
        )

    assert {tool.name for tool in tools} == {"run_mlip", "run_mlip_batch"}
    assert json.loads(single_result.content[0].text) == {
        "status": "single",
        "input": "input.xyz",
    }
    assert json.loads(batch_result.content[0].text) == {
        "status": "batch",
        "total": 2,
    }
