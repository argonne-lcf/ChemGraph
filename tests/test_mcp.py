"""Test suite for MCP servers."""

import json
from pathlib import Path

import pytest

try:
    from mcp.types import TextContent
    from fastmcp import Client
    from chemgraph.mcp.mcp_tools import mcp
    from chemgraph.mcp.data_analysis_mcp import mcp as data_mcp
except ModuleNotFoundError:
    pytest.skip("MCP test dependencies are not installed", allow_module_level=True)

TEST_DIR = Path(__file__).parent


@pytest.mark.asyncio
async def test_split_cif_dataset(tmp_path):
    """Test splitting a dataset of CIF files."""
    # Create dummy CIF files
    input_dir = tmp_path / "input_cifs"
    input_dir.mkdir()
    for i in range(5):
        (input_dir / f"mof_{i}.cif").touch()

    output_root = tmp_path / "batches"

    async with Client(data_mcp) as client:
        # Test split by batch_size
        res = await client.call_tool(
            "split_cif_dataset",
            {
                "input_dir": str(input_dir),
                "output_root": str(output_root),
                "batch_size": 2,
            },
        )
        assert "Success" in res.content[0].text
        assert "batches" in res.content[0].text

        # Verify output structure
        assert (output_root / "batch_000").exists()
        assert (output_root / "batch_001").exists()
        assert (output_root / "batch_002").exists()  # 5 files / 2 = 3 batches
        assert len(list((output_root / "batch_000").iterdir())) == 2


@pytest.fixture(name="base_ase_input_dict")
def fixture_base_ase_input_dict():
    """Fixture providing base ASE input parameters."""
    return {
        "input_structure_file": str(TEST_DIR / "water.xyz"),
        "output_results_file": str(TEST_DIR / "water_output.json"),
        "optimizer": "bfgs",
        "calculator": {
            "calculator_type": "mace_mp",
        },
    }


@pytest.mark.asyncio
async def test_run_ase_energy_simple(base_ase_input_dict):
    """Simplified test using FastMCP's in-memory client."""

    input_data = base_ase_input_dict.copy()
    input_data["driver"] = "energy"

    async with Client(mcp) as client:
        res = await client.call_tool("run_ase", {"params": input_data})
        result_dict = json.loads(res.content[0].text)
        assert result_dict["status"] == "success"
        assert "single_point_energy" in result_dict

        assert isinstance(res.content[0], TextContent)
        assert "Simulation completed" in res.content[0].text


@pytest.mark.asyncio
async def test_run_ase_thermo_simple(base_ase_input_dict):
    """Simplified thermochemistry test"""
    input_data = base_ase_input_dict.copy()
    input_data["driver"] = "thermo"
    input_data["temperature"] = 298.15

    async with Client(mcp) as client:
        res = await client.call_tool("run_ase", {"params": input_data})
        result_dict = json.loads(res.content[0].text)

        assert result_dict["status"] == "success"
        assert "result" in result_dict

        assert isinstance(res.content[0], TextContent)
        assert "Thermochemistry computed and returned" in res.content[0].text


@pytest.mark.asyncio
async def test_aggregate_and_rank(tmp_path):
    """Test aggregating simulation results and ranking them."""
    # Create dummy JSONL output
    log_file = tmp_path / "simulations.jsonl"
    dummy_data = [
        {
            "status": "success",
            "cif_path": "/abs/path/to/mof_1.cif",
            "uptake_in_mol_kg": 10.5,
            "temperature_in_K": 298.0,
            "pressure_in_Pa": 100000.0,
        },
        {
            "status": "success",
            "cif_path": "/abs/path/to/mof_2.cif",
            "uptake_in_mol_kg": 2.1,
            "temperature_in_K": 298.15,
            "pressure_in_Pa": 100000.0,
        },
        {"status": "failure", "cif_path": "/abs/path/to/mof_3.cif"},
    ]

    with open(log_file, "w", encoding="utf-8") as f:
        for entry in dummy_data:
            f.write(json.dumps(entry) + "\n")

    output_csv = tmp_path / "results.csv"

    async with Client(data_mcp) as client:
        # 1. Aggregate
        res_agg = await client.call_tool(
            "aggregate_simulation_results",
            {"file_paths": [str(log_file)], "output_csv_path": str(output_csv)},
        )
        assert "Success" in res_agg.content[0].text
        assert output_csv.exists()

        # 2. Ranking
        # Note: We simulate a ranking query.
        res_rank = await client.call_tool(
            "rank_mofs_performance",
            {
                "input_csv_path": str(output_csv),
                "ads_pressure": 100000.0,
                "ads_temp": 298.0,
                "top_percentile": 1.0,  # Return all
            },
        )
        text = res_rank.content[0].text
        assert "Analysis Complete" in text
        assert "mof_1.cif" in text
        assert "mof_2.cif" in text  # Should find both due to tolerance
