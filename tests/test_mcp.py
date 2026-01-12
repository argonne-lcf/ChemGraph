from pathlib import Path


import pytest
import json
from chemgraph.mcp.mcp_tools import mcp  # Import your FastMCP instance
from fastmcp import Client
from mcp.types import TextContent

TEST_DIR = Path(__file__).parent

@pytest.fixture
def base_ase_input_dict():
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
