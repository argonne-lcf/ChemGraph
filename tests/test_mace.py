"""Opt-in integration coverage for real MACE-MP inference."""

import json
import math
import os
from pathlib import Path

import pytest

from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.tools.ase_tools import run_ase


@pytest.mark.skipif(
    os.environ.get("CHEMGRAPH_TEST_MACE") != "1",
    reason="real MACE inference requires CHEMGRAPH_TEST_MACE=1 (MACE CI job)",
)
def test_run_ase_mace_mp_energy(monkeypatch, tmp_path):
    """Load the default pretrained model and calculate a water single point."""
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    output = tmp_path / "water_result.json"
    params = ASEInputSchema(
        input_structure_file=str(Path(__file__).with_name("water.xyz")),
        output_results_file=str(output),
        driver="energy",
        calculator={"calculator_type": "mace_mp", "device": "cpu"},
    )

    result = run_ase.invoke({"params": params})

    assert result["status"] == "success", result
    assert result["driver"] == "energy"
    assert isinstance(result["potential_energy"], float)
    assert math.isfinite(result["potential_energy"])
    assert result["energy_unit"] == "eV"
    assert Path(result["results_file"]) == output
    data = json.loads(output.read_text())
    assert data["success"] is True
    assert data["simulation_input"]["calculator"]["calculator_type"] == "mace_mp"
    assert data["potential_energy"] == result["potential_energy"]
    assert data["single_point_energy"] == result["potential_energy"]
