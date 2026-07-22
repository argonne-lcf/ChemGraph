"""Tests for the FairChem/UMA tool stack.

Mirrors ``tests/test_faircalc.py`` but exercises the dedicated
``run_fairchem`` surface (schema + core adapter). The schema->ASE
conversion test runs without FairChem installed; the end-to-end runs
are skipped unless ``fairchem.core`` is available (and the UMA model is
HF-gated, so they also require authentication at run time).
"""

from pathlib import Path
import importlib.util
import json

import pytest

from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.schemas.fairchem_schema import fairchem_input_schema
from chemgraph.tools.fairchem_tools import (
    _fairchem_input_to_ase_input,
    run_fairchem_core,
)

TEST_DIR = Path(__file__).parent


def is_fairchem_installed():
    try:
        return importlib.util.find_spec("fairchem.core") is not None
    except (ImportError, ModuleNotFoundError):
        return False


@pytest.fixture
def base_fairchem_input():
    """Base fixture for FairChem input with common parameters."""
    return {
        "input_structure_file": str(TEST_DIR / "water.xyz"),
        "output_result_file": str(TEST_DIR / "water_output.json"),
        "device": "cpu",
    }


def test_fairchem_schema_defaults(base_fairchem_input):
    """The FairChem input schema builds with sensible UMA defaults
    (does not require fairchem to be installed)."""
    params = fairchem_input_schema(driver="energy", **base_fairchem_input)

    assert params.model_name == "uma-s-1p1"
    assert params.inference_settings == "default"
    assert params.charge == 0
    assert params.multiplicity == 1
    assert params.driver == "energy"
    assert params.output_result_file == base_fairchem_input["output_result_file"]


@pytest.mark.skipif(not is_fairchem_installed(), reason="FairChem is not installed")
def test_schema_to_ase_conversion(base_fairchem_input):
    """The FairChem schema converts to a valid ASEInputSchema carrying a
    FAIRChem calculator. Requires fairchem installed because ASEInputSchema
    only permits calculators whose engine is available."""
    params = fairchem_input_schema(driver="energy", **base_fairchem_input)
    ase_input = _fairchem_input_to_ase_input(params)

    assert isinstance(ase_input, ASEInputSchema)
    assert ase_input.driver == "energy"
    assert ase_input.input_structure_file == base_fairchem_input["input_structure_file"]
    # output_result_file -> output_results_file
    assert ase_input.output_results_file == base_fairchem_input["output_result_file"]

    calc = ase_input.calculator
    calc_type = (
        calc.get("calculator_type")
        if isinstance(calc, dict)
        else getattr(calc, "calculator_type", None)
    )
    assert "fairchem" in str(calc_type).lower()
    model_name = (
        calc.get("model_name")
        if isinstance(calc, dict)
        else getattr(calc, "model_name", None)
    )
    assert model_name == "uma-s-1p1"


@pytest.mark.skipif(not is_fairchem_installed(), reason="FairChem is not installed")
def test_run_fairchem_energy(base_fairchem_input):
    """Test a single-point energy calculation via run_fairchem_core."""
    params = fairchem_input_schema(driver="energy", **base_fairchem_input)
    result = run_fairchem_core(params)

    assert isinstance(result, dict)
    assert result["single_point_energy"] is not None
    assert result["unit"] == "eV"


@pytest.mark.skipif(not is_fairchem_installed(), reason="FairChem is not installed")
def test_run_fairchem_opt(base_fairchem_input):
    """Test geometry optimization via run_fairchem_core."""
    params = fairchem_input_schema(driver="opt", **base_fairchem_input)
    result = run_fairchem_core(params)

    assert isinstance(result, dict)
    assert result["status"]

    output_file = Path(base_fairchem_input["output_result_file"])
    assert output_file.exists()
    with open(output_file) as f:
        data = json.load(f)
    assert data["simulation_input"]["driver"] == "opt"
