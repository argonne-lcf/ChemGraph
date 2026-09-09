"""Simulation failures must reach every caller without successful artifacts."""

import json
from unittest.mock import Mock

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.calculator import Calculator
from ase.calculators.emt import EMT
from ase.io import write

from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.schemas.calculators.emt_calc import EMTCalc
from chemgraph.schemas.calculators.mace_calc import MaceCalc
from chemgraph.tools import ase_core
from chemgraph.utils import calculator_defaults


@pytest.fixture
def params(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(ase_core, "_ensure_ase_core_file_log", lambda: None)
    write(tmp_path / "water.xyz", molecule("H2O"))
    return ASEInputSchema(
        input_structure_file=str(tmp_path / "water.xyz"),
        output_results_file=str(tmp_path / "result.json"),
        calculator=EMTCalc(), driver="energy",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["core", "langchain", "mcp"])
@pytest.mark.parametrize("inputs", [
    {"calculator_type": "mace_polar"}, {"charge": 1},
    {"multiplicity": 2}, {"external_field": (0.1, 0, 0)},
])
async def test_missing_polar_returns_install_guidance(params, monkeypatch, tmp_path, entrypoint, inputs):
    import mace.calculators

    monkeypatch.setattr(calculator_defaults, "mace_polar_available", lambda: False)
    loader = Mock(side_effect=AssertionError("must not download model weights"))
    monkeypatch.setattr(mace.calculators, "mace_polar", loader)
    monkeypatch.setattr(mace.calculators, "mace_mp", loader)
    params.calculator = MaceCalc(**inputs)
    if entrypoint == "core":
        result = ase_core.run_ase_core(params)
    elif entrypoint == "langchain":
        from chemgraph.tools.ase_tools import run_ase
        result = run_ase.invoke({"params": params.model_dump()})
    else:
        from chemgraph.mcp.mcp_tools import run_ase
        result = await run_ase(params)
    assert result["status"] == "failure"
    assert result["error_type"] == "ImportError"
    assert "requirements/mace-polar.txt" in result["message"]
    loader.assert_not_called()
    assert not (tmp_path / "result.json").exists()


def test_inferred_polar_forwards_and_records_physical_inputs(params, monkeypatch, tmp_path):
    engine = EMT()
    monkeypatch.setattr(calculator_defaults, "mace_polar_available", lambda: True)
    monkeypatch.setattr(MaceCalc, "get_calculator", lambda self: engine)
    params.calculator = MaceCalc(charge=1, multiplicity=2, external_field=(0.1, 0, 0))
    result = ase_core.run_ase_core(params)
    assert result["status"] == "success", result
    assert engine.atoms.info == {"charge": 1, "spin": 2, "external_field": [0.1, 0, 0]}
    recorded = json.loads((tmp_path / "result.json").read_text())["simulation_input"]["calculator"]
    assert recorded["calculator_type"] == "mace_polar"
    assert recorded["model"] == "polar-1-m"
    assert recorded["charge"] == 1
    assert recorded["multiplicity"] == 2
    assert recorded["external_field"] == [0.1, 0, 0]


@pytest.mark.parametrize("driver", ["dipole", "ir"])
def test_unsupported_dipole_fails_before_optimization(params, monkeypatch, tmp_path, driver):
    import ase.optimize
    import ase.vibrations

    optimization = Mock(side_effect=AssertionError("must not optimize"))
    vibrations = Mock(side_effect=AssertionError("must not run vibrations"))
    monkeypatch.setattr(ase.optimize, "BFGS", optimization)
    monkeypatch.setattr(ase.vibrations, "Vibrations", vibrations)
    params.driver = driver
    params.calculator = MaceCalc(calculator_type="mace_mp")
    monkeypatch.setattr(MaceCalc, "get_calculator", lambda self: EMT())
    result = ase_core.run_ase_core(params)
    assert result["status"] == "failure"
    assert result["error_type"] == "PropertyNotImplementedError"
    assert "dipole-capable" in result["message"]
    optimization.assert_not_called()
    vibrations.assert_not_called()
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("dipole", [None, [1, 2], [1, 2, 3, 4], [0, np.nan, 1], [0, np.inf, 1], [0, "bad", 1], [0, 0, 0]])
def test_dipole_result_requires_three_finite_components(params, monkeypatch, tmp_path, dipole):
    class DipoleCalculator(Calculator):
        implemented_properties = ["energy", "dipole"]

        def calculate(self, atoms=None, properties=None, system_changes=None):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": -1.0, "dipole": dipole}

    params.driver = "dipole"
    monkeypatch.setattr(EMTCalc, "get_calculator", lambda self: DipoleCalculator())
    result = ase_core.run_ase_core(params)
    if dipole == [0, 0, 0]:
        assert result["status"] == "success"
        assert result["dipole_moment"] == [0, 0, 0]
        assert json.loads((tmp_path / "result.json").read_text())["success"] is True
    else:
        assert result["status"] == "failure"
        assert not (tmp_path / "result.json").exists()


def test_single_point_engine_failure_returns_original_message(params, monkeypatch, tmp_path):
    monkeypatch.setattr(EMT, "get_potential_energy", Mock(side_effect=RuntimeError("engine failed")))
    result = ase_core.run_ase_core(params)
    assert result == {"status": "failure", "error_type": "RuntimeError", "message": "engine failed"}
    assert not (tmp_path / "result.json").exists()


def test_ir_with_dipole_support_still_completes(params, monkeypatch, tmp_path):
    class DipoleEMT(EMT):
        def get_dipole_moment(self, atoms=None):
            return np.zeros(3)

    monkeypatch.setattr(EMTCalc, "get_calculator", lambda self: DipoleEMT())
    params.driver = "ir"
    params.fmax = 100  # Keep the fixture geometry; exercise the full IR path.
    result = ase_core.run_ase_core(params)
    assert result["status"] == "success", result
    output = json.loads((tmp_path / "result.json").read_text())
    assert output["success"] is True
    assert (tmp_path / "ir_spectrum_water.csv").exists()
    assert (tmp_path / "ir_peaks_water.csv").exists()
