"""Hermetic gRASPA tests exercise the public tool, real preparation and parser."""

from concurrent.futures import ThreadPoolExecutor
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

from ase import Atoms
from ase.io import write
import pytest

from chemgraph.registry import AgentRegistry, ToolRegistry
from chemgraph.schemas.graspa_schema import (
    graspa_input_schema,
    graspa_input_schema_ensemble,
)
from chemgraph.tools import graspa_core
from chemgraph.tools.graspa_tools import run_graspa
from chemgraph.utils.executables import resolve_executable


@pytest.fixture
def simulation(tmp_path, monkeypatch):
    source = tmp_path / "source with spaces.CIF"
    write(source, Atoms("C", cell=[30, 30, 30], pbc=True), format="cif")
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("CHEMGRAPH_GRASPA_EXECUTABLE", sys.executable)

    def execute(command, **kwargs):
        kwargs["stdout"].write(
            "Input UnitCells [0] = 1 1 1\nOverall: Average: 12.011,\n"
        )
        return subprocess.CompletedProcess(command, 0)

    runner = Mock(side_effect=execute)
    monkeypatch.setattr(graspa_core.subprocess, "run", runner)
    return graspa_input_schema(
        input_structure_file=str(source), adsorbate="H2O", n_cycles=10
    ), runner


def test_public_tool_runs_and_preserves_float_contract(simulation):
    params, runner = simulation
    assert run_graspa.invoke({"graspa_input": params.model_dump()}) == pytest.approx(
        1000.0
    )
    call = runner.call_args
    directory = call.kwargs["cwd"]
    assert isinstance(call.args[0], list)
    assert not call.kwargs.get("shell", False)
    assert call.kwargs["timeout"] is None
    assert (
        "NumberOfProductionCycles     10"
        in (directory / "simulation.input").read_text()
    )
    assert "FrameworkName source_with_spaces" in (directory / "simulation.input").read_text()
    result = json.loads((directory / "results.json").read_text())
    assert result["status"] == "success"
    assert result["input_structure_file"] == params.input_structure_file
    assert result["returncode"] == 0
    assert Path(result["stdout_path"]).is_file()
    assert directory.parent.name == "graspa_runs"
    assert directory.parent.parent.name == "logs"


def test_process_failure_cannot_be_parsed_as_success(simulation):
    params, runner = simulation

    def fail(command, **kwargs):
        kwargs["stdout"].write("UnitCells 0 1 1 1\nOverall: Average: 12.011,\n")
        kwargs["stderr"].write("device failure")
        return subprocess.CompletedProcess(command, 137)

    runner.side_effect = fail
    result = graspa_core.run_graspa_core(params)
    assert result["status"] == "failure"
    assert result["uptake_in_mol_kg"] is None
    assert result["returncode"] == 137
    assert result["temperature_in_K"] == params.temperature
    assert Path(result["stderr_path"]).read_text() == "device failure"
    with pytest.raises(RuntimeError, match="137.*stdout=.*stderr="):
        run_graspa.invoke({"graspa_input": params.model_dump()})


@pytest.mark.parametrize("absolute_root", [False, True])
def test_public_tool_forwards_single_run_controls(simulation, tmp_path, absolute_root):
    params, runner = simulation
    output_root = tmp_path / "custom-runs" if absolute_root else "custom-runs"
    request = {
        **params.model_dump(),
        "output_directory": str(output_root),
        "output_result_file": "custom.log",
        "timeout_seconds": 2.5,
    }
    assert run_graspa.invoke({"graspa_input": request}) == pytest.approx(1000.0)
    expected_root = output_root if absolute_root else tmp_path / "logs" / output_root
    directory = runner.call_args.kwargs["cwd"]
    assert directory.parent == expected_root
    assert runner.call_args.kwargs["timeout"] == 2.5
    result = json.loads((directory / "results.json").read_text())
    assert Path(result["stdout_path"]) == directory / "custom.log"


def test_timeout_and_launch_error_preserve_diagnostics(simulation):
    params, runner = simulation
    params.timeout_seconds = 0.5
    for error in (subprocess.TimeoutExpired("sycl.out", 0.5), OSError("cannot launch")):
        runner.side_effect = error
        result = graspa_core.run_graspa_core(params)
        assert result["status"] == "failure"
        assert result["error_type"] == type(error).__name__
        assert json.loads(Path(result["results_path"]).read_text()) == result
        assert runner.call_args.kwargs["timeout"] == 0.5


def test_missing_executable_is_actionable(simulation, monkeypatch):
    params, runner = simulation
    monkeypatch.setenv("CHEMGRAPH_GRASPA_EXECUTABLE", "/missing/sycl.out")
    result = graspa_core.run_graspa_core(params)
    assert result["status"] == "failure"
    assert "CHEMGRAPH_GRASPA_EXECUTABLE" in result["message"]
    runner.assert_not_called()


def test_repeated_concurrent_and_close_pressure_runs_are_isolated(simulation):
    params, _ = simulation
    jobs = [
        params.model_copy(update=updates)
        for updates in (
            {},
            {},
            {"pressure": 100000.01},
            {"pressure": 100000.02},
            {"n_cycles": 20},
        )
    ]
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(graspa_core.run_graspa_core, jobs))
    assert len({r["run_id"] for r in results}) == 5
    assert all(r["status"] == "success" for r in results)
    assert all(Path(r["results_path"]).is_file() for r in results)


def test_bare_input_and_read_only_source(simulation, monkeypatch):
    params, _ = simulation
    source = Path(params.input_structure_file)
    source.chmod(0o444)
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(source.parent))
    params.input_structure_file = source.name
    original = source.read_bytes()
    result = graspa_core.run_graspa_core(params)
    assert result["status"] == "success"
    assert source.read_bytes() == original


def test_legacy_output_parent_and_operator_environment(
    simulation, tmp_path, monkeypatch
):
    params, runner = simulation
    params.output_result_file = str(tmp_path / "old-output" / "custom.log")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.warns(FutureWarning, match="output_directory"):
        result = graspa_core.run_graspa_core(params)
    assert Path(result["run_dir"]).parent == tmp_path / "old-output"
    assert Path(result["stdout_path"]).name == "custom.log"
    assert runner.call_args.kwargs["env"]["OMP_NUM_THREADS"] == "4"


@pytest.mark.parametrize(
    "updates",
    [
        {"temperature": 0},
        {"temperature": float("nan")},
        {"pressure": -1},
        {"pressure": float("inf")},
        {"n_cycles": 0},
        {"adsorbate": "CO2"},
        {"timeout_seconds": -1},
        {"output_result_file": "simulation.input"},
        {"output_result_file": "framework.cif"},
        {"output_result_file": "H2O.def"},
        {"output_result_file": "results.json"},
        {"output_result_file": "raspa.err"},
        {"output_directory": "runs", "output_result_file": "other/raspa.log"},
    ],
)
def test_invalid_inputs_rejected(updates):
    with pytest.raises(ValueError):
        graspa_input_schema(
            **{"input_structure_file": "test.cif", "adsorbate": "H2O", **updates}
        )


@pytest.mark.parametrize(
    "updates",
    [
        {},
        {"input_structures": "local", "remote_structure_directory": "/remote"},
        {"input_structures": [""]},
        {"input_structures": "local", "conditions": []},
    ],
)
def test_ensemble_requires_unambiguous_nonempty_request(updates):
    with pytest.raises(ValueError):
        graspa_input_schema_ensemble(adsorbate="H2O", **updates)


def test_ensemble_schema_does_not_advertise_deferred_controls():
    properties = graspa_input_schema_ensemble.model_json_schema()["properties"]
    assert not {
        "output_directory", "timeout_seconds", "discovery_timeout_seconds"
    }.intersection(properties)
    single_properties = graspa_input_schema.model_json_schema()["properties"]
    assert {"output_directory", "timeout_seconds"} <= single_properties.keys()


@pytest.mark.parametrize(
    "source",
    [{"input_structures": "local"}, {"remote_structure_directory": "/remote"}],
)
@pytest.mark.parametrize(
    "field, value",
    [
        ("output_directory", "runs"),
        ("output_directory", None),
        ("timeout_seconds", 5),
        ("timeout_seconds", None),
        ("discovery_timeout_seconds", 60),
        ("discovery_timeout_seconds", None),
    ],
)
def test_ensemble_rejects_deferred_controls(source, field, value):
    with pytest.raises(ValueError, match=f"Unsupported ensemble controls: {field}"):
        graspa_input_schema_ensemble(adsorbate="H2O", **source, **{field: value})


@pytest.mark.parametrize(
    "source",
    [
        {"input_structures": "local"},
        {"input_structures": ["one.cif", "two.cif"]},
        {"remote_structure_directory": "/remote"},
    ],
)
def test_ensemble_preserves_supported_requests(source):
    request = {
        **source,
        "adsorbate": "H2O",
        "output_result_file": "legacy/custom.log",
        "n_cycles": 25,
        "conditions": [{"temperature": 300, "pressure": 1000}],
    }
    params = graspa_input_schema_ensemble(**request)
    assert params.model_dump().items() >= request.items()
    assert graspa_input_schema_ensemble.model_validate(params.model_dump()) == params


@pytest.mark.parametrize("token", ["12.011,", "12.011", "1.2011e1;"])
def test_parser_preserves_numeric_tokens(simulation, token):
    params, runner = simulation

    def output(command, **kwargs):
        kwargs["stdout"].write(f"UnitCells 0 1 1 1\nOverall: Average: {token}\n")
        return subprocess.CompletedProcess(command, 0)

    runner.side_effect = output
    assert graspa_core.run_graspa_core(params)["uptake_in_mol_kg"] == pytest.approx(
        1000
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "UnitCells 0 1 1 1\n",
        "UnitCells 0 1 1 1\nOverall: Average:\n",
        "UnitCells 0 1 1 1\nOverall: Average: nan,\n",
        "UnitCells 0 1 1 1\nOverall: Average: inf,\n",
        "UnitCells 0 1 1 1\nOverall: Average: -1,\n",
        "UnitCells 0 0 1 1\nOverall: Average: 1,\n",
        "UnitCells 0 1.5 1 1\nOverall: Average: 1,\n",
    ],
)
def test_malformed_output_is_failure(simulation, text):
    params, runner = simulation

    def output(command, **kwargs):
        kwargs["stdout"].write(text)
        return subprocess.CompletedProcess(command, 0)

    runner.side_effect = output
    result = graspa_core.run_graspa_core(params)
    assert result["status"] == "failure"
    assert result["uptake_in_mol_kg"] is None
    assert result["message"]


def test_invalid_cell_and_mock_data(simulation):
    params, _ = simulation
    with pytest.raises(ValueError, match="nondegenerate"):
        graspa_core._calculate_cell_size(Atoms("C"))
    result = graspa_core.mock_graspa(params)
    assert result == graspa_core.mock_graspa(params)
    assert result["is_mock"] is True


def test_runtime_availability_is_dynamic(monkeypatch):
    tool_registry, agent_registry = ToolRegistry(), AgentRegistry()
    monkeypatch.setenv("CHEMGRAPH_GRASPA_EXECUTABLE", sys.executable)
    assert tool_registry.availability("run_graspa").available
    assert agent_registry.availability("graspa").available
    monkeypatch.setenv("CHEMGRAPH_GRASPA_EXECUTABLE", "/missing/sycl.out")
    assert not tool_registry.availability("run_graspa").available
    assert not agent_registry.availability("graspa").available
    monkeypatch.delenv("CHEMGRAPH_GRASPA_EXECUTABLE")
    monkeypatch.setattr(
        "chemgraph.utils.executables.shutil.which",
        lambda name: sys.executable if name == "sycl.out" else None,
    )
    assert resolve_executable("sycl.out", "CHEMGRAPH_GRASPA_EXECUTABLE") == str(
        Path(sys.executable).resolve()
    )


def test_supported_templates_are_readable_resources():
    directory = files("chemgraph.tools.files.template_graspa_sycl")
    assert all(
        directory.joinpath(name).read_bytes() for name in graspa_core.TEMPLATE_FILES
    )
    assert not directory.joinpath("N2.def").is_file()
    assert not directory.joinpath("CO2.def").is_file()
