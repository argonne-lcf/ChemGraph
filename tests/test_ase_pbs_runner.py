"""Exercise the bundled runner without a scheduler or model downloads."""

import builtins
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from chemgraph.skills.chemgraph.scripts import run_ase_pbs as runner


@pytest.fixture
def allocation(tmp_path, monkeypatch):
    nodefile = tmp_path / "nodes"
    nodefile.write_text("compute01.cluster\ncompute01.cluster\ncompute02\n")
    monkeypatch.setenv("PBS_JOBID", "123.server")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "compute01")
    return nodefile


@pytest.mark.parametrize("host", ["compute01", "COMPUTE01.cluster", "compute02.cluster"])
def test_allocated_hostname_forms(allocation, monkeypatch, host):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: host)
    runner._require_allocation()


@pytest.mark.parametrize(
    "failure",
    ["job", "nodefile", "missing", "unreadable", "empty", "wrong_host", "arguments"],
)
def test_preflight_rejects_before_scientific_imports(
    allocation, monkeypatch, capsys, failure
):
    if failure in {"job", "nodefile"}:
        monkeypatch.delenv("PBS_JOBID" if failure == "job" else "PBS_NODEFILE")
    elif failure == "missing":
        allocation.unlink()
    elif failure == "unreadable":
        monkeypatch.setattr(Path, "read_text", Mock(side_effect=PermissionError("denied")))
    elif failure == "empty":
        allocation.write_text(" \n")
    elif failure == "wrong_host":
        monkeypatch.setattr(runner.socket, "gethostname", lambda: "login01")
    original_import = builtins.__import__

    def guard(name, *args, **kwargs):
        assert not name.startswith(("chemgraph", "ase", "numpy", "torch")), name
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    args = [] if failure == "arguments" else ["--input", "missing.json"]
    assert runner.main(args) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "failure"
    assert summary["error_type"] != "AssertionError", summary


@pytest.fixture
def calculation(allocation, tmp_path, monkeypatch):
    from chemgraph.tools import ase_core

    monkeypatch.chdir(tmp_path)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    path = inputs / "input.json"
    params = {
        "input_structure_file": "water.xyz",
        "output_results_file": "output.json",
        "calculator": {"calculator_type": "EMT"},
        "driver": "opt",
    }
    path.write_text(json.dumps(params))
    output = {
        "input_structure_file": "water.xyz",
        "simulation_input": params,
        "final_structure": {"numbers": [1], "positions": [[0, 0, 0]]},
        "success": True,
        "converged": True,
    }
    artifact = tmp_path / "actual.json"
    artifact.write_text(json.dumps(output))
    result = {
        "status": "success", "results_file": str(artifact),
        "driver": "opt", "potential_energy": 1.5, "energy_unit": "eV",
        "converged": True, "optimization_steps": 3,
    }
    core = Mock(return_value=result)
    monkeypatch.setattr(ase_core, "run_ase_core", core)
    return path, params, artifact, output, result, core


@pytest.mark.parametrize("driver", ["energy", "dipole", "opt", "vib", "thermo", "ir"])
@pytest.mark.parametrize("converged", [False, True])
def test_driver_exit_codes_preserve_metadata(calculation, capsys, driver, converged):
    path, params, artifact, output, result, core = calculation
    params["driver"] = result["driver"] = driver
    output["converged"] = result["converged"] = converged
    path.write_text(json.dumps(params))
    artifact.write_text(json.dumps(output))

    def execute(params):
        print("engine diagnostic")
        return result

    core.side_effect = execute
    expected = 0 if converged or driver in {"energy", "dipole"} else 2
    assert runner.main(["--input", str(path)]) == expected
    captured = capsys.readouterr()
    assert json.loads(captured.out) == result
    assert "engine diagnostic" in captured.err
    core.assert_called_once()


@pytest.mark.parametrize("contents", [None, "{", '{"calculator": {"calculator_type": "EMT"}}'])
def test_invalid_input_never_runs_core(calculation, capsys, contents):
    path, _, _, _, _, core = calculation
    if contents is None:
        path.unlink()
    else:
        path.write_text(contents)
    assert runner.main(["--input", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failure"
    core.assert_not_called()


@pytest.mark.parametrize("failure", ["core", "exception", "path", "missing", "json", "schema", "unsuccessful"])
def test_execution_and_artifact_failures(calculation, capsys, failure):
    path, _, artifact, output, result, core = calculation
    if failure == "core":
        result.update(status="failure", message="engine failed")
        artifact.unlink()
    elif failure == "exception":
        core.side_effect = RuntimeError("engine raised")
    elif failure == "path":
        result.pop("results_file")
    elif failure == "missing":
        artifact.unlink()
    elif failure in {"json", "schema"}:
        artifact.write_text("{" if failure == "json" else "{}")
    else:
        output.update(success=False, error="invalid calculation")
        artifact.write_text(json.dumps(output))
    assert runner.main(["--input", str(path)]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "failure"
    if failure == "core":
        assert summary == result
    core.assert_called_once()


@pytest.mark.parametrize("driver,steps,expected", [("energy", 0, 0), ("opt", 0, 2), ("opt", 100, 0)])
def test_emt_uses_returned_artifact_and_preserves_cwd(
    allocation, tmp_path, monkeypatch, capsys, driver, steps, expected
):
    from ase.build import molecule
    from ase.io import write
    from chemgraph.tools import ase_core

    monkeypatch.chdir(tmp_path)
    logs = tmp_path / "logs"
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(logs))
    monkeypatch.setattr(ase_core, "_ensure_ase_core_file_log", lambda: None)
    write("water.xyz", molecule("H2O"))
    # Neither stale cwd output nor the input JSON's directory may replace core paths.
    (tmp_path / "output.json").write_text("stale result")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    path = inputs / "input.json"
    path.write_text(json.dumps({
        "input_structure_file": "water.xyz", "output_results_file": "output.json",
        "calculator": {"calculator_type": "EMT"}, "driver": driver, "steps": steps,
    }))
    assert runner.main(["--input", str(path)]) == expected
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "success"
    assert summary["results_file"] == str(logs / "output.json")
    assert json.loads((logs / "output.json").read_text())["success"] is True
    assert (tmp_path / "output.json").read_text() == "stale result"
    assert Path.cwd() == tmp_path
