"""Regression tests for input_structure_file path resolution in run_ase_core.

Background
----------
Coordinate files are written into the session log directory via
``_resolve_path`` (see ``ase_core._resolve_path``). When a tool writes
``water.xyz`` into ``CHEMGRAPH_LOG_DIR`` but ``run_ase`` is later invoked
with the BARE relative name ``"water.xyz"`` (and cwd is not the log dir),
``run_ase_core`` used to raise ``FileNotFoundError`` because it looked for
the file relative to cwd only.

The fix (ase_core.py ~369-376) resolves a relative ``input_structure_file``
against ``CHEMGRAPH_LOG_DIR`` when the raw path is not already an existing
file, so ``run_ase`` finds the structure the earlier tool wrote.

These tests use the EMT calculator: it needs no model downloads and no
network, and handles water (H, O) fine, so the tests stay hermetic and fast.
"""

import os

import pytest

from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.schemas.calculators.emt_calc import EMTCalc
from chemgraph.tools.ase_core import (
    run_ase_core,
    extract_output_json_core,
    _resolve_existing_path,
)
from chemgraph.tools.ase_tools import file_to_atomsdata
from chemgraph.tools.cheminformatics_core import smiles_to_coordinate_file_core


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """Point CHEMGRAPH_LOG_DIR at a throwaway session dir (auto-restored)."""
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    return tmp_path


def test_run_ase_resolves_bare_name_from_log_dir(log_dir, tmp_path, monkeypatch):
    """A bare relative input_structure_file is found in CHEMGRAPH_LOG_DIR.

    This is the regression: before the fix, calling run_ase with the bare
    name "water.xyz" (written into the log dir) while cwd is elsewhere
    raised FileNotFoundError.
    """
    # Write the structure into the log dir through the real write path
    # (smiles_to_coordinate_file_core -> _resolve_path), using a bare name.
    result = smiles_to_coordinate_file_core("O", output_file="water.xyz")
    assert result["ok"] is True
    written_path = result["path"]
    # It must have landed inside the log dir, not cwd.
    assert os.path.dirname(written_path) == str(log_dir)
    assert os.path.isfile(written_path)

    # Run from a directory that is deliberately NOT the log dir, so the bare
    # name cannot be found relative to cwd.
    other_dir = tmp_path / "elsewhere"
    other_dir.mkdir()
    monkeypatch.chdir(other_dir)
    assert not os.path.isfile("water.xyz")  # bare name not resolvable via cwd

    schema = ASEInputSchema(
        input_structure_file="water.xyz",
        output_results_file="out.json",
        driver="energy",
        calculator=EMTCalc(),
    )
    out = run_ase_core(schema)

    assert out["status"] == "success", out
    assert out["single_point_energy"] is not None
    # The output should have been written into the log dir too.
    assert os.path.isfile(os.path.join(str(log_dir), "out.json"))


def test_run_ase_missing_file_still_fails(log_dir, tmp_path, monkeypatch):
    """A genuinely missing input file still returns a FileNotFoundError dict.

    Guards against the fix masking real missing-file errors: no such file
    exists anywhere (log dir or cwd), so run_ase must report failure.
    """
    monkeypatch.chdir(tmp_path)
    assert not os.path.isfile(os.path.join(str(log_dir), "does_not_exist.xyz"))

    schema = ASEInputSchema(
        input_structure_file="does_not_exist.xyz",
        output_results_file="out.json",
        driver="energy",
        calculator=EMTCalc(),
    )
    out = run_ase_core(schema)

    assert out["status"] == "failure", out
    assert out["error_type"] == "FileNotFoundError", out


def test_resolve_existing_path_prefers_existing_then_log_dir(
    log_dir, tmp_path, monkeypatch
):
    """_resolve_existing_path: cwd file wins, else log-dir file, else raw."""
    # 1. A real cwd-relative file is returned unchanged.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "here.txt").write_text("x")
    assert _resolve_existing_path("here.txt") == "here.txt"

    # 2. A bare name only present in the log dir resolves to the log dir.
    (log_dir / "only_in_log.txt").write_text("y")
    other = tmp_path / "cwd2"
    other.mkdir()
    monkeypatch.chdir(other)
    resolved = _resolve_existing_path("only_in_log.txt")
    assert resolved == os.path.join(str(log_dir), "only_in_log.txt")
    assert os.path.isfile(resolved)

    # 3. A path that exists nowhere is returned unchanged (caller reports it).
    assert _resolve_existing_path("nope.txt") == "nope.txt"


def test_extract_output_json_resolves_bare_name(log_dir, tmp_path, monkeypatch):
    """extract_output_json_core finds a results JSON written into the log dir.

    Regression for the same asymmetry as run_ase: the agent calls
    extract_output_json("out.json") by bare name after run_ase wrote it into
    CHEMGRAPH_LOG_DIR.
    """
    # Produce a result JSON in the log dir via the real run_ase write path.
    cf = smiles_to_coordinate_file_core("O", output_file="water.xyz")
    assert os.path.dirname(cf["path"]) == str(log_dir)
    out = run_ase_core(
        ASEInputSchema(
            input_structure_file="water.xyz",
            output_results_file="results.json",
            driver="energy",
            calculator=EMTCalc(),
        )
    )
    assert out["status"] == "success", out
    assert os.path.isfile(os.path.join(str(log_dir), "results.json"))

    # Read it back by BARE name from a different cwd.
    other = tmp_path / "elsewhere2"
    other.mkdir()
    monkeypatch.chdir(other)
    data = extract_output_json_core("results.json")
    assert isinstance(data, dict) and data, data


def test_file_to_atomsdata_resolves_bare_name(log_dir, tmp_path, monkeypatch):
    """file_to_atomsdata reads a coordinate file written into the log dir."""
    cf = smiles_to_coordinate_file_core("O", output_file="mol.xyz")
    assert os.path.dirname(cf["path"]) == str(log_dir)

    other = tmp_path / "elsewhere3"
    other.mkdir()
    monkeypatch.chdir(other)
    atoms = file_to_atomsdata("mol.xyz")  # bare name; before fix -> FileNotFoundError
    assert atoms is not None
    # water has 3 atoms
    assert len(atoms.numbers) == 3
