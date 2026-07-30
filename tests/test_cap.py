"""Tests for the calc-side wall-clock cap (max_wall_seconds) in run_ase_core.

Hermetic: EMT calculator on a small metal cluster, no network, no LLM, no HPC.
The cap lets a geometry optimization self-terminate at a deadline and return a
resumable partial short of convergence, avoiding an external kill.
"""

import json
import time
from pathlib import Path

import ase.calculators.emt as emt_mod
import numpy as np
import pytest
from ase.cluster import Icosahedron
from ase.io import read, write

from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.tools.ase_core import run_ase_core


@pytest.fixture
def cu_cluster(tmp_path, monkeypatch):
    """A rattled 13-atom Cu cluster that needs many EMT opt steps."""
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    atoms = Icosahedron("Cu", noshells=2)  # 13 atoms
    atoms.rattle(0.2, seed=1)  # perturb so optimization has real work to do
    p = tmp_path / "cu.xyz"
    write(str(p), atoms)
    return p


@pytest.fixture
def slow_emt(monkeypatch):
    """Slow every EMT force evaluation so the wall-clock cap trips deterministically.

    Bare EMT steps are sub-millisecond, so a fixed 50 ms budget races the first
    optimizer step and loses under CPU load (0 steps -> no restart file). Making
    each step ~0.1 s means step *timing* dominates the budget, so a fixed number
    of steps run before the cap regardless of machine load.
    """
    orig = emt_mod.EMT.calculate

    def _slow(self, *args, **kwargs):
        time.sleep(0.1)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(emt_mod.EMT, "calculate", _slow)
    return _slow


def test_cap_stops_early_and_marks_partial(cu_cluster, tmp_path, slow_emt):
    """With a tiny max_wall_seconds the opt caps, marks partial, saves restart."""
    params = ASEInputSchema(
        input_structure_file=str(cu_cluster),
        output_results_file=str(tmp_path / "out.json"),
        driver="opt",
        calculator={"calculator_type": "emt"},
        fmax=1e-6,  # effectively unreachable -> would run many steps
        steps=100000,
        # With slow_emt each step takes ~0.1 s, so a 1.0 s budget lets several
        # optimizer steps run before the cap trips, exercising the real
        # "resumable partial" path (>=1 step, a restart file on disk). The wide
        # budget-to-step ratio keeps a large pre-step-1 stall from starving step
        # 1 on a loaded runner, which would leave the degenerate no-restart case.
        max_wall_seconds=1.0,
    )
    t0 = time.time()
    result = run_ase_core(params)
    elapsed = time.time() - t0

    assert result["status"] == "success"
    assert result["wall_time_capped"] is True
    assert elapsed < 30  # capped, not run to 100000 steps

    data = json.loads((tmp_path / "out.json").read_text())
    assert data["wall_time_capped"] is True
    assert data["converged"] is False
    # A capped run that completed >=1 step advertises a restart file that exists.
    assert data["restart_file"]
    assert Path(data["restart_file"]).exists()


def test_opt_cap_writes_resumable_partial_geometry(cu_cluster, tmp_path, slow_emt):
    """A capped opt writes a standalone, ASE-readable partial geometry to resume from.

    The output schema's ``restart_file`` is the BFGS Hessian JSON (not a
    structure). C1 adds a separate ``resume_input_file`` -- an xyz of the moved
    atoms -- so a resumed opt continues from the partial geometry and skips
    re-reading the original input. This asserts the file exists, reads back as a
    real geometry (right atom count, not the Hessian JSON), and differs from the
    input (a step actually moved the atoms).
    """
    params = ASEInputSchema(
        input_structure_file=str(cu_cluster),
        output_results_file=str(tmp_path / "out.json"),
        driver="opt",
        calculator={"calculator_type": "emt"},
        fmax=1e-6,
        steps=100000,
        max_wall_seconds=1.0,  # slow_emt ~0.1 s/step -> a few steps then cap WITH a restart file
    )
    result = run_ase_core(params)

    assert result["wall_time_capped"] is True
    # The enriched return dict carries the resume contract at the top level.
    resume_file = result["resume_input_file"]
    assert resume_file
    assert resume_file.endswith("_opt.partial.xyz")
    assert Path(resume_file).exists()
    # restart_file is still the Hessian JSON, a distinct artifact.
    assert result["restart_file"].endswith("_opt.restart.json")
    assert resume_file != result["restart_file"]

    # It reads back as a real structure (the full cluster), not the Hessian JSON.
    original = read(str(cu_cluster))
    partial = read(resume_file)
    assert len(partial) == len(original) == 13
    # A step moved the atoms, so the partial geometry is not the input geometry.
    assert not np.allclose(partial.get_positions(), original.get_positions())


def test_no_cap_is_unchanged(cu_cluster, tmp_path):
    """Without max_wall_seconds, behavior is the original path (runs to converge)."""
    params = ASEInputSchema(
        input_structure_file=str(cu_cluster),
        output_results_file=str(tmp_path / "out.json"),
        driver="opt",
        calculator={"calculator_type": "emt"},
        fmax=0.05,
        steps=1000,
        # max_wall_seconds omitted -> None -> uncapped
    )
    result = run_ase_core(params)
    assert result["status"] == "success"
    assert result.get("wall_time_capped", False) is False

    data = json.loads((tmp_path / "out.json").read_text())
    assert data["wall_time_capped"] is False
    assert data["converged"] is True
    assert data["restart_file"] is None
