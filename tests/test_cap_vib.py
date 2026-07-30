"""Tests for the wall-clock cap on vibrational analysis (vib/thermo/ir).

Hermetic: EMT calculator on a water molecule, no network, no LLM, no HPC.

Unlike a geometry optimization -- where a capped run still yields a usable
partial geometry -- a capped vibrational sweep cannot produce frequencies until
every one of its 6N+1 displacements is done (ASE's ``get_energies`` needs the
full cache). So a capped vib/thermo/ir run hands off: it keeps a persistent
displacement cache on disk, marks ``wall_time_capped``, and a re-run of the same
driver with more budget resumes from that cache and skips recomputing.

EMT force evaluations on water are sub-millisecond, so a pure wall-clock budget
cannot reliably trip the displacement sweep. These tests slow ``EMT.calculate``
with a small sleep to make the cap deterministic -- the calc is built internally
by ``run_ase_core`` from ``calculator_type`` and bound to ``atoms.calc``, which
is exactly what ASE ``Vibrations`` reads.
"""

import glob
import json
import time
from pathlib import Path

import ase.calculators.emt as emt_mod
import pytest
from ase.build import molecule
from ase.io import write

from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.tools.ase_core import run_ase_core


@pytest.fixture
def water(tmp_path, monkeypatch):
    """A water molecule written to the sandboxed log dir."""
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    p = tmp_path / "water.xyz"
    write(str(p), molecule("H2O"))
    return p


@pytest.fixture
def slow_emt(monkeypatch):
    """Slow every EMT force evaluation so a wall-clock cap trips deterministically."""
    orig = emt_mod.EMT.calculate

    def _slow(self, *args, **kwargs):
        time.sleep(0.03)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(emt_mod.EMT, "calculate", _slow)
    return _slow


def _vib_params(water, tmp_path, **extra):
    # fmax huge -> the pre-analysis optimization converges immediately, so the
    # wall-clock budget is spent on the displacement sweep (what we're testing),
    # not on the optimization.
    return ASEInputSchema(
        input_structure_file=str(water),
        output_results_file=str(tmp_path / "out.json"),
        driver="vib",
        calculator={"calculator_type": "emt"},
        fmax=1e3,
        steps=200,
        **extra,
    )


def test_vib_uncapped_is_unchanged(water, tmp_path):
    """Without max_wall_seconds, vib runs to completion and returns frequencies."""
    result = run_ase_core(_vib_params(water, tmp_path))
    assert result["status"] == "success"
    assert result.get("wall_time_capped", False) is False

    data = json.loads((tmp_path / "out.json").read_text())
    assert data["wall_time_capped"] is False
    assert data["restart_file"] is None
    # H2O has 3 atoms -> 3N = 9 vibrational modes.
    assert len(data["vibrational_frequencies"]["energies"]) == 9
    # The uncapped path uses an ephemeral cache dir -> nothing left behind.
    assert not glob.glob(str(tmp_path / "*_vibcache"))


def test_vib_sweep_caps_and_keeps_cache(water, tmp_path, slow_emt):
    """A tiny budget caps the displacement sweep, marks partial, keeps the cache."""
    t0 = time.time()
    result = run_ase_core(
        _vib_params(water, tmp_path, max_wall_seconds=0.25)
    )
    elapsed = time.time() - t0

    assert result["status"] == "success"
    assert result["wall_time_capped"] is True
    assert "CAPPED" in result["message"]
    assert elapsed < 30  # capped, not the full 19-displacement sweep at 0.03s+

    data = json.loads((tmp_path / "out.json").read_text())
    assert data["wall_time_capped"] is True
    # No frequencies on a capped sweep -- it hands off instead.
    assert not data["vibrational_frequencies"].get("energies")
    # restart_file points at a persistent cache dir holding partial work.
    cache_dir = data["restart_file"]
    assert cache_dir and Path(cache_dir).is_dir()
    partial = glob.glob(str(Path(cache_dir) / "vib" / "cache.*.json"))
    assert 0 < len(partial) < 19  # some done, but not the full 6N+1


def test_vib_resume_finishes_from_cache(water, tmp_path, slow_emt):
    """Re-running the capped driver with more budget resumes and completes."""
    # First run caps partway through the sweep.
    r1 = run_ase_core(_vib_params(water, tmp_path, max_wall_seconds=0.25))
    assert r1["wall_time_capped"] is True
    cache_dir = json.loads((tmp_path / "out.json").read_text())["restart_file"]
    n_after_first = len(glob.glob(str(Path(cache_dir) / "vib" / "cache.*.json")))
    assert 0 < n_after_first < 19

    # Second run: same driver + structure, generous budget -> skips cached
    # displacements and finishes.
    r2 = run_ase_core(_vib_params(water, tmp_path, max_wall_seconds=120))
    assert r2["status"] == "success"
    assert r2.get("wall_time_capped", False) is False

    data = json.loads((tmp_path / "out.json").read_text())
    assert data["wall_time_capped"] is False
    assert len(data["vibrational_frequencies"]["energies"]) == 9


def test_capped_optimization_skips_vib(water, tmp_path):
    """If the pre-analysis opt itself caps, vib is skipped (needs a min geometry)."""
    params = ASEInputSchema(
        input_structure_file=str(water),
        output_results_file=str(tmp_path / "out.json"),
        driver="vib",
        calculator={"calculator_type": "emt"},
        fmax=1e-6,  # unreachable -> opt would run many steps
        steps=100000,
        max_wall_seconds=1e-9,  # caps the opt before it converges
    )
    result = run_ase_core(params)
    assert result["status"] == "success"
    assert result["wall_time_capped"] is True
    assert "optimization" in result["message"].lower()

    data = json.loads((tmp_path / "out.json").read_text())
    assert data["wall_time_capped"] is True
    assert data["converged"] is False
    # No vibrational analysis was attempted on the half-optimized geometry.
    assert not data["vibrational_frequencies"].get("energies")
