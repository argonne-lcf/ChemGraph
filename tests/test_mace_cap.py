"""Tests for the wall-clock cap on the MACE/Parsl entry point.

Slice D wires ``max_wall_seconds`` through the MACE path: the MACE-specific
``mace_input_schema`` gains the input field, and ``run_mace_core`` forwards it
into the :class:`ASEInputSchema` it builds before delegating to
:func:`chemgraph.tools.ase_core.run_ase_core`.

Two things are proven here, both hermetically (no network, no MACE weights):

1. WIRING (APPROACH 1 -- spy): ``run_ase_core`` is monkeypatched with a spy that
   captures the ``ASEInputSchema`` it receives and returns a dummy dict. This
   proves the explicit ``max_wall_seconds`` now flows MACE -> ASE (and defaults
   to ``None``) without running a real MACE calculation. Running ``mace_mp`` for
   real would download foundation-model weights and is far too heavy for a unit
   test, so we verify the plumbing at the boundary instead.

2. ENV CAP (core boundary): the allocation env (``CHEMGRAPH_ALLOCATION_*``) is
   read *inside* ``run_ase_core`` at call time, not passed as a schema field, so
   it cannot be observed through the spy. ``run_mace_core`` delegates straight
   into the real ``run_ase_core``, so ``test_env_allocation_would_cap_via_core``
   drives the REAL ``run_ase_core`` (with a cheap EMT calculator standing in for
   the MACE model) under an already-exhausted allocation and asserts the run
   caps -- proving the env path the MACE entry point delegates into.
"""

import time

import ase.calculators.emt as emt_mod
import pytest
from ase.cluster import Icosahedron
from ase.io import write

import chemgraph.tools.parsl_tools as parsl_tools
from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.schemas.mace_parsl_schema import mace_input_schema
from chemgraph.tools.ase_core import run_ase_core
from chemgraph.tools.parsl_tools import run_mace_core


@pytest.fixture
def xyz_path(tmp_path):
    """A trivial structure file so schema construction has a real path.

    The spy tests never read it (run_ase_core is stubbed), but it keeps the
    inputs realistic and mirrors how the other suites stage a structure.
    """
    p = tmp_path / "h2.xyz"
    p.write_text("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
    return p


@pytest.fixture
def ase_spy(monkeypatch):
    """Replace run_ase_core (as seen by parsl_tools) with a capturing spy.

    Returns a one-element list; after ``run_mace_core`` the captured
    ``ASEInputSchema`` is at index 0. Patch the name bound in the
    ``parsl_tools`` module (that is the reference ``run_mace_core`` calls).
    """
    captured = {}

    def _spy(ase_params):
        captured["params"] = ase_params
        return {"status": "success", "message": "spy"}

    monkeypatch.setattr(parsl_tools, "run_ase_core", _spy)
    return captured


# ---------------------------------------------------------------------------
# Wiring: explicit max_wall_seconds flows MACE -> ASE, defaults to None
# ---------------------------------------------------------------------------


def test_explicit_max_wall_seconds_flows_through(xyz_path, ase_spy):
    """An explicit max_wall_seconds on the MACE schema reaches ASEInputSchema."""
    params = mace_input_schema(
        input_structure_file=str(xyz_path),
        driver="opt",
        max_wall_seconds=123.0,
    )
    result = run_mace_core(params)

    assert result["status"] == "success"  # the spy ran, not a real calc
    ase_params = ase_spy["params"]
    assert isinstance(ase_params, ASEInputSchema)
    assert ase_params.max_wall_seconds == 123.0


def test_default_max_wall_seconds_is_none(xyz_path, ase_spy):
    """With no max_wall_seconds set, the MACE path forwards None (uncapped)."""
    params = mace_input_schema(
        input_structure_file=str(xyz_path),
        driver="opt",
    )
    run_mace_core(params)

    ase_params = ase_spy["params"]
    assert ase_params.max_wall_seconds is None


def test_mace_schema_has_max_wall_seconds_field():
    """The input schema exposes max_wall_seconds (default None)."""
    assert "max_wall_seconds" in mace_input_schema.model_fields
    assert mace_input_schema(input_structure_file="x").max_wall_seconds is None


# ---------------------------------------------------------------------------
# Env cap at the boundary run_mace_core delegates into (real run_ase_core)
# ---------------------------------------------------------------------------


@pytest.fixture
def cu_cluster(tmp_path, monkeypatch):
    """A rattled 13-atom Cu cluster that needs many EMT opt steps.

    Copied minimally from tests/test_cap.py / tests/test_allocation_cap.py.
    """
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    atoms = Icosahedron("Cu", noshells=2)  # 13 atoms
    atoms.rattle(0.2, seed=1)
    p = tmp_path / "cu.xyz"
    write(str(p), atoms)
    return p


@pytest.fixture
def slow_emt(monkeypatch):
    """Slow every EMT force evaluation so the wall-clock cap trips deterministically.

    ~0.1 s/step makes step timing dominate a sub-second budget, so a handful of
    steps run before the cap regardless of machine load (mirrors test_cap.py).
    """
    orig = emt_mod.EMT.calculate

    def _slow(self, *args, **kwargs):
        time.sleep(0.1)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(emt_mod.EMT, "calculate", _slow)
    return _slow


def test_env_allocation_would_cap_via_core(cu_cluster, tmp_path, slow_emt, monkeypatch):
    """CHEMGRAPH_ALLOCATION_* caps the REAL run_ase_core that run_mace_core calls.

    The allocation budget is read inside ase_core at call time (not a schema
    field), so it is invisible to the wiring spy above. run_mace_core delegates
    unconditionally into run_ase_core, so exercising the real core under an
    almost-spent allocation proves the env path the MACE entry point relies on.
    A cheap EMT calculator stands in for mace_mp (whose weights we will not
    download); the allocation gate is calculator-agnostic.
    """
    # CHEMGRAPH_ALLOCATION_SECONDS is measured from process start (module
    # import); anchor _PROCESS_START to now so the tiny budget is measured from
    # here, well after an import many seconds in the past.
    import chemgraph.tools.ase_core as ase_core_mod

    monkeypatch.setattr(ase_core_mod, "_PROCESS_START", time.time())
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_SECONDS", "0.35")
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_MARGIN", "0.10")

    params = ASEInputSchema(
        input_structure_file=str(cu_cluster),
        output_results_file=str(tmp_path / "out.json"),
        driver="opt",
        calculator={"calculator_type": "emt"},
        fmax=1e-6,  # effectively unreachable -> would run many steps
        steps=100000,
        # No explicit max_wall_seconds: the cap comes purely from the env.
    )
    t0 = time.time()
    result = run_ase_core(params)
    elapsed = time.time() - t0

    assert result["status"] == "success"
    assert result["wall_time_capped"] is True
    assert elapsed < 30  # capped, not run to 100000 steps


# ---------------------------------------------------------------------------
# Ensemble: max_wall_seconds propagates onto every per-structure job dict
# ---------------------------------------------------------------------------


@pytest.fixture
def structure_dir(tmp_path):
    """A directory of two trivial structures for ensemble expansion."""
    d = tmp_path / "structs"
    d.mkdir()
    (d / "a.xyz").write_text("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
    (d / "b.xyz").write_text("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")
    return d


def test_ensemble_schema_has_max_wall_seconds_field():
    """The ensemble schema exposes max_wall_seconds (default None)."""
    from chemgraph.schemas.mace_parsl_schema import mace_input_schema_ensemble

    assert "max_wall_seconds" in mace_input_schema_ensemble.model_fields
    assert mace_input_schema_ensemble().max_wall_seconds is None


def test_ensemble_hpc_expansion_forwards_max_wall_seconds(structure_dir):
    """_expand_mace_ensemble (HPC) puts max_wall_seconds on every job, round-trips.

    The workers re-hydrate each job dict with ``mace_input_schema(**job)``, whose
    ``_mace_input_to_ase_input`` already forwards ``max_wall_seconds`` into the
    ASEInputSchema (proven by the single-structure wiring tests above). So proving
    the ensemble expansion stamps the field on every per-structure job -- and that
    it survives the schema round-trip the worker performs -- covers the cap
    reaching each ensemble member.
    """
    from chemgraph.mcp.mace_mcp_hpc import _expand_mace_ensemble
    from chemgraph.schemas.mace_parsl_schema import (
        mace_input_schema,
        mace_input_schema_ensemble,
    )

    params = mace_input_schema_ensemble(
        input_structure_directory=str(structure_dir),
        driver="opt",
        max_wall_seconds=0.5,
    )
    jobs = _expand_mace_ensemble(params)

    assert len(jobs) == 2
    assert all(job["max_wall_seconds"] == 0.5 for job in jobs)
    # the exact round-trip a worker performs before delegating to the core.
    assert all(mace_input_schema(**job).max_wall_seconds == 0.5 for job in jobs)


# NOTE: the legacy chemgraph.mcp.mace_mcp_parsl entry point also stamps
# max_wall_seconds onto its per-structure job dict (kept in sync for parity), but
# that module reads PBS_NODEFILE and builds a full Parsl Config at import time, so
# it cannot be imported off a PBS node and has no hermetic test. It is deprecated
# in favor of mace_mcp_hpc (exercised above), which is the current dispatch path.
