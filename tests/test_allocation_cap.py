"""Tests for the Layer-2 allocation-aware wall-clock cap in run_ase_core.

This is the enforcement side of the auto-continue gate: a calculation running
inside a PBS allocation must self-terminate with a resumable partial *before*
walltime kills it, even when the user set no explicit ``max_wall_seconds``. The
allocation budget is advertised purely through the environment
(``CHEMGRAPH_ALLOCATION_SECONDS`` / ``CHEMGRAPH_ALLOCATION_DEADLINE`` /
``CHEMGRAPH_ALLOCATION_MARGIN``), so a batch script can set it once for the whole
run without any per-call plumbing.

Hermetic: EMT on a small metal cluster, no network/LLM/HPC. The pure-function
tests exercise the env parsing + min() logic directly; the end-to-end tests drive
a real (slowed) optimization and assert the partial/complete signal.
"""

import json
import time
from pathlib import Path

import ase.calculators.emt as emt_mod
import pytest
from ase.cluster import Icosahedron
from ase.io import write

from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.tools.ase_core import (
    _allocation_deadline,
    _allocation_margin,
    _effective_wall_seconds,
    run_ase_core,
)


@pytest.fixture
def cu_cluster(tmp_path, monkeypatch):
    """A rattled 13-atom Cu cluster that needs many EMT opt steps."""
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    atoms = Icosahedron("Cu", noshells=2)  # 13 atoms
    atoms.rattle(0.2, seed=1)
    p = tmp_path / "cu.xyz"
    write(str(p), atoms)
    return p


@pytest.fixture
def slow_emt(monkeypatch):
    """Slow every EMT force evaluation so a wall-clock cap trips deterministically.

    Bare EMT steps are sub-millisecond; ~0.1 s/step makes step *timing* dominate
    a sub-second budget, so a handful of steps run before the cap regardless of
    machine load (mirrors tests/test_cap.py).
    """
    orig = emt_mod.EMT.calculate

    def _slow(self, *args, **kwargs):
        time.sleep(0.1)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(emt_mod.EMT, "calculate", _slow)
    return _slow


# The ambient CHEMGRAPH_ALLOCATION_* env is stripped globally by the autouse
# clear_allocation_env fixture in tests/conftest.py, so every test here starts
# from a clean slate and sets only what it needs via monkeypatch.


# ---------------------------------------------------------------------------
# Pure-function logic: env parsing + tighter-of-two selection
# ---------------------------------------------------------------------------


def test_effective_none_when_nothing_set():
    """No explicit cap and no allocation env -> uncapped (None)."""
    assert _effective_wall_seconds(None, time.time()) is None


def test_effective_uses_explicit_when_no_allocation():
    """With only an explicit cap, the effective value is that cap."""
    assert _effective_wall_seconds(42.0, time.time()) == 42.0


def test_effective_uses_allocation_when_no_explicit(monkeypatch):
    """With only an allocation budget, the effective value is remaining-margin."""
    import chemgraph.tools.ase_core as ase_core_mod

    # _PROCESS_START is captured at import, arbitrarily far in the past during a
    # full-suite run; anchor it to a known point so remaining is exact and the
    # bound below can't flake on a slow session.
    now = time.time()
    monkeypatch.setattr(ase_core_mod, "_PROCESS_START", now)
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_SECONDS", "1000")
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_MARGIN", "60")
    # remaining = (now + 1000) - 60 - start, and start >= now by a hair, so eff
    # is 940 minus a tiny elapsed slack.
    eff = _effective_wall_seconds(None, time.time())
    assert eff is not None
    assert 939.0 < eff <= 940.0


def test_effective_takes_the_tighter_bound(monkeypatch):
    """min(explicit, allocation-remaining): whichever is smaller wins."""
    import chemgraph.tools.ase_core as ase_core_mod

    now = time.time()
    monkeypatch.setattr(ase_core_mod, "_PROCESS_START", now)
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_SECONDS", "1000")
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_MARGIN", "60")
    # Explicit 100 is tighter than ~940 remaining -> explicit wins.
    assert _effective_wall_seconds(100.0, time.time()) == 100.0
    # Explicit 5000 is looser than ~940 remaining -> allocation wins.
    eff = _effective_wall_seconds(5000.0, time.time())
    assert eff is not None and eff < 5000.0


def test_deadline_takes_precedence_over_seconds(monkeypatch):
    """CHEMGRAPH_ALLOCATION_DEADLINE (absolute) wins over _SECONDS when both set."""
    target = time.time() + 500.0
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_DEADLINE", str(target))
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_SECONDS", "99999")
    assert _allocation_deadline() == pytest.approx(target)


def test_stale_past_deadline_is_ignored(monkeypatch):
    """A deadline BEFORE this process started is a leftover -> ignored, not honored.

    A CHEMGRAPH_ALLOCATION_DEADLINE from a prior allocation still in the
    environment would (if honored) clamp every calc to 0.001 s and cap
    immediately, forever. _allocation_deadline must drop it (return None), so
    with no other budget the run is uncapped.
    """
    import chemgraph.tools.ase_core as ase_core_mod

    now = time.time()
    # deadline is 100 s before this process began -> stale, cannot be ours.
    monkeypatch.setattr(ase_core_mod, "_PROCESS_START", now)
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_DEADLINE", str(now - 100.0))
    assert _allocation_deadline() is None
    # and with nothing else set, the effective cap is None (uncapped).
    assert _effective_wall_seconds(None, now) is None


def test_stale_deadline_falls_through_to_seconds(monkeypatch):
    """A stale deadline is ignored but CHEMGRAPH_ALLOCATION_SECONDS still applies.

    Dropping the leftover deadline must fall through to the _SECONDS budget rather
    than disabling the allocation cap entirely.
    """
    import chemgraph.tools.ase_core as ase_core_mod

    now = time.time()
    monkeypatch.setattr(ase_core_mod, "_PROCESS_START", now)
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_DEADLINE", str(now - 100.0))
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_SECONDS", "1000")
    # deadline ignored -> falls through to _PROCESS_START + 1000.
    assert _allocation_deadline() == pytest.approx(now + 1000.0)


def test_genuinely_spent_allocation_still_clamps(monkeypatch):
    """A deadline AFTER process start but now past is honored and clamps to tiny.

    This is the genuinely-spent-mid-run case (distinct from a stale leftover): the
    deadline belongs to this allocation, we've simply run out of time, so the cap
    must still fire immediately (0.001) and never be ignored.
    """
    import chemgraph.tools.ase_core as ase_core_mod

    now = time.time()
    monkeypatch.setattr(ase_core_mod, "_PROCESS_START", now)
    # deadline 10 s after start (ours), but we check from 100 s later (spent).
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_DEADLINE", str(now + 10.0))
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_MARGIN", "0")
    assert _allocation_deadline() == pytest.approx(now + 10.0)
    eff = _effective_wall_seconds(None, now + 100.0)
    assert eff == 0.001


def test_exhausted_allocation_clamps_to_tiny_positive(monkeypatch):
    """A deadline spent mid-run yields a tiny positive, not <= 0.

    A non-positive effective cap would be falsy at the gate and silently disable
    the cap -- the opposite of intended when we are out of time. It must stay a
    small positive so the run caps as early as cleanly possible. The deadline is
    anchored AFTER _PROCESS_START (genuinely spent this run, not a stale leftover
    that _allocation_deadline now ignores -- see test_stale_past_deadline).
    """
    import chemgraph.tools.ase_core as ase_core_mod

    now = time.time()
    monkeypatch.setattr(ase_core_mod, "_PROCESS_START", now - 200.0)
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_DEADLINE", str(now - 100.0))
    eff = _effective_wall_seconds(None, now)
    assert eff is not None and eff > 0.0


def test_margin_default_and_override(monkeypatch):
    """Margin defaults to 60 s and honors CHEMGRAPH_ALLOCATION_MARGIN."""
    assert _allocation_margin() == 60.0
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_MARGIN", "120")
    assert _allocation_margin() == 120.0


def test_nonnumeric_env_is_ignored(monkeypatch):
    """Garbage env values are ignored and never raise."""
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_DEADLINE", "soon")
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_SECONDS", "later")
    assert _allocation_deadline() is None
    monkeypatch.setenv("CHEMGRAPH_ALLOCATION_MARGIN", "lots")
    assert _allocation_margin() == 60.0


# ---------------------------------------------------------------------------
# End-to-end: allocation env alone caps run_ase_core (no explicit cap)
# ---------------------------------------------------------------------------


def test_allocation_seconds_caps_without_explicit(
    cu_cluster, tmp_path, slow_emt, monkeypatch
):
    """CHEMGRAPH_ALLOCATION_SECONDS alone caps the opt with no max_wall_seconds.

    This is the core Layer-2 behavior: a run with no explicit cap still self-
    terminates with a resumable partial because it is inside a (nearly spent)
    allocation.
    """
    # CHEMGRAPH_ALLOCATION_SECONDS is measured from process start (module
    # import), so that one budget counts down across every calc in the run. In a
    # long test session import time is many seconds in the past; anchor it to now
    # so this test's budget is measured from here, not from import.
    import chemgraph.tools.ase_core as ase_core_mod

    monkeypatch.setattr(ase_core_mod, "_PROCESS_START", time.time())
    # ~1.5 s of usable budget: total 2.0 s minus a 0.5 s margin. The budget is
    # anchored before run_ase_core does its own setup (schema build, calculator
    # load, read(atoms)), so it needs headroom above one 0.1 s step; a tighter
    # budget lets setup latency starve step 1 under load and cap before any
    # restart file is written.
    _set_allocation_env(monkeypatch, seconds="2.0", margin="0.5")

    params = ASEInputSchema(
        input_structure_file=str(cu_cluster),
        output_results_file=str(tmp_path / "out.json"),
        driver="opt",
        calculator={"calculator_type": "emt"},
        fmax=1e-6,
        steps=100000,
        # NOTE: no max_wall_seconds -- the cap comes purely from the allocation.
    )
    t0 = time.time()
    result = run_ase_core(params)
    elapsed = time.time() - t0

    assert result["status"] == "success"
    assert result["wall_time_capped"] is True
    assert elapsed < 30

    data = json.loads((tmp_path / "out.json").read_text())
    assert data["wall_time_capped"] is True
    assert data["converged"] is False
    assert data["restart_file"]
    assert Path(data["restart_file"]).exists()


def test_allocation_deadline_caps_without_explicit(
    cu_cluster, tmp_path, slow_emt, monkeypatch
):
    """CHEMGRAPH_ALLOCATION_DEADLINE (absolute epoch) alone caps the opt."""
    # ~1.5 s of usable budget (2.0 s out minus a 0.5 s margin) so setup latency
    # cannot starve step 1 before a restart file is written; see the seconds test.
    deadline = time.time() + 2.0
    _set_allocation_env(monkeypatch, deadline=str(deadline), margin="0.5")

    params = ASEInputSchema(
        input_structure_file=str(cu_cluster),
        output_results_file=str(tmp_path / "out.json"),
        driver="opt",
        calculator={"calculator_type": "emt"},
        fmax=1e-6,
        steps=100000,
    )
    result = run_ase_core(params)
    assert result["wall_time_capped"] is True
    data = json.loads((tmp_path / "out.json").read_text())
    assert data["converged"] is False
    assert data["restart_file"]


def test_generous_allocation_lets_short_run_complete(
    cu_cluster, tmp_path, monkeypatch
):
    """A generous allocation does NOT cap a short optimization (auto-complete).

    No slow_emt here: the opt converges in well under the budget, so the run
    finishes normally and is not marked partial -- the "short calcs auto-complete
    within one allocation" half of the design.
    """
    _set_allocation_env(monkeypatch, seconds="3600", margin="60")

    params = ASEInputSchema(
        input_structure_file=str(cu_cluster),
        output_results_file=str(tmp_path / "out.json"),
        driver="opt",
        calculator={"calculator_type": "emt"},
        fmax=0.05,
        steps=1000,
    )
    result = run_ase_core(params)
    assert result["status"] == "success"
    assert result.get("wall_time_capped", False) is False

    data = json.loads((tmp_path / "out.json").read_text())
    assert data["wall_time_capped"] is False
    assert data["converged"] is True
    assert data["restart_file"] is None


def test_explicit_cap_still_wins_when_tighter(
    cu_cluster, tmp_path, slow_emt, monkeypatch
):
    """A tight explicit max_wall_seconds caps even under a generous allocation."""
    _set_allocation_env(monkeypatch, seconds="3600", margin="60")

    params = ASEInputSchema(
        input_structure_file=str(cu_cluster),
        output_results_file=str(tmp_path / "out.json"),
        driver="opt",
        calculator={"calculator_type": "emt"},
        fmax=1e-6,
        steps=100000,
        max_wall_seconds=0.25,  # far tighter than the 1 h allocation
    )
    result = run_ase_core(params)
    assert result["wall_time_capped"] is True
    data = json.loads((tmp_path / "out.json").read_text())
    assert data["converged"] is False


def test_exhausted_allocation_caps_with_no_restart_and_honest_message(
    cu_cluster, tmp_path, slow_emt, monkeypatch
):
    """An already-spent allocation caps before step 1: no restart, honest message.

    Guards the boundary the feature exists to protect. With the deadline already
    in the past, the effective cap clamps to a tiny positive and the very first
    deadline check (before any optimizer step completes) trips, so no restart
    file is written. The result must still be coherent: wall_time_capped=True but
    restart_file=None, and the returned message must NOT claim a restart was
    saved.
    """
    import chemgraph.tools.ase_core as ase_core_mod

    # Anchor _PROCESS_START in the past so the deadline is a genuinely-spent
    # mid-run deadline (still clamps to cap immediately), NOT a stale leftover
    # that _allocation_deadline ignores (see test_stale_past_deadline).
    now = time.time()
    monkeypatch.setattr(ase_core_mod, "_PROCESS_START", now - 200.0)
    _set_allocation_env(monkeypatch, deadline=str(now - 100.0), margin="0")

    params = ASEInputSchema(
        input_structure_file=str(cu_cluster),
        output_results_file=str(tmp_path / "out.json"),
        driver="opt",
        calculator={"calculator_type": "emt"},
        fmax=1e-6,
        steps=100000,
    )
    result = run_ase_core(params)

    assert result["status"] == "success"
    assert result["wall_time_capped"] is True
    # No step completed -> no restart file, and the message must be honest.
    assert "restart" not in result["message"].lower() or "no restart" in (
        result["message"].lower()
    )

    data = json.loads((tmp_path / "out.json").read_text())
    assert data["wall_time_capped"] is True
    assert data["converged"] is False
    assert data["restart_file"] is None


def test_margin_larger_than_remaining_caps_immediately(
    cu_cluster, tmp_path, slow_emt, monkeypatch
):
    """A margin exceeding the remaining budget behaves like an exhausted allocation."""
    import chemgraph.tools.ase_core as ase_core_mod

    monkeypatch.setattr(ase_core_mod, "_PROCESS_START", time.time())
    # 10 s budget but a 1000 s margin -> remaining is deeply negative -> clamp.
    _set_allocation_env(monkeypatch, seconds="10", margin="1000")

    params = ASEInputSchema(
        input_structure_file=str(cu_cluster),
        output_results_file=str(tmp_path / "out.json"),
        driver="opt",
        calculator={"calculator_type": "emt"},
        fmax=1e-6,
        steps=100000,
    )
    result = run_ase_core(params)
    assert result["wall_time_capped"] is True
    data = json.loads((tmp_path / "out.json").read_text())
    assert data["converged"] is False


def test_allocation_env_caps_vib_driver(cu_cluster, tmp_path, slow_emt, monkeypatch):
    """The allocation env path also caps a vib displacement sweep (not just opt).

    Exercises the vib deadline site, which the explicit-cap tests in
    test_cap_vib.py cover only via max_wall_seconds, never via the allocation env.
    """
    import chemgraph.tools.ase_core as ase_core_mod

    monkeypatch.setattr(ase_core_mod, "_PROCESS_START", time.time())
    # ~1.5 s of usable budget (2.0 s minus a 0.5 s margin); the 6N+1 displacement
    # sweep still cannot finish in it, but setup latency cannot starve the run.
    _set_allocation_env(monkeypatch, seconds="2.0", margin="0.5")

    params = ASEInputSchema(
        input_structure_file=str(cu_cluster),
        output_results_file=str(tmp_path / "out.json"),
        driver="vib",
        calculator={"calculator_type": "emt"},
        # A converged geometry is assumed; the point is that the 6N+1 displacement
        # sweep on a 13-atom cluster (40 evals @ ~0.1 s) cannot finish in ~1.5 s.
    )
    result = run_ase_core(params)
    assert result["status"] == "success"
    assert result["wall_time_capped"] is True

    data = json.loads((tmp_path / "out.json").read_text())
    assert data["wall_time_capped"] is True
    # vib caps point restart_file at the persistent displacement cache dir.
    assert data["restart_file"]
    assert Path(data["restart_file"]).exists()


def _set_allocation_env(monkeypatch, *, seconds=None, deadline=None, margin=None):
    """Set allocation env vars for the current test via monkeypatch.

    A plain helper (not a fixture) so each end-to-end test states its own budget
    inline; monkeypatch reverts them at teardown.
    """
    if seconds is not None:
        monkeypatch.setenv("CHEMGRAPH_ALLOCATION_SECONDS", seconds)
    if deadline is not None:
        monkeypatch.setenv("CHEMGRAPH_ALLOCATION_DEADLINE", deadline)
    if margin is not None:
        monkeypatch.setenv("CHEMGRAPH_ALLOCATION_MARGIN", margin)
