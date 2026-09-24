"""Tests for the linked optimization-step explorer."""

import json
import re

import pytest

from ui import opt_explorer


@pytest.fixture()
def traj_path(tmp_path):
    """Write a short EMT optimization trajectory of a Cu dimer."""
    from ase import Atoms
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS

    atoms = Atoms("Cu2", positions=[[0, 0, 0], [0, 0, 2.9]])
    atoms.calc = EMT()
    path = tmp_path / "cu2_opt.traj"
    opt = BFGS(atoms, logfile=None, trajectory=str(path))
    opt.run(fmax=0.01, steps=30)
    return str(path)


def _payload(html: str) -> dict:
    match = re.search(r"const DATA = (\{.*?\});\nconst ACCENT", html, re.S)
    assert match, "payload not embedded"
    return json.loads(match.group(1).replace("<\\/", "</"))


def test_read_optimization_steps_aligns_frames_with_energies(traj_path):
    result = opt_explorer.read_optimization_steps(traj_path)
    assert result is not None
    steps, frames = result
    assert len(steps) >= 2
    assert [s["step"] for s in steps] == list(range(len(steps)))
    assert all(isinstance(s["energy"], float) for s in steps)
    assert all(s["fmax"] is None or s["fmax"] >= 0 for s in steps)
    # One XYZ block ("2\nStep i\n<2 atoms>") per step.
    assert frames.count("Step ") == len(steps)
    assert frames.splitlines()[1] == "Step 0"


def test_read_optimization_steps_downsamples_but_keeps_endpoints(traj_path):
    full = opt_explorer.read_optimization_steps(traj_path)
    assert full is not None
    n_total = len(full[0])
    assert n_total > 3
    steps, frames = opt_explorer.read_optimization_steps(traj_path, max_frames=3)
    assert len(steps) <= 4
    assert steps[0]["step"] == 0
    assert steps[-1]["step"] == n_total - 1
    # Frames stay aligned with the kept steps.
    assert frames.count("Step ") == len(steps)
    assert f"Step {n_total - 1}" in frames


def test_read_optimization_steps_missing_file_returns_none(tmp_path):
    assert opt_explorer.read_optimization_steps(str(tmp_path / "none.traj")) is None


def test_build_html_embeds_payload_and_defaults_to_last_step():
    steps = [
        {"step": 0, "energy": -1.0, "fmax": 0.5},
        {"step": 1, "energy": -1.5, "fmax": 0.05},
    ]
    frames = "2\nStep 0\nCu 0 0 0\nCu 0 0 2.9\n2\nStep 1\nCu 0 0 0\nCu 0 0 2.5"
    html = opt_explorer.build_opt_explorer_html(steps, frames)
    data = _payload(html)
    assert data["steps"] == steps
    assert data["frames"] == frames
    assert data["selected_index"] == 1
    assert data["interval_ms"] == 250
    assert data["autoplay"] is False
    # Both CDN scripts and the controls are present.
    assert opt_explorer._3DMOL_JS in html and opt_explorer._PLOTLY_JS in html
    assert 'id="playbtn"' in html and 'id="slider"' in html


def test_build_html_escapes_script_terminators():
    steps = [{"step": 0, "energy": 0.0, "fmax": None}]
    frames = "1\n</script><script>alert(1)</script>\nH 0 0 0"
    html = opt_explorer.build_opt_explorer_html(steps, frames, selected_index=0)
    # The raw terminator must not appear inside the payload literal.
    payload_start = html.index("const DATA = ")
    payload_end = html.index("const ACCENT")
    assert "</script>" not in html[payload_start:payload_end]
    assert _payload(html)["frames"] == frames


def test_build_html_clamps_selected_index():
    steps = [{"step": 0, "energy": 0.0, "fmax": None}]
    html = opt_explorer.build_opt_explorer_html(
        steps, "1\nStep 0\nH 0 0 0", selected_index=-5, interval_ms=100, autoplay=True
    )
    data = _payload(html)
    assert data["selected_index"] == 0
    assert data["interval_ms"] == 100
    assert data["autoplay"] is True
