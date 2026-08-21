"""Tests for the combined IR explorer component and frame serialization."""

from ase import Atoms

from ui.ir_explorer import build_ir_explorer_html
from ui.visualization import trajectory_to_xyz_frames


def _peak(mode, freq, intensity, frames="3\nFrame 0\nO 0 0 0\nH 0 0 1\nH 0 1 0"):
    return {
        "mode": mode,
        "freq": freq,
        "intensity": intensity,
        "imaginary": False,
        "y": intensity,
        "frames": frames,
    }


def test_explorer_html_embeds_payload_and_libraries():
    html = build_ir_explorer_html(
        curve_x=[500.0, 1595.0, 4000.0],
        curve_y=[0.0, 1.3, 0.0],
        peaks=[_peak(6, 1595.0, 1.3), _peak(7, 3650.0, 0.8)],
        selected_mode=6,
        height=400,
    )

    assert "cdn.jsdelivr.net/npm/3dmol" in html
    assert "cdn.plot.ly/plotly" in html
    assert "__HEIGHT__" not in html and "400px" in html
    assert '"selected_mode": 6' in html
    assert '"mode": 7' in html
    assert "Frame 0" in html  # frames text made it into the payload
    assert "plotly_hover" in html and "plotly_unhover" in html


def test_explorer_html_escapes_closing_tags_in_payload():
    html = build_ir_explorer_html(
        curve_x=[1.0],
        curve_y=[1.0],
        peaks=[_peak(0, 1.0, 1.0, frames="1\n</script>\nH 0 0 0")],
        selected_mode=0,
    )

    # A literal "</script>" inside the JSON would terminate the script tag.
    assert "</script>\\nH" not in html
    assert "<\\/script>" in html


def test_explorer_html_handles_missing_frames_and_selection():
    peak = _peak(3, 100.0, 0.5, frames=None)
    html = build_ir_explorer_html([1.0], [0.0], [peak], selected_mode=None)

    assert '"frames": null' in html
    assert '"selected_mode": null' in html


def test_trajectory_to_xyz_frames_serializes_all_frames():
    frames = [
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7 + 0.01 * i]])
        for i in range(5)
    ]

    text = trajectory_to_xyz_frames(frames)

    blocks = text.split("Frame ")
    assert len(blocks) == 6  # header split: preamble + 5 frames
    assert text.startswith("2\nFrame 0\n")
    assert text.count("\nH ") == 10


def test_trajectory_to_xyz_frames_downsamples_evenly():
    frames = [Atoms("H", positions=[[0, 0, float(i)]]) for i in range(30)]

    text = trajectory_to_xyz_frames(frames, max_frames=10)

    n_frames = text.count("Frame ")
    assert n_frames <= 10
    assert "0.000000" in text  # first frame kept
