"""Combined 3D-viewer + convergence-plot component for optimization runs.

The convergence chart and the trajectory animation used to be independent
widgets: the chart showed energies per step while the viewer looped through
the frames on its own, so a step in the plot could not be matched to a
geometry.  Streamlit cannot forward hover/click events from a Plotly chart
to Python without a full rerun, so both panels live in one HTML component
wired together in JavaScript:

* hovering or clicking a step marker shows that step's geometry;
* a Play/Pause button animates through every step while the plot marker
  tracks the current frame;
* a slider scrubs through the steps manually.

``read_optimization_steps`` and ``build_opt_explorer_html`` are pure
(unit-testable); ``render_opt_explorer`` is the thin Streamlit wrapper.
"""

from __future__ import annotations

import json
from typing import Optional

import streamlit as st

# Same CDN builds the IR explorer relies on.
_3DMOL_JS = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"
_PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"

#: Upper bound on embedded frames; longer trajectories are downsampled
#: evenly (the plot still labels the true step index of every kept frame).
DEFAULT_MAX_FRAMES = 300

_TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<script src="__3DMOL_JS__"></script>
<script src="__PLOTLY_JS__"></script>
<style>
  html, body { margin: 0; padding: 0; height: 100%;
               font-family: "Source Sans Pro", sans-serif; }
  #wrap { display: flex; gap: 10px; height: __HEIGHT__px; }
  #left { flex: 0 0 42%; display: flex; flex-direction: column; gap: 6px;
          min-width: 0; }
  #viewerbox { flex: 1; position: relative;
               border: 1px solid rgba(128,128,128,.35);
               border-radius: 8px; overflow: hidden; background: white; }
  #viewer { width: 100%; height: 100%; position: relative; }
  #steplabel { position: absolute; left: 8px; top: 6px; z-index: 5;
               font-size: 13px; color: #444;
               background: rgba(255,255,255,.85);
               padding: 2px 8px; border-radius: 6px; }
  #controls { display: flex; align-items: center; gap: 8px;
              font-size: 13px; color: #444; }
  #playbtn { min-width: 74px; padding: 4px 10px; border-radius: 6px;
             border: 1px solid #0E9594; background: white; color: #0E9594;
             cursor: pointer; font-size: 13px; }
  #playbtn:hover { background: rgba(14,149,148,.1); }
  #slider { flex: 1; accent-color: #0E9594; }
  #chart { flex: 1; min-width: 0; }
</style></head>
<body>
<div id="wrap">
  <div id="left">
    <div id="viewerbox"><div id="viewer"></div><div id="steplabel"></div></div>
    <div id="controls">
      <button id="playbtn" type="button" aria-label="Play">&#9654; Play</button>
      <input id="slider" type="range" min="0" max="0" value="0" step="1"
             aria-label="Optimization step"/>
      <span id="counter"></span>
    </div>
  </div>
  <div id="chart"></div>
</div>
<script>
const DATA = __PAYLOAD__;
const ACCENT = "#0E9594";
const AXIS_INK = "#8a8f98";
const steps = DATA.steps;
const N = steps.length;

// ---------------- 3D viewer ----------------
// WebGL can be unavailable (remote desktops, headless browsers); the
// chart must keep working, so the viewer fails soft.
let viewer = null;
const label = document.getElementById("steplabel");
try {
  viewer = $3Dmol.createViewer("viewer", {backgroundColor: "white"});
  viewer.addModelsAsFrames(DATA.frames, "xyz");
  viewer.setStyle({}, {stick: {radius: 0.12}, sphere: {scale: 0.25}});
  viewer.zoomTo();
  viewer.render();
} catch (e) {
  viewer = null;
  label.textContent = "3D viewer unavailable (WebGL required)";
}

function fmtEnergy(v) { return v === null ? "n/a" : v.toFixed(4) + " eV"; }
function fmtForce(v) { return v === null ? "n/a" : v.toFixed(4) + " eV/\\u00c5"; }

// ---------------- chart ----------------
const haveForces = steps.some(s => s.fmax !== null);
const xs = steps.map(s => s.step);
const traces = [
  {x: xs, y: steps.map(s => s.energy), mode: "lines+markers", name: "Energy",
   line: {color: ACCENT, width: 2},
   marker: {size: xs.map(() => 7), color: "rgba(14,149,148,.35)",
            line: {color: ACCENT, width: xs.map(() => 1.2)}},
   hovertemplate: "step %{x}<br>%{y:.6f} eV<extra></extra>",
   xaxis: "x", yaxis: "y"}
];
if (haveForces) {
  traces.push(
    {x: xs, y: steps.map(s => s.fmax), mode: "lines+markers", name: "Max force",
     line: {color: ACCENT, width: 2},
     marker: {size: xs.map(() => 7), color: "rgba(14,149,148,.35)",
              line: {color: ACCENT, width: xs.map(() => 1.2)}},
     hovertemplate: "step %{x}<br>%{y:.4f} eV/\\u00c5<extra></extra>",
     xaxis: "x", yaxis: "y2"});
}
const grid = {gridcolor: "rgba(128,128,128,.25)", zeroline: false, color: AXIS_INK};
const layout = {
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  showlegend: false, hovermode: "closest",
  margin: {l: 60, r: 10, t: 10, b: 40},
  xaxis: Object.assign({title: {text: "Step"}, anchor: haveForces ? "y2" : "y"}, grid),
  yaxis: Object.assign({title: {text: "Energy (eV)"},
                        domain: haveForces ? [0.56, 1] : [0, 1]}, grid),
  shapes: []
};
if (haveForces) {
  layout.yaxis2 = Object.assign({title: {text: "Max force (eV/\\u00c5)"}, type: "log",
                                 dtick: 1, domain: [0, 0.44]}, grid);
}
const chart = document.getElementById("chart");
Plotly.newPlot(chart, traces, layout, {responsive: true, displayModeBar: false});

// ---------------- linking ----------------
let current = -1;
let playing = false;
let timer = null;
const slider = document.getElementById("slider");
const counter = document.getElementById("counter");
const playbtn = document.getElementById("playbtn");
slider.max = String(Math.max(N - 1, 0));

function showStep(idx, fromSlider) {
  if (idx < 0 || idx >= N) return;
  const s = steps[idx];
  current = idx;
  label.textContent = "Step " + s.step + " \\u00b7 E = " + fmtEnergy(s.energy) +
      (s.fmax === null ? "" : " \\u00b7 F\\u2098\\u2090\\u2093 = " + fmtForce(s.fmax));
  counter.textContent = (idx + 1) + " / " + N;
  if (!fromSlider) slider.value = String(idx);
  if (viewer) {
    try { viewer.setFrame(idx); viewer.render(); } catch (e) { /* keep chart alive */ }
  }
  const sizes = xs.map((_, i) => i === idx ? 13 : 7);
  const widths = xs.map((_, i) => i === idx ? 2.5 : 1.2);
  const indices = haveForces ? [0, 1] : [0];
  Plotly.restyle(chart, {"marker.size": indices.map(() => sizes),
                         "marker.line.width": indices.map(() => widths)}, indices);
  Plotly.relayout(chart, {shapes: [{type: "line", x0: s.step, x1: s.step,
                                    y0: 0, y1: 1, yref: "paper",
                                    line: {color: ACCENT, width: 1, dash: "dot"}}]});
}

function setPlaying(on) {
  playing = on && N > 1;
  playbtn.innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Play";
  playbtn.setAttribute("aria-label", playing ? "Pause" : "Play");
  if (timer) { clearInterval(timer); timer = null; }
  if (playing) {
    timer = setInterval(() => showStep((current + 1) % N, false), DATA.interval_ms);
  }
}

playbtn.addEventListener("click", () => setPlaying(!playing));
slider.addEventListener("input", () => { setPlaying(false); showStep(parseInt(slider.value, 10), true); });
chart.on("plotly_hover", ev => {
  if (playing) return;  // do not fight the animation
  const pt = ev.points && ev.points[0];
  if (pt) showStep(pt.pointIndex, false);
});
chart.on("plotly_click", ev => {
  const pt = ev.points && ev.points[0];
  if (!pt) return;
  setPlaying(false);
  showStep(pt.pointIndex, false);
});

showStep(Math.min(Math.max(DATA.selected_index, 0), Math.max(N - 1, 0)), false);
if (DATA.autoplay) setPlaying(true);

// Debug handle for automated checks.
window.__cg_opt_debug = {
  step: () => current,
  playing: () => playing,
  frames: () => (viewer ? viewer.getNumFrames() : -1)
};
</script>
</body></html>
"""


def _sampled_indices(total: int, max_frames: int) -> list[int]:
    """Return evenly spaced frame indices, always keeping the first and last.

    Parameters
    ----------
    total : int
        Number of frames in the trajectory.
    max_frames : int
        Target maximum (``0`` or ``None`` keeps every frame).

    Returns
    -------
    list[int]
        Sorted frame indices.
    """
    if total <= 0:
        return []
    if not max_frames or total <= max_frames:
        return list(range(total))
    stride = -(-total // max_frames)  # ceil division
    indices = list(range(0, total, stride))
    if indices[-1] != total - 1:
        indices.append(total - 1)
    return indices


def read_optimization_steps(
    path: str, max_frames: int = DEFAULT_MAX_FRAMES
) -> Optional[tuple[list[dict], str]]:
    """Read per-step energies, forces and geometries from an ASE trajectory.

    Parameters
    ----------
    path : str
        ASE ``.traj`` file written by the optimizer.
    max_frames : int, optional
        Downsample evenly to at most this many frames (the first and last
        step are always kept) to bound the embedded payload.

    Returns
    -------
    tuple[list[dict], str] or None
        ``(steps, frames_xyz)`` where each step is
        ``{"step": int, "energy": float, "fmax": float | None}`` and
        ``frames_xyz`` is the matching multi-model XYZ text consumed by
        3Dmol.js ``addModelsAsFrames``.  ``None`` when the file cannot be
        read or holds no energies.
    """
    try:
        from ase.io.trajectory import Trajectory

        # Pick the frame indices first and read only those frames, so the
        # frame limit also bounds memory and load time for long runs.
        records: list[tuple[int, object]] = []
        with Trajectory(path) as traj:
            for index in _sampled_indices(len(traj), max_frames):
                records.append((index, traj[index]))
    except Exception:
        return None
    if not records:
        return None

    steps: list[dict] = []
    xyz_blocks: list[str] = []
    for index, atoms in records:
        try:
            energy = float(atoms.get_potential_energy())
        except Exception:
            continue
        try:
            forces = atoms.get_forces()
            fmax: Optional[float] = float((forces**2).sum(axis=1).max() ** 0.5)
        except Exception:
            fmax = None
        symbols = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        lines = [str(len(symbols)), f"Step {index}"]
        lines += [
            f"{s} {x:.6f} {y:.6f} {z:.6f}"
            for s, (x, y, z) in zip(symbols, positions)
        ]
        xyz_blocks.append("\n".join(lines))
        steps.append({"step": index, "energy": energy, "fmax": fmax})
    if not steps:
        return None
    return steps, "\n".join(xyz_blocks)


def build_opt_explorer_html(
    steps: list[dict],
    frames_xyz: str,
    selected_index: int = -1,
    height: int = 430,
    interval_ms: int = 250,
    autoplay: bool = False,
) -> str:
    """Build the self-contained HTML for the optimization explorer.

    Parameters
    ----------
    steps : list[dict]
        Per-frame records ``{"step": int, "energy": float, "fmax": float | None}``
        aligned with the frames in *frames_xyz*.
    frames_xyz : str
        Multi-model XYZ text with one block per step.
    selected_index : int, optional
        Frame shown initially; negative values select the last frame
        (the converged geometry).
    height : int, optional
        Component height in pixels.
    interval_ms : int, optional
        Delay between frames while playing.
    autoplay : bool, optional
        Start the animation immediately.

    Returns
    -------
    str
        Complete HTML document.
    """
    if selected_index < 0:
        selected_index = max(len(steps) - 1, 0)
    payload = json.dumps(
        {
            "steps": steps,
            "frames": frames_xyz,
            "selected_index": selected_index,
            "interval_ms": int(interval_ms),
            "autoplay": bool(autoplay),
        }
    ).replace("</", "<\\/")
    return (
        _TEMPLATE.replace("__3DMOL_JS__", _3DMOL_JS)
        .replace("__PLOTLY_JS__", _PLOTLY_JS)
        .replace("__HEIGHT__", str(height))
        .replace("__PAYLOAD__", payload)
    )


def render_opt_explorer(
    steps: list[dict],
    frames_xyz: str,
    selected_index: int = -1,
    height: int = 430,
    interval_ms: int = 250,
    autoplay: bool = False,
) -> None:
    """Render the combined viewer + convergence-plot component."""
    html = build_opt_explorer_html(
        steps, frames_xyz, selected_index, height, interval_ms, autoplay
    )
    st.components.v1.html(html, height=height + 12, scrolling=False)
