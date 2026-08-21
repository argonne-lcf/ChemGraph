"""Combined 3D-viewer + IR-spectrum component with hover linking.

Streamlit widgets cannot forward hover events to Python, so linking
"hover a peak" to "animate that normal mode" requires both panels in one
HTML component: a 3Dmol.js viewer (left) and a Plotly.js spectrum
(right) wired together in JavaScript. Hovering a peak highlights it and
plays its mode; leaving reverts to the mode chosen in the Streamlit
dropdown, which is drawn with a persistent highlight.

``build_ir_explorer_html`` is pure (unit-testable); ``render_ir_explorer``
is the thin Streamlit wrapper.
"""

from __future__ import annotations

import json
from typing import Optional

import streamlit as st

# Same CDN builds the rest of the app already relies on (py3Dmol embeds
# 3dmol from jsdelivr; st.plotly_chart bundles its own plotly).
_3DMOL_JS = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"
_PLOTLY_JS = "https://cdn.plot.ly/plotly-2.35.2.min.js"

_TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<script src="__3DMOL_JS__"></script>
<script src="__PLOTLY_JS__"></script>
<style>
  html, body { margin: 0; padding: 0; height: 100%;
               font-family: "Source Sans Pro", sans-serif; }
  #wrap { display: flex; gap: 10px; height: __HEIGHT__px; }
  #viewerbox { flex: 0 0 42%; position: relative;
               border: 1px solid rgba(128,128,128,.35);
               border-radius: 8px; overflow: hidden; background: white; }
  #viewer { width: 100%; height: 100%; position: relative; }
  #modelabel { position: absolute; left: 8px; top: 6px; z-index: 5;
               font-size: 13px; color: #444;
               background: rgba(255,255,255,.8);
               padding: 2px 8px; border-radius: 6px; }
  #spectrum { flex: 1; min-width: 0; }
</style></head>
<body>
<div id="wrap">
  <div id="viewerbox"><div id="viewer"></div><div id="modelabel"></div></div>
  <div id="spectrum"></div>
</div>
<script>
const DATA = __PAYLOAD__;
const ACCENT = "#0E9594";
const AXIS_INK = "#8a8f98";
const peaks = DATA.peaks;

// ---------------- 3D viewer ----------------
// WebGL can be unavailable (remote desktops, headless browsers, old
// GPUs); the spectrum must keep working, so the viewer fails soft.
let viewer = null;
try {
  viewer = $3Dmol.createViewer("viewer", {backgroundColor: "white"});
} catch (e) {
  document.getElementById("modelabel").textContent =
      "3D viewer unavailable (WebGL required)";
}
let activeMode = null;
function showMode(mode) {
  if (mode === activeMode) return;
  activeMode = mode;
  const p = peaks.find(q => q.mode === mode);
  const label = document.getElementById("modelabel");
  if (p) {
    label.textContent = "Mode " + p.mode + " \\u2014 " +
        p.freq.toFixed(1) + (p.imaginary ? "i" : "") + " cm\\u207b\\u00b9";
  }
  if (!viewer) return;
  try {
    viewer.removeAllModels();
    if (!p) { label.textContent = ""; viewer.render(); return; }
    if (!p.frames) { viewer.render(); return; }
    viewer.addModelsAsFrames(p.frames, "xyz");
    viewer.setStyle({}, {stick: {radius: 0.12}, sphere: {scale: 0.25}});
    viewer.zoomTo();
    viewer.animate({loop: "backAndForth", interval: 60});
    viewer.render();
  } catch (e) {
    label.textContent = "3D viewer unavailable (WebGL required)";
    viewer = null;
  }
}

// ---------------- spectrum ----------------
const real = peaks.filter(p => !p.imaginary);
const baseSize = real.map(p => p.mode === DATA.selected_mode ? 13 : 9);
const baseLine = real.map(p => p.mode === DATA.selected_mode ? 2.5 : 1.4);
const traces = [
  {x: DATA.curve.x, y: DATA.curve.y, mode: "lines", hoverinfo: "skip",
   line: {color: ACCENT, width: 2}},
  {x: real.map(p => p.freq), y: real.map(p => p.y), mode: "markers",
   marker: {size: baseSize.slice(), color: "rgba(14,149,148,.25)",
            line: {color: ACCENT, width: baseLine.slice()}},
   customdata: real.map(p => [p.mode, p.intensity]),
   hovertemplate: "mode %{customdata[0]}<br>%{x:.1f} cm\\u207b\\u00b9" +
                  "<br>I = %{customdata[1]:.3g}<extra></extra>"}
];
const selected = real.find(p => p.mode === DATA.selected_mode);
const layout = {
  xaxis: {title: {text: "Wavenumber (cm\\u207b\\u00b9)"},
          autorange: "reversed", color: AXIS_INK,
          gridcolor: "rgba(128,128,128,.25)", zeroline: false},
  yaxis: {title: {text: "Intensity ((D/\\u00c5)\\u00b2 amu\\u207b\\u00b9)"},
          color: AXIS_INK, gridcolor: "rgba(128,128,128,.25)",
          zeroline: false},
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
  showlegend: false, hovermode: "closest",
  margin: {l: 60, r: 10, t: 10, b: 45},
  shapes: selected ? [{type: "line", x0: selected.freq, x1: selected.freq,
                       y0: 0, y1: 1, yref: "paper",
                       line: {color: ACCENT, width: 1, dash: "dot"}}] : []
};
const spec = document.getElementById("spectrum");
Plotly.newPlot(spec, traces, layout, {responsive: true, displayModeBar: false});

function setHighlight(hoverIdx) {
  const sizes = baseSize.slice();
  const widths = baseLine.slice();
  if (hoverIdx !== null) { sizes[hoverIdx] = 17; widths[hoverIdx] = 3; }
  Plotly.restyle(spec, {"marker.size": [sizes],
                        "marker.line.width": [widths]}, [1]);
}
spec.on("plotly_hover", ev => {
  const pt = ev.points && ev.points.find(p => p.curveNumber === 1);
  if (!pt) return;
  setHighlight(pt.pointIndex);
  showMode(real[pt.pointIndex].mode);
});
spec.on("plotly_unhover", () => {
  setHighlight(null);
  showMode(DATA.selected_mode);
});

showMode(DATA.selected_mode);
</script>
</body></html>
"""


def build_ir_explorer_html(
    curve_x: list[float],
    curve_y: list[float],
    peaks: list[dict],
    selected_mode: Optional[int],
    height: int = 430,
) -> str:
    """Build the self-contained HTML for the IR explorer.

    Parameters
    ----------
    curve_x : list[float]
        Broadened-spectrum grid (cm^-1).
    curve_y : list[float]
        Broadened-spectrum intensities.
    peaks : list[dict]
        Per-mode records: ``mode`` (int), ``freq`` (float), ``intensity``
        (float), ``y`` (marker height on the curve), ``imaginary`` (bool),
        and ``frames`` (multi-model XYZ text or ``None``).
    selected_mode : int, optional
        Mode chosen in the dropdown; drawn with a persistent highlight
        and shown in the viewer when nothing is hovered.
    height : int, optional
        Component height in pixels.

    Returns
    -------
    str
        Complete HTML document.
    """
    payload = json.dumps(
        {
            "curve": {"x": curve_x, "y": curve_y},
            "peaks": peaks,
            "selected_mode": selected_mode,
        }
    ).replace("</", "<\\/")
    return (
        _TEMPLATE.replace("__3DMOL_JS__", _3DMOL_JS)
        .replace("__PLOTLY_JS__", _PLOTLY_JS)
        .replace("__HEIGHT__", str(height))
        .replace("__PAYLOAD__", payload)
    )


def render_ir_explorer(
    curve_x: list[float],
    curve_y: list[float],
    peaks: list[dict],
    selected_mode: Optional[int],
    height: int = 430,
) -> None:
    """Render the combined viewer + spectrum component."""
    html = build_ir_explorer_html(curve_x, curve_y, peaks, selected_mode, height)
    st.components.v1.html(html, height=height + 12, scrolling=False)
