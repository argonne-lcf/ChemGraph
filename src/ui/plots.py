"""Interactive Plotly figures for ChemGraph artifacts.

Data-reading helpers are Streamlit-free so they can be unit-tested; the
figures are rendered by the pages with ``st.plotly_chart``, whose
built-in Streamlit theme adapts fonts and surfaces to light/dark mode.
"""

from __future__ import annotations

import csv
from typing import Optional

import plotly.graph_objects as go

# Single-series data-mark color, validated for chroma (>= 0.1) and
# contrast (>= 3:1) against both the light and dark Streamlit surfaces.
LINE_COLOR = "#0E9594"

_GRID_STYLE = {"showgrid": True, "gridwidth": 1, "zeroline": False}


def load_ir_spectrum_csv(
    path: str,
) -> Optional[tuple[list[float], list[float]]]:
    """Read a ``frequency_cm1,intensity`` CSV written by the IR driver.

    Parameters
    ----------
    path : str
        Spectrum CSV path.

    Returns
    -------
    tuple[list[float], list[float]] or None
        ``(frequencies, intensities)``, or ``None`` when the file is
        missing/empty/malformed.
    """
    frequencies: list[float] = []
    intensities: list[float] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                try:
                    freq, intensity = float(row[0]), float(row[1])
                except ValueError:
                    continue  # header or junk line
                frequencies.append(freq)
                intensities.append(intensity)
    except OSError:
        return None
    if not frequencies:
        return None
    return frequencies, intensities


def ir_spectrum_figure(
    frequencies: list[float], intensities: list[float]
) -> go.Figure:
    """Build an interactive IR spectrum figure.

    Follows IR convention: wavenumber decreases left to right.

    Parameters
    ----------
    frequencies : list[float]
        Spectrum frequencies in cm^-1.
    intensities : list[float]
        Absorption intensities.

    Returns
    -------
    plotly.graph_objects.Figure
        Configured single-series line figure.
    """
    fig = go.Figure(
        go.Scatter(
            x=frequencies,
            y=intensities,
            mode="lines",
            line={"color": LINE_COLOR, "width": 2},
            name="IR spectrum",
            hovertemplate=(
                "%{x:.0f} cm⁻¹<br>intensity %{y:.3g}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis={
            "title": {"text": "Wavenumber (cm⁻¹)"},
            "autorange": "reversed",
            **_GRID_STYLE,
        },
        yaxis={
            "title": {"text": "Intensity ((D/Å)² amu⁻¹)"},
            **_GRID_STYLE,
        },
        showlegend=False,
        hovermode="x",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=380,
    )
    return fig


def read_optimization_trajectory(
    path: str,
) -> Optional[tuple[list[float], list[Optional[float]]]]:
    """Extract per-step energies and max forces from an ASE trajectory.

    Parameters
    ----------
    path : str
        ASE ``.traj`` file written by the optimizer.

    Returns
    -------
    tuple or None
        ``(energies_eV, fmax_eV_per_A)`` per optimization step; force
        entries are ``None`` for frames without stored forces. ``None``
        when the file cannot be read or holds no energies.
    """
    try:
        from ase.io.trajectory import Trajectory

        energies: list[float] = []
        fmax_values: list[Optional[float]] = []
        with Trajectory(path) as traj:
            for atoms in traj:
                energies.append(float(atoms.get_potential_energy()))
                try:
                    forces = atoms.get_forces()
                    fmax_values.append(
                        float((forces**2).sum(axis=1).max() ** 0.5)
                    )
                except Exception:
                    fmax_values.append(None)
    except Exception:
        return None
    if not energies:
        return None
    return energies, fmax_values


def convergence_figure(
    energies: list[float], fmax_values: Optional[list[Optional[float]]] = None
) -> go.Figure:
    """Build an optimization-convergence figure.

    Energy and max force are different measures, so they get their own
    stacked panels sharing the step axis (never a dual axis).

    Parameters
    ----------
    energies : list[float]
        Potential energy per optimization step (eV).
    fmax_values : list[float | None], optional
        Max force per step (eV/Å); the force panel is added only when at
        least one value is present.

    Returns
    -------
    plotly.graph_objects.Figure
        Configured convergence figure.
    """
    from plotly.subplots import make_subplots

    steps = list(range(len(energies)))
    have_forces = bool(fmax_values) and any(
        v is not None for v in fmax_values
    )
    rows = 2 if have_forces else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=energies,
            mode="lines+markers",
            line={"color": LINE_COLOR, "width": 2},
            marker={"size": 8},
            name="Energy",
            hovertemplate="step %{x}<br>%{y:.6f} eV<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title={"text": "Energy (eV)"}, row=1, col=1, **_GRID_STYLE
    )
    if have_forces:
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=[v for v in fmax_values],
                mode="lines+markers",
                line={"color": LINE_COLOR, "width": 2},
                marker={"size": 8},
                name="Max force",
                hovertemplate="step %{x}<br>%{y:.4f} eV/Å<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title={"text": "Max force (eV/Å)"},
            type="log",
            dtick=1,  # one tick per decade; SI-style log ticks are ambiguous
            row=2,
            col=1,
            **_GRID_STYLE,
        )
        fig.update_xaxes(title={"text": "Step"}, row=2, col=1, **_GRID_STYLE)
    else:
        fig.update_xaxes(title={"text": "Step"}, row=1, col=1, **_GRID_STYLE)
    fig.update_layout(
        showlegend=False,
        hovermode="x",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=420 if have_forces else 300,
    )
    return fig
