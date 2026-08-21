"""Tests for interactive plot data helpers and figures (ui.plots)."""

from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.io.trajectory import Trajectory

from ui import plots


def _write_spectrum_csv(path: Path) -> None:
    path.write_text(
        "frequency_cm1,intensity\n500.0,0.0\n1500.5,0.25\n3000.0,1.5\n"
    )


def test_load_ir_spectrum_csv_skips_header_and_junk(tmp_path):
    csv_path = tmp_path / "ir_spectrum_water.csv"
    csv_path.write_text(
        "frequency_cm1,intensity\n"
        "500.0,0.0\n"
        "not,a-number\n"
        "1500.5,0.25\n"
    )

    freqs, intens = plots.load_ir_spectrum_csv(str(csv_path))

    assert freqs == [500.0, 1500.5]
    assert intens == [0.0, 0.25]


def test_load_ir_spectrum_csv_handles_missing_and_empty(tmp_path):
    assert plots.load_ir_spectrum_csv(str(tmp_path / "missing.csv")) is None
    empty = tmp_path / "empty.csv"
    empty.write_text("frequency_cm1,intensity\n")
    assert plots.load_ir_spectrum_csv(str(empty)) is None


def test_ir_spectrum_figure_follows_ir_convention(tmp_path):
    csv_path = tmp_path / "ir_spectrum_water.csv"
    _write_spectrum_csv(csv_path)
    freqs, intens = plots.load_ir_spectrum_csv(str(csv_path))

    fig = plots.ir_spectrum_figure(freqs, intens)

    assert fig.layout.xaxis.autorange == "reversed"  # wavenumber convention
    assert fig.layout.showlegend is False  # single series needs no legend
    trace = fig.data[0]
    assert list(trace.x) == freqs
    assert trace.line.color == plots.LINE_COLOR


@pytest.fixture()
def emt_trajectory(tmp_path):
    """A tiny real optimization trajectory written with EMT."""
    from ase.optimize import BFGS

    atoms = Atoms(
        "H2O",
        positions=[[0, 0, 0], [0, 0, 1.2], [0, 1.2, 0]],
        calculator=EMT(),
    )
    traj_path = tmp_path / "water_opt.traj"
    BFGS(atoms, trajectory=str(traj_path)).run(fmax=0.5, steps=4)
    return str(traj_path)


def test_read_optimization_trajectory(emt_trajectory):
    energies, fmax_values = plots.read_optimization_trajectory(emt_trajectory)

    assert len(energies) >= 2
    assert len(fmax_values) == len(energies)
    assert all(isinstance(e, float) for e in energies)
    assert all(v is None or v >= 0 for v in fmax_values)


def test_read_optimization_trajectory_missing_file(tmp_path):
    assert plots.read_optimization_trajectory(str(tmp_path / "no.traj")) is None


def test_convergence_figure_uses_stacked_panels(emt_trajectory):
    energies, fmax_values = plots.read_optimization_trajectory(emt_trajectory)

    fig = plots.convergence_figure(energies, fmax_values)

    assert len(fig.data) == 2  # energy + force panels, never a dual axis
    assert fig.layout.yaxis2.type == "log"
    assert fig.layout.showlegend is False


def test_convergence_figure_without_forces_is_single_panel():
    fig = plots.convergence_figure([1.0, 0.5, 0.4], [None, None, None])

    assert len(fig.data) == 1


def test_load_ir_peaks_csv_parses_modes_and_imaginary_flags(tmp_path):
    path = tmp_path / "ir_peaks_water.csv"
    path.write_text(
        "mode,frequency_cm1,intensity\n"
        "0,45.1200i,0.001\n"
        "6,1595.4321,1.25\n"
        "not,a,row\n"
        "7,3650.0000,0.8\n"
    )

    peaks = plots.load_ir_peaks_csv(str(path))

    assert [p["mode"] for p in peaks] == [0, 6, 7]
    assert peaks[0]["imaginary"] is True
    assert peaks[0]["frequency"] == 45.12
    assert peaks[1] == {
        "mode": 6,
        "frequency": 1595.4321,
        "intensity": 1.25,
        "imaginary": False,
    }
    assert plots.load_ir_peaks_csv(str(tmp_path / "missing.csv")) is None


def test_gaussian_broadening_matches_ase_fold_semantics():
    # ASE fold with normalize=False: each peak is an unnormalized Gaussian
    # of height == intensity and FWHM == width.
    freqs = [1000.0, 3000.0]
    intens = [2.0, 0.5]
    width = 20.0

    grid, spectrum = plots.gaussian_broadened_spectrum(freqs, intens, width)

    import numpy as np

    grid = np.asarray(grid)
    spectrum = np.asarray(spectrum)
    for f, i in zip(freqs, intens):
        peak_height = spectrum[np.argmin(abs(grid - f))]
        assert abs(peak_height - i) < 0.01 * i  # height == intensity
        half = spectrum[np.argmin(abs(grid - (f + width / 2)))]
        assert abs(half - i / 2) < 0.05 * i  # FWHM == width
    # The grid covers both peaks with margin.
    assert grid[0] < freqs[0] and grid[-1] > freqs[-1]


def test_gaussian_broadening_width_smooths_valleys():
    freqs = [1000.0, 1100.0]
    intens = [1.0, 1.0]

    import numpy as np

    def valley(width):
        grid, spec = plots.gaussian_broadened_spectrum(freqs, intens, width)
        grid = np.asarray(grid)
        return np.asarray(spec)[np.argmin(abs(grid - 1050.0))]

    assert valley(5) < valley(30) < valley(80)  # wider peaks fill the gap


def test_trajectory_reader_roundtrip_matches_ase(emt_trajectory):
    energies, _ = plots.read_optimization_trajectory(emt_trajectory)
    with Trajectory(emt_trajectory) as traj:
        expected = [float(a.get_potential_energy()) for a in traj]

    assert energies == expected
