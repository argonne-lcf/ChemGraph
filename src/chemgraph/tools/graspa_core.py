"""Compatibility helpers for the original single-component gRASPA API."""

from __future__ import annotations

from pathlib import Path

from ase.io import read as ase_read

from chemgraph.schemas.graspa_schema import graspa_input_schema
from chemgraph.tools.adsorption_core import run_adsorption_core
from chemgraph.tools.adsorption_config import AdsorptionRuntimeConfig


def _read_graspa_sycl_output(
    output_path: str,
    adsorbate: str = "H2O",
    cifname: str | None = None,
    output_fname: str = "raspa.log",
    temperature: float | None = None,
    pressure: float | None = None,
) -> dict:
    """Parse legacy SYCL molecule-count output for compatibility."""

    directory = Path(output_path)
    target = directory / Path(output_fname).name
    cifs = list(directory.glob("*.cif")) if cifname is None else [directory / f"{cifname}.cif"]
    result = {
        "status": "failure",
        "uptake_in_mol_kg": 0.0,
        "adsorbate": adsorbate,
        "temperature_in_K": temperature,
        "pressure_in_Pa": pressure,
        "cif_path": str(cifs[0].resolve()) if len(cifs) == 1 else None,
    }
    if not target.is_file() or len(cifs) != 1 or not cifs[0].is_file():
        return result

    unit_cells = None
    average = None
    for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        if "UnitCells" in line:
            unit_cells = [int(float(value)) for value in line.split()[4:7]]
        elif "Overall: Average:" in line:
            average = float(line.split()[2].rstrip(","))
    if unit_cells is None or average is None:
        return result

    atoms = ase_read(cifs[0])
    mass = float(sum(atoms.get_masses()))
    for multiplier in unit_cells:
        mass *= multiplier
    result["uptake_in_mol_kg"] = average / mass * 1000.0
    result["status"] = "success"
    return result


def mock_graspa(params: graspa_input_schema) -> dict:
    """Return a deterministic mock result without sleeping or running gRASPA."""

    return {
        "status": "success",
        "uptake_in_mol_kg": 1.0,
        "adsorbate": params.adsorbate,
        "temperature_in_K": float(params.temperature),
        "pressure_in_Pa": float(params.pressure),
    }


def run_graspa_core(
    params: graspa_input_schema,
    *,
    runtime: AdsorptionRuntimeConfig | dict | None = None,
    config_path: str | None = None,
) -> dict:
    """Run the generic engine through the legacy single-component contract."""

    result = run_adsorption_core(
        params.to_adsorption_request(), runtime=runtime, config_path=config_path
    )
    result["adsorbate"] = params.adsorbate
    result["temperature_in_K"] = float(params.temperature)
    result["pressure_in_Pa"] = float(params.pressure)
    result["output_result_file"] = result["stdout_path"]
    if result["status"] == "success" and result["components"]:
        result["uptake_in_mol_kg"] = result["components"][0]["uptake"]
    else:
        result["uptake_in_mol_kg"] = 0.0
    return result


__all__ = [
    "_read_graspa_sycl_output",
    "mock_graspa",
    "run_graspa_core",
]
