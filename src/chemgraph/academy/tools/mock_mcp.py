"""Deterministic mock MCP server for iteration + reviewer reproducibility.

Serves matched-schema stand-ins for the science tools the mof-crux-aurora
pipeline calls: mofforge_build, run_ase, pacmof2_assign_charges,
run_graspa. Each returns plausibly-shaped output derived from a hash of
its args (so the LLM sees stable numbers across retries) and writes a
plausible output file so downstream tools' inputs exist on disk.

Wall time per call is proportional to a token in the tool's real cost
(cycles, steps, atoms) but capped low so a full pipeline finishes in
under a minute. Real: ~15 min. Mock: <10 s.

Same CLI shape as the other MCP servers in this package.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("chemgraph.academy.tools.mock")

mcp = FastMCP(
    name="mock",
    instructions=(
        "Mock science tools for the mof-crux-aurora pipeline. Behaves like "
        "the real mofforge/UMA/PACMOF2/gRASPA tools (same schemas, same "
        "output shapes, plausible numbers), but runs in seconds. Use for "
        "iteration and reviewer reproducibility."
    ),
)


def _seed(*keys: Any) -> float:
    key = "|".join(str(k) for k in keys).encode()
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _write_stub_cif(path: Path, source: str, atoms: int = 98) -> None:
    """Write a minimal CIF-shaped placeholder. Deterministic contents so
    downstream tools receive the same bytes on retry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        f"data_mock\n_cell_length_a  20.000\n_cell_length_b  20.000\n"
        f"_cell_length_c  20.000\n_atom_site_label\n"
        + "".join(f"C{i}\n" for i in range(atoms))
        + f"# derived_from: {source}\n"
    )
    path.write_text(body)


def mofforge_build(
    topology: str,
    backend: str,
    node_files: list[str],
    edge_files: list[str],
    output_dir: str,
) -> dict[str, Any]:
    time.sleep(1.0)
    stem = f"{topology}_{Path(node_files[0]).stem}_{Path(edge_files[0]).stem}"
    out = Path(output_dir) / f"{stem}.cif"
    _write_stub_cif(out, source="mofforge_build")
    return {
        "success": True,
        "topology": topology,
        "backend": backend,
        "elapsed_seconds": 1.0,
        "output_paths": [str(out)],
        "atoms": 98,
    }


def run_ase(params: dict[str, Any]) -> dict[str, Any]:
    input_path = params.get("input_structure_file")
    output_json = params.get("output_results_file", "opt_results.json")
    driver = params.get("driver", "opt")
    calc = params.get("calculator", {}) or {}
    if not calc.get("task_name") or not calc.get("model_name"):
        return {
            "is_error": True,
            "error": (
                "calculator.task_name and calculator.model_name are required "
                "for FAIRChem (mock enforces same schema as the real tool)."
            ),
        }
    time.sleep(2.0)
    seed = _seed(input_path, driver, calc.get("model_name"))
    energy = -(500.0 + seed * 300.0)
    result = {
        "success": True,
        "driver": driver,
        "converged": True,
        "final_energy_eV": round(energy, 4),
        "steps": 12,
        "wall_time_s": 2.0,
        "input_structure_file": input_path,
        "output_results_file": output_json,
    }
    _write_json(Path(output_json), result)
    return result


def pacmof2_assign_charges(
    cif_path: str,
    output_dir: str,
    identifier: str = "_pacmof",
    net_charge: float | None = None,
) -> dict[str, Any]:
    time.sleep(1.0)
    src = Path(cif_path)
    charges_cif = Path(output_dir) / f"{src.stem}{identifier}.cif"
    _write_stub_cif(charges_cif, source=f"pacmof2_from:{src.name}")
    return {
        "success": True,
        "cif_path": cif_path,
        "charges_cif": str(charges_cif),
        "elapsed_seconds": 1.0,
        "net_charge": net_charge if net_charge is not None else 0.0,
    }


def run_graspa(
    input_structure_file: str,
    adsorbate: str,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    n_cycles: int = 10000,
    output_result_file: str = "raspa.log",
) -> dict[str, Any]:
    cif = Path(input_structure_file)
    if not cif.is_file():
        return {"is_error": True, "error": f"CIF not found: {input_structure_file}"}
    time.sleep(min(3.0, n_cycles / 1e5))
    seed_a = _seed(input_structure_file, adsorbate, temperature, pressure)
    seed_b = _seed(input_structure_file, adsorbate, "b")
    seed_c = _seed(input_structure_file, adsorbate, "c")
    result = {
        "status": "ok",
        "input_structure_file": input_structure_file,
        "adsorbate": adsorbate,
        "temperature": temperature,
        "pressure": pressure,
        "n_cycles": n_cycles,
        "uptake_mol_kg": round(0.5 + seed_a * 11.5, 4),
        "uptake_cm3_stp_g": round(15.0 + seed_b * 335.0, 3),
        "avg_energy_kj_mol": round(-(5.0 + seed_c * 30.0), 3),
        "wall_time_s": 3.0,
        "output_result_file": output_result_file,
    }
    _write_json(Path(output_result_file), result)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Mock science MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "streamable_http"],
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9016)
    args = parser.parse_args()

    mcp.add_tool(
        mofforge_build,
        name="mofforge_build",
        description=(
            "Build a MOF structure from topology + node/edge building "
            "blocks. Mock: returns a stub CIF at output_paths[0]."
        ),
    )
    mcp.add_tool(
        run_ase,
        name="run_ase",
        description=(
            "ASE geometry optimization / single-point / MD driver. Mock: "
            "returns a plausible energy + writes results JSON. Requires "
            "calculator.task_name and calculator.model_name (same schema "
            "as the real tool)."
        ),
    )
    mcp.add_tool(
        pacmof2_assign_charges,
        name="pacmof2_assign_charges",
        description=(
            "PACMOF2 partial-charge assignment. Mock: writes a stub "
            "charged CIF and returns its path."
        ),
    )
    mcp.add_tool(
        run_graspa,
        name="run_graspa",
        description=(
            "gRASPA GCMC uptake simulation. Mock: returns plausible uptake "
            "numbers + writes result JSON in <3s."
        ),
    )

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
