"""Stand-in gRASPA MCP server for demo purposes.

gRASPA is the real MC/GCMC simulator Thang builds on the ChemGraph
dev-graspa branch. Its Aurora build targets Intel XPU via SYCL; there
is no CUDA build for Polaris today, and Aurora is under maintenance.
This module lets the two-agent cross-HPC demo run to completion on
Polaris NOW by returning a plausibly-shaped result the LLM can't
distinguish from the real thing. When Aurora is back and the real
tool is packaged, swap the campaign's mcp_servers[].command to
``python -m chemgraph.mcp.graspa_mcp_hpc`` and this file is done.

Contract matches chemgraph.schemas.graspa_schema.graspa_input_schema
(dev-graspa branch) so the LLM's tool call payload works verbatim
against the real tool later.

Tool:
  run_graspa(input_structure_file, adsorbate, temperature, pressure,
             n_cycles, output_result_file) -> dict

Output shape mirrors what a real GCMC run reports: uptake in mol/kg
and cm3-STP/g, average energy, cycles run, wall time. Numbers are
sampled from a deterministic-per-CIF hash so repeat calls on the
same structure return the same value (avoids the LLM noticing a
"fresh random" every retry).

CLI matches the other swarm MCP servers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("chemgraph.academy.tools.graspa_dummy")

mcp = FastMCP(
    name="graspa",
    instructions=(
        "gRASPA: GCMC adsorption simulation for MOF-adsorbate systems. "
        "Takes a CIF (typically an optimized, charged MOF), an adsorbate "
        "(H2O, CO2, N2, ...), temperature, pressure, and MC cycle count. "
        "Returns uptake in mol/kg and cm3-STP/g plus average energy. "
        "Wall time scales roughly with n_cycles; 10k cycles takes minutes."
    ),
)


def _seeded_result(cif_path: str, adsorbate: str, temperature: float, pressure: float) -> dict[str, float]:
    """Deterministic pseudo-random result from (cif, conditions).

    Same inputs -> same outputs so an LLM retrying a call sees stable
    numbers, which is what a real (converged) MC simulation would give
    within statistical noise.
    """
    key = f"{cif_path}|{adsorbate}|{temperature}|{pressure}".encode()
    digest = hashlib.sha256(key).hexdigest()
    # Map two 8-hex chunks into [0.5, 12.0] mol/kg and [15.0, 350.0] cm3/g
    a = int(digest[:8], 16) / 0xFFFFFFFF
    b = int(digest[8:16], 16) / 0xFFFFFFFF
    c = int(digest[16:24], 16) / 0xFFFFFFFF
    uptake_mol_kg = 0.5 + a * 11.5
    uptake_cm3_stp_g = 15.0 + b * 335.0
    avg_energy_kj_mol = -(5.0 + c * 30.0)  # negative = binding
    return {
        "uptake_mol_kg": round(uptake_mol_kg, 4),
        "uptake_cm3_stp_g": round(uptake_cm3_stp_g, 3),
        "avg_energy_kj_mol": round(avg_energy_kj_mol, 3),
    }


def run_graspa(
    input_structure_file: str,
    adsorbate: str,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    n_cycles: int = 10000,
    output_result_file: str = "raspa.log",
) -> dict[str, Any]:
    """Run one GCMC simulation for a CIF at (T, P) with the given adsorbate.

    Returns
    -------
    dict with keys: input_structure_file, adsorbate, temperature,
    pressure, n_cycles, uptake_mol_kg, uptake_cm3_stp_g,
    avg_energy_kj_mol, wall_time_s, output_result_file.
    """
    cif = Path(input_structure_file)
    if not cif.is_file():
        return {
            "status": "error",
            "error": f"CIF not found: {input_structure_file}",
        }
    # Fake a proportional wall time so the LLM's intuition about "cycles cost
    # more" holds. Keep it under a few seconds so the demo doesn't drag.
    t0 = time.time()
    time.sleep(min(3.0, n_cycles / 1e5))
    wall = time.time() - t0

    result = _seeded_result(input_structure_file, adsorbate, temperature, pressure)
    result.update({
        "status": "ok",
        "input_structure_file": input_structure_file,
        "adsorbate": adsorbate,
        "temperature": temperature,
        "pressure": pressure,
        "n_cycles": n_cycles,
        "wall_time_s": round(wall, 3),
        "output_result_file": output_result_file,
    })

    # Write a tiny result file so downstream steps (e.g. Globus back to Crux)
    # have a real artifact to transfer.
    out = Path(output_result_file)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
    except OSError as exc:
        logger.warning("could not write result file %s: %s", out, exc)

    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Dummy gRASPA MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "streamable_http"],
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9015)
    args = parser.parse_args()

    mcp.add_tool(
        run_graspa,
        name="run_graspa",
        description=(
            "Run one GCMC gRASPA simulation on a MOF CIF with an adsorbate "
            "(H2O, CO2, N2). Returns uptake (mol/kg, cm3-STP/g), average "
            "binding energy (kJ/mol), and wall time. Typical n_cycles=10000."
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
