"""HTTP-based MCP server for ChemGraph chemistry tools.

This is a thin wrapper that delegates to the core implementations
in :mod:`chemgraph.tools.ase_core` and :mod:`chemgraph.tools.cheminformatics_core`.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from typing import Literal

import uvicorn
from mcp.server.fastmcp import FastMCP

from chemgraph.tools.ase_core import extract_output_json_core, run_ase_core
from chemgraph.tools.cheminformatics_core import (
    molecule_name_to_smiles_core,
    smiles_to_coordinate_file_core,
)
from chemgraph.schemas.ase_input import ASEInputSchema

mcp = FastMCP(
    name="Chemistry Tools MCP",
    instructions=(
        "You provide chemistry tools for converting molecule names to SMILES, "
        "building 3D coordinates, running ASE simulations, and reading results. "
        "Each tool has its own description — follow those to decide when to use them.\n\n"
        "General guidance:\n"
        "• Keep outputs compact; large results are written to files.\n"
        "• Do not invent data. If a tool raises an error, report it as-is.\n"
        "• Use absolute file paths when returning artifacts.\n"
        "• Energies are in eV, vibrational frequencies in cm-1, wall times in seconds.\n"
    ),
)


@mcp.tool(
    name="molecule_name_to_smiles",
    description="Convert a molecule name to a canonical SMILES string using PubChem.",
)
async def molecule_name_to_smiles(name: str) -> str:
    """Resolve a molecule name to its canonical SMILES via PubChem."""
    return molecule_name_to_smiles_core(name)


@mcp.tool(
    name="smiles_to_coordinate_file",
    description="Convert a SMILES string to a coordinate file",
)
async def smiles_to_coordinate_file(
    smiles: str,
    output_file: str = "molecule.xyz",
    randomSeed: int = 2025,
    fmt: Literal["xyz"] = "xyz",
) -> dict:
    """Convert a SMILES string to a coordinate file on disk."""
    return smiles_to_coordinate_file_core(
        smiles, output_file=output_file, seed=randomSeed, fmt=fmt
    )


@mcp.tool(
    name="extract_output_json",
    description="Load simulation results from a JSON file produced by run_ase.",
)
def extract_output_json(json_file: str) -> dict:
    """Load simulation results from a JSON file produced by run_ase."""
    return extract_output_json_core(json_file)


@mcp.tool(
    name="run_ase",
    description="Run ASE calculations using specified input parameters.",
)
async def run_ase(params: ASEInputSchema) -> dict:
    """Run ASE calculations using specified input parameters."""
    f = io.StringIO()
    with redirect_stdout(f):
        return run_ase_core(params)


app = mcp.streamable_http_app()  # exposes endpoints under /mcp

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9001)
