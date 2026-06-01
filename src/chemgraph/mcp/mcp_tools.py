"""FastMCP server exposing general chemistry tools.

The ``run_ase`` tool delegates to :func:`chemgraph.tools.ase_core.run_ase_core`
so that the simulation logic lives in a single place.
"""

from __future__ import annotations

from typing import Literal

from mcp.server.fastmcp import FastMCP

from chemgraph.tools.ase_core import extract_output_json_core, run_ase_core
from chemgraph.tools.cheminformatics_core import (
    molecule_name_to_smiles_core,
    smiles_to_coordinate_file_core,
)
from chemgraph.schemas.ase_input import ASEInputSchema


mcp = FastMCP(
    name="ChemGraph General Tools",
    instructions="""
        You provide chemistry tools for converting molecule names to SMILES,
        building 3D coordinates, running ASE simulations (geometry optimization, thermochemistry, vibrational calculations), and reading results. "
        Each tool has its own description — follow those to decide when to use them.\n\n
        General guidance:\n
        • Keep outputs compact; large results are written to files.\n
        • Do not invent data. If a tool raises an error, report it as-is.\n
        • Use absolute file paths when returning artifacts.\n
        • Energies are in eV, vibrational frequencies in cm⁻¹, wall times in seconds.
    """,
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
    seed: int = 2025,
    fmt: Literal["xyz"] = "xyz",
) -> dict:
    """Convert a SMILES string to a coordinate file on disk."""
    return smiles_to_coordinate_file_core(
        smiles, output_file=output_file, seed=seed, fmt=fmt
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
    """Run ASE calculations using specified input parameters.

    Parameters
    ----------
    params : ASEInputSchema
        Input parameters for the ASE calculation

    Returns
    -------
    dict
        Output containing calculation status

    Raises
    ------
    ValueError
        If the calculator is not supported or if the calculation fails
    """
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        return run_ase_core(params)


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    run_mcp_server(mcp, default_port=9003)
