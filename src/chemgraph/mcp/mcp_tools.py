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
from chemgraph.schemas.pyscf_schema import (
    PySCFMolecularInput,
    PySCFPeriodicInput,
    PySCFPropertyInput,
    PySCFRecipeInput,
)
from chemgraph.tools.pyscf_tools import (
    extract_pyscf_output_core,
    get_pyscf_capability_manifest_core,
    run_pyscf_molecular_core,
    run_pyscf_periodic_core,
    run_pyscf_property_core,
    run_pyscf_recipe_core,
)


mcp = FastMCP(
    name="ChemGraph General Tools",
    instructions="""
        You provide chemistry tools for converting molecule names to SMILES,
        building 3D coordinates, running ASE simulations (geometry optimization, thermochemistry, vibrational calculations), and reading results.
        Each tool has its own description; follow those to decide when to use them.\n\n
        General guidance:\n
        • Keep outputs compact; large results are written to files.\n
        • Do not invent data. If a tool raises an error, report it as-is.\n
        • Use absolute file paths when returning artifacts.\n
        • Energies are in eV, vibrational frequencies in cm⁻¹, wall times in seconds.
        • PySCF tools report electronic energies in Hartree and eV.
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


@mcp.tool(
    name="get_pyscf_capability_manifest",
    description=(
        "Return ChemGraph's PySCF capability manifest: exposed PySCF MCP "
        "tools, supported options, and current limitations."
    ),
)
def get_pyscf_capability_manifest() -> dict:
    """Return the supported PySCF MCP surface and limitations."""
    return get_pyscf_capability_manifest_core()


@mcp.tool(
    name="run_pyscf_molecular",
    description=(
        "Run a molecular PySCF calculation. Supports HF/DFT references, optional "
        "MP2/CCSD/CCSD(T), and selected properties such as dipole, population, "
        "MO energies, and gradients."
    ),
)
def run_pyscf_molecular(params: PySCFMolecularInput) -> dict:
    """Run the main molecular PySCF workflow."""
    return run_pyscf_molecular_core(params)


@mcp.tool(
    name="run_pyscf_periodic",
    description=(
        "Run a minimal periodic PySCF calculation for a cell using HF/DFT references "
        "and optional k-point meshes."
    ),
)
def run_pyscf_periodic(params: PySCFPeriodicInput) -> dict:
    """Run the minimal periodic PySCF workflow."""
    return run_pyscf_periodic_core(params)


@mcp.tool(
    name="run_pyscf_property",
    description=(
        "Extract properties already stored in a JSON artifact from a previous "
        "PySCF run."
    ),
)
def run_pyscf_property(params: PySCFPropertyInput) -> dict:
    """Extract stored PySCF properties from a prior result JSON."""
    return run_pyscf_property_core(params)


@mcp.tool(
    name="run_pyscf_recipe",
    description=(
        "Run a whitelisted advanced PySCF recipe. Current recipe: casscf_single_point."
    ),
)
def run_pyscf_recipe(params: PySCFRecipeInput) -> dict:
    """Run a whitelisted PySCF recipe."""
    return run_pyscf_recipe_core(params)


@mcp.tool(
    name="extract_pyscf_output",
    description="Load a JSON artifact produced by a ChemGraph PySCF MCP tool.",
)
def extract_pyscf_output(json_file: str) -> dict:
    """Load a saved PySCF JSON result."""
    return extract_pyscf_output_core(json_file)


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    run_mcp_server(mcp, default_port=9003)
