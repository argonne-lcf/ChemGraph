"""FastMCP server exposing general chemistry tools.

The ``run_ase`` tool delegates to :func:`chemgraph.tools.ase_core.run_ase_core`
so that the simulation logic lives in a single place.
"""

from __future__ import annotations

import io
import logging
from contextlib import redirect_stdout
from typing import Any, Callable, Literal

from mcp.server.fastmcp import FastMCP

from chemgraph.tools.ase_core import extract_output_json_core, run_ase_core
from chemgraph.tools.cheminformatics_core import (
    molecule_name_to_smiles_core,
    smiles_to_coordinate_file_core,
)
from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.schemas.pyscf_schema import (
    PySCFCrystalReference,
    PySCFDevice,
    PySCFDriver,
    PySCFMoleculeReference,
    PySCFUnit,
)
from chemgraph.tools.pyscf_tools import (
    create_pyscf_crystal_core,
    create_pyscf_molecule_core,
    run_pyscf_crystal_core,
    run_pyscf_molecule_core,
)

mcp = FastMCP(
    name="ChemGraph General Tools",
    instructions="""
        You provide chemistry tools for converting molecule names to SMILES,
        building 3D coordinates, running ASE simulations (geometry optimization, thermochemistry, vibrational calculations), creating PySCF molecule/crystal objects, and running PySCF workflows.
        Each tool has its own description; follow those to decide when to use them.\n\n
        General guidance:\n
        • Keep outputs compact; large results are written to files.\n
        • Do not invent data. If a tool raises an error, report it as-is.\n
        • Use absolute file paths when returning artifacts.\n
        • Energies are in eV, vibrational frequencies in cm⁻¹, wall times in seconds.
        • PySCF tools report electronic energies in Hartree and eV.
    """,
)

logger = logging.getLogger(__name__)


def _call_core_silencing_stdout(
    core_fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Run a tool core without letting library prints corrupt MCP stdio.

    MCP stdio transport reserves stdout for JSON-RPC messages. PySCF and some
    scientific libraries print progress logs to stdout, so wrappers must capture
    those logs and return a compact diagnostic summary instead.
    """

    stdout_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer):
            result = core_fn(*args, **kwargs)
    except Exception:
        captured = stdout_buffer.getvalue()
        if captured:
            logger.warning(
                "Captured stdout before tool error in %s:\n%s",
                getattr(core_fn, "__name__", repr(core_fn)),
                captured[-4000:],
            )
        raise

    captured = stdout_buffer.getvalue()
    if captured and isinstance(result, dict):
        diagnostics = result.setdefault("diagnostics", {})
        diagnostics["captured_stdout_line_count"] = len(captured.splitlines())
        diagnostics["captured_stdout_tail"] = captured[-4000:]
    elif captured:
        logger.info(
            "Captured stdout from %s:\n%s",
            getattr(core_fn, "__name__", repr(core_fn)),
            captured[-4000:],
        )
    return result


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
    return _call_core_silencing_stdout(run_ase_core, params)


@mcp.tool(
    name="create_pyscf_molecule",
    description=(
        "Create a JSON-serializable PySCF molecule from a structure file. "
        "The returned pyscf_molecule object, or the JSON file written via "
        "output_json, is intended for run_pyscf_molecule."
    ),
)
def create_pyscf_molecule(
    structure_file: str,
    charge: int = 0,
    spin: int = 0,
    basis: str = "sto-3g",
    unit: PySCFUnit = "Angstrom",
    reference: PySCFMoleculeReference = "RHF",
    xc: str | None = None,
    device: PySCFDevice = "cpu",
    fmt: str | None = None,
    max_memory: int = 4000,
    verbose: int = 0,
    output_json: str | None = None,
) -> dict:
    """Create a PySCF molecule specification from a structure file."""
    return _call_core_silencing_stdout(
        create_pyscf_molecule_core,
        structure_file,
        charge=charge,
        spin=spin,
        basis=basis,
        unit=unit,
        reference=reference,
        xc=xc,
        device=device,
        fmt=fmt,
        max_memory=max_memory,
        verbose=verbose,
        output_json=output_json,
    )


@mcp.tool(
    name="create_pyscf_crystal",
    description=(
        "Create a JSON-serializable PySCF periodic crystal from a structure file "
        "with lattice vectors. The returned pyscf_crystal object is intended for "
        "run_pyscf_crystal; the JSON file written via output_json can also be "
        "passed to run_pyscf_crystal."
    ),
)
def create_pyscf_crystal(
    structure_file: str,
    charge: int = 0,
    spin: int = 0,
    basis: str = "gth-szv",
    pseudo: str | None = "gth-pade",
    unit: PySCFUnit = "Angstrom",
    reference: PySCFCrystalReference = "RKS",
    xc: str | None = "pbe",
    kpts: list[int] | None = None,
    device: PySCFDevice = "cpu",
    fmt: str | None = None,
    max_memory: int = 4000,
    verbose: int = 0,
    output_json: str | None = None,
) -> dict:
    """Create a PySCF periodic crystal specification from a structure file."""
    return _call_core_silencing_stdout(
        create_pyscf_crystal_core,
        structure_file,
        charge=charge,
        spin=spin,
        basis=basis,
        pseudo=pseudo,
        unit=unit,
        reference=reference,
        xc=xc,
        kpts=kpts,
        device=device,
        fmt=fmt,
        max_memory=max_memory,
        verbose=verbose,
        output_json=output_json,
    )


@mcp.tool(
    name="run_pyscf_molecule",
    description=(
        "Run a PySCF molecule object returned by create_pyscf_molecule, or load "
        "one from pyscf_molecule_json written by create_pyscf_molecule. "
        "Drivers: energy, optimization, vibration, thermochemistry. "
        "Use device='cpu' or device='gpu'."
    ),
)
def run_pyscf_molecule(
    pyscf_molecule: dict | None = None,
    pyscf_molecule_json: str | None = None,
    driver: PySCFDriver = "optimization",
    device: PySCFDevice | None = None,
    optimizer: str = "bfgs",
    fmax: float = 0.05,
    steps: int = 100,
    displacement: float = 0.01,
    temperature: float = 298.15,
    pressure: float = 101325.0,
    symmetry_number: int = 1,
    optimize_before_analysis: bool = True,
    max_cycle: int = 50,
    conv_tol: float = 1e-9,
    chkfile: str | None = None,
    output_json: str | None = "pyscf_molecule_results.json",
) -> dict:
    """Run a PySCF molecular workflow."""
    return _call_core_silencing_stdout(
        run_pyscf_molecule_core,
        pyscf_molecule,
        pyscf_molecule_json=pyscf_molecule_json,
        driver=driver,
        device=device,
        optimizer=optimizer,
        fmax=fmax,
        steps=steps,
        displacement=displacement,
        temperature=temperature,
        pressure=pressure,
        symmetry_number=symmetry_number,
        optimize_before_analysis=optimize_before_analysis,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        chkfile=chkfile,
        output_json=output_json,
    )


@mcp.tool(
    name="run_pyscf_crystal",
    description=(
        "Run a PySCF crystal object returned by create_pyscf_crystal, or load "
        "one from pyscf_crystal_json written by create_pyscf_crystal. "
        "Drivers: energy, optimization, vibration, thermochemistry. "
        "Use device='cpu' or device='gpu'. Crystal thermochemistry is not "
        "implemented in the first iteration."
    ),
)
def run_pyscf_crystal(
    pyscf_crystal: dict | None = None,
    pyscf_crystal_json: str | None = None,
    driver: PySCFDriver = "energy",
    device: PySCFDevice | None = None,
    optimizer: str = "bfgs",
    fmax: float = 0.05,
    steps: int = 50,
    displacement: float = 0.01,
    force_delta: float = 0.005,
    optimize_before_analysis: bool = False,
    max_cycle: int = 50,
    conv_tol: float = 1e-9,
    output_json: str | None = "pyscf_crystal_results.json",
) -> dict:
    """Run a PySCF periodic workflow."""
    return _call_core_silencing_stdout(
        run_pyscf_crystal_core,
        pyscf_crystal,
        pyscf_crystal_json=pyscf_crystal_json,
        driver=driver,
        device=device,
        optimizer=optimizer,
        fmax=fmax,
        steps=steps,
        displacement=displacement,
        force_delta=force_delta,
        optimize_before_analysis=optimize_before_analysis,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        output_json=output_json,
    )


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    run_mcp_server(mcp, default_port=9003)
