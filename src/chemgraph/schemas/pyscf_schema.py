"""Schemas for PySCF-backed MCP tools."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class StructureInput(BaseModel):
    """Structure input accepted by PySCF tools.

    Provide either ``input_structure_file`` or a PySCF-compatible ``atom`` string.
    File inputs are read through ASE and converted to PySCF atom tuples.
    """

    input_structure_file: Optional[str] = Field(
        default=None,
        description="Path to a structure file readable by ASE, e.g. XYZ, CIF, POSCAR.",
    )
    atom: Optional[str] = Field(
        default=None,
        description=(
            "PySCF-compatible atom specification, e.g. "
            "'O 0 0 0; H 0 0 0.96; H 0.92 0 0'."
        ),
    )
    fmt: Optional[str] = Field(
        default=None,
        description="Optional ASE file format hint for input_structure_file.",
    )
    lattice_vectors: Optional[List[List[float]]] = Field(
        default=None,
        description=(
            "Optional 3x3 lattice vectors for periodic calculations. "
            "If omitted, periodic tools try to read the cell from the structure file."
        ),
    )

    @model_validator(mode="after")
    def require_structure_source(self):
        if not self.input_structure_file and not self.atom:
            raise ValueError("Provide either input_structure_file or atom.")
        return self


class ActiveSpaceInput(BaseModel):
    """Active-space parameters for CASCI/CASSCF-like recipes."""

    ncas: int = Field(description="Number of active orbitals.")
    nelecas: Union[int, List[int]] = Field(
        description=(
            "Number of active electrons. Use an int for spin-balanced active spaces "
            "or [n_alpha, n_beta] for spin-resolved active spaces."
        )
    )


PySCFReference = Literal["RHF", "UHF", "ROHF", "RKS", "UKS"]
PySCFPostHF = Literal["MP2", "CCSD", "CCSD(T)"]
PySCFProperty = Literal[
    "dipole",
    "population",
    "mulliken_population",
    "mo_energy",
    "gradient",
]


class PySCFMolecularInput(BaseModel):
    """Input for the main molecular PySCF MCP tool."""

    structure: StructureInput = Field(description="Molecular structure.")
    charge: int = Field(default=0, description="Total molecular charge.")
    spin: int = Field(
        default=0,
        description=(
            "PySCF spin value, N_alpha - N_beta. This is not multiplicity."
        ),
    )
    basis: str = Field(default="sto-3g", description="AO basis set.")
    unit: Literal["Angstrom", "Bohr"] = Field(
        default="Angstrom", description="Coordinate unit."
    )
    reference: PySCFReference = Field(
        default="RHF",
        description="SCF reference. RKS/UKS trigger DFT and use xc.",
    )
    xc: Optional[str] = Field(
        default=None,
        description="DFT exchange-correlation functional for RKS/UKS, e.g. 'b3lyp'.",
    )
    post_hf: List[PySCFPostHF] = Field(
        default_factory=list,
        description="Optional post-HF methods to run after SCF.",
    )
    properties: List[PySCFProperty] = Field(
        default_factory=list,
        description="Optional properties to compute after SCF.",
    )
    solvent: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Reserved for future solvent support. Non-null values are rejected in v0."
        ),
    )
    output_dir: str = Field(
        default="pyscf_output", description="Directory for JSON/checkpoint artifacts."
    )
    output_json: str = Field(
        default="pyscf_results.json", description="Name or path of result JSON."
    )
    chkfile: Optional[str] = Field(
        default=None,
        description="Optional PySCF checkpoint file path. Defaults under output_dir.",
    )
    max_cycle: int = Field(default=50, description="Maximum SCF cycles.")
    conv_tol: float = Field(default=1e-9, description="SCF convergence tolerance.")
    max_memory: int = Field(default=4000, description="PySCF max memory in MB.")
    verbose: int = Field(default=0, description="PySCF verbosity level.")


class PySCFPeriodicInput(BaseModel):
    """Input for minimal periodic PySCF HF/DFT calculations."""

    structure: StructureInput = Field(description="Periodic structure.")
    charge: int = Field(default=0, description="Cell charge.")
    spin: int = Field(default=0, description="PySCF spin value, N_alpha - N_beta.")
    basis: str = Field(default="gth-szv", description="Periodic AO basis set.")
    pseudo: Optional[str] = Field(
        default="gth-pade", description="Pseudopotential label or None."
    )
    unit: Literal["Angstrom", "Bohr"] = Field(default="Angstrom")
    reference: Literal["RHF", "UHF", "RKS", "UKS", "KRHF", "KUHF", "KRKS", "KUKS"] = (
        Field(default="RKS", description="Periodic SCF reference.")
    )
    xc: Optional[str] = Field(default="pbe", description="DFT functional.")
    kpts: List[int] = Field(
        default_factory=lambda: [1, 1, 1],
        description="Monkhorst-Pack k-point mesh for K* references.",
    )
    output_dir: str = Field(default="pyscf_output")
    output_json: str = Field(default="pyscf_periodic_results.json")
    max_cycle: int = Field(default=50)
    conv_tol: float = Field(default=1e-9)
    max_memory: int = Field(default=4000)
    verbose: int = Field(default=0)


class PySCFPropertyInput(BaseModel):
    """Input for extracting post-processed properties from a saved PySCF summary."""

    result_json: str = Field(
        description="Path to a JSON file produced by a PySCF tool."
    )
    properties: List[PySCFProperty] = Field(
        default_factory=list,
        description="Properties to extract. Empty means return all stored properties.",
    )
    output_dir: Optional[str] = Field(
        default=None, description="Optional directory to write extracted property JSON."
    )
    output_json: str = Field(default="pyscf_property_results.json")


class PySCFRecipeInput(BaseModel):
    """Input for whitelisted advanced PySCF recipes."""

    recipe: Literal["casscf_single_point"] = Field(
        description="Whitelisted recipe name."
    )
    structure: StructureInput = Field(description="Molecular structure.")
    active_space: ActiveSpaceInput = Field(description="CASSCF active space.")
    charge: int = Field(default=0)
    spin: int = Field(default=0)
    basis: str = Field(default="sto-3g")
    unit: Literal["Angstrom", "Bohr"] = Field(default="Angstrom")
    reference: Literal["RHF", "UHF", "ROHF"] = Field(default="RHF")
    output_dir: str = Field(default="pyscf_output")
    output_json: str = Field(default="pyscf_recipe_results.json")
    chkfile: Optional[str] = Field(default=None)
    max_cycle: int = Field(default=50)
    conv_tol: float = Field(default=1e-9)
    max_memory: int = Field(default=4000)
    verbose: int = Field(default=0)
