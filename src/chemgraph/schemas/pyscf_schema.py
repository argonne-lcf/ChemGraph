"""Schemas and shared type aliases for PySCF-backed MCP tools."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


PySCFDevice = Literal["cpu", "gpu"]
PySCFDriver = Literal["energy", "optimization", "vibration", "thermochemistry"]
PySCFUnit = Literal["Angstrom", "Bohr"]
PySCFMoleculeReference = Literal["RHF", "UHF", "ROHF", "RKS", "UKS"]
PySCFCrystalReference = Literal[
    "RHF",
    "UHF",
    "RKS",
    "UKS",
    "KRHF",
    "KUHF",
    "KRKS",
    "KUKS",
]


class PySCFMoleculeSpec(BaseModel):
    """JSON-serializable molecular input produced by create_pyscf_molecule."""

    object_type: Literal["pyscf_molecule"] = "pyscf_molecule"
    source_structure_file: str = Field(
        description="Absolute path to the structure file used to create the molecule."
    )
    symbols: List[str] = Field(description="Atomic symbols in input order.")
    positions: List[List[float]] = Field(
        description="Cartesian coordinates matching symbols."
    )
    charge: int = 0
    spin: int = Field(
        default=0,
        description="PySCF spin value, N_alpha - N_beta. This is not multiplicity.",
    )
    basis: str = "sto-3g"
    unit: PySCFUnit = "Angstrom"
    reference: PySCFMoleculeReference = "RHF"
    xc: Optional[str] = Field(
        default=None,
        description="DFT exchange-correlation functional for RKS/UKS.",
    )
    device: PySCFDevice = Field(
        default="cpu",
        description="Default execution device for downstream PySCF runs.",
    )
    max_memory: int = Field(default=4000, description="PySCF max memory in MB.")
    verbose: int = Field(default=0, description="PySCF verbosity level.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_atoms(self):
        if len(self.symbols) != len(self.positions):
            raise ValueError("symbols and positions must have the same length.")
        if not self.symbols:
            raise ValueError("A PySCF molecule requires at least one atom.")
        return self


class PySCFCrystalSpec(BaseModel):
    """JSON-serializable periodic input produced by create_pyscf_crystal."""

    object_type: Literal["pyscf_crystal"] = "pyscf_crystal"
    source_structure_file: str = Field(
        description="Absolute path to the structure file used to create the crystal."
    )
    symbols: List[str] = Field(description="Atomic symbols in input order.")
    positions: List[List[float]] = Field(
        description="Cartesian coordinates matching symbols."
    )
    lattice_vectors: List[List[float]] = Field(
        description="3x3 lattice vectors used for the PySCF periodic Cell."
    )
    pbc: List[bool] = Field(description="Periodic boundary flags from ASE.")
    charge: int = 0
    spin: int = Field(default=0, description="PySCF spin value, N_alpha - N_beta.")
    basis: str = "gth-szv"
    pseudo: Optional[str] = "gth-pade"
    unit: PySCFUnit = "Angstrom"
    reference: PySCFCrystalReference = "RKS"
    xc: Optional[str] = "pbe"
    kpts: List[int] = Field(
        default_factory=lambda: [1, 1, 1],
        description="Monkhorst-Pack k-point mesh for K* references.",
    )
    device: PySCFDevice = Field(
        default="cpu",
        description="Default execution device for downstream PySCF runs.",
    )
    max_memory: int = Field(default=4000, description="PySCF max memory in MB.")
    verbose: int = Field(default=0, description="PySCF verbosity level.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_valid_crystal(self):
        if len(self.symbols) != len(self.positions):
            raise ValueError("symbols and positions must have the same length.")
        if not self.symbols:
            raise ValueError("A PySCF crystal requires at least one atom.")
        if len(self.lattice_vectors) != 3 or any(
            len(vector) != 3 for vector in self.lattice_vectors
        ):
            raise ValueError("lattice_vectors must be a 3x3 matrix.")
        if len(self.kpts) != 3:
            raise ValueError("kpts must contain exactly three integers.")
        return self
