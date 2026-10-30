"""Versioned, engine-independent molecular preparation records."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

SUPPORTED_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35})


class ConformerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    embedding_method: Literal["ETKDGv3"] = "ETKDGv3"
    force_field: Literal["MMFF94s"] = "MMFF94s"
    threads: Literal[1] = 1
    candidates: int = Field(default=20, ge=1, le=1000, strict=True)
    retained: int = Field(default=5, ge=1, le=1000, strict=True)
    seed: int = Field(default=2025, ge=0, le=2147483647, strict=True)
    rmsd_threshold: float = Field(default=0.5, gt=0)
    mmff_max_iters: int = Field(default=2000, ge=1, le=10000, strict=True)


class ConformerRecord(BaseModel):
    conformer_id: str
    candidate_id: int
    mmff_energy: float = Field(allow_inf_nan=False)
    energy_unit: Literal["kcal/mol"] = "kcal/mol"
    convergence_code: Literal[0] = 0
    xyz_path: str
    xyz_sha256: str


class MolecularRecord(BaseModel):
    schema_version: Literal[1] = 1
    result_type: Literal["conformer_manifest"] = "conformer_manifest"
    success: bool
    molecule_id: str
    isomeric_smiles: str
    formal_charge: int
    multiplicity: Literal[1] = 1
    atom_correspondence: list[dict]
    reference_molblock: str
    rdkit_version: str
    settings: ConformerSettings
    rmsd_atom_set: Literal["heavy", "all"]
    conformers: list[ConformerRecord]
    candidates: list[dict]
    artifacts: list[str]
    sdf_path: str
    sdf_sha256: str
    run_directory: str
    manifest_path: str
