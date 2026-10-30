"""Electronic records consumed without an installed computational engine."""

from typing import Literal
import re
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Orbital(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    index: int = Field(ge=0)
    energy: float
    occupation: float


class ElectronicStructureResult(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    result_type: Literal["electronic_structure"] = "electronic_structure"
    success: Literal[True] = True
    scf_converged: Literal[True] = True
    energy_unit: Literal["eV"] = "eV"
    total_energy: float
    charge: int
    multiplicity: Literal[1] = 1
    electron_count: int = Field(gt=0)
    spin: Literal["restricted"] = "restricted"
    method: str
    basis: str
    basis_convention: Literal["spherical"] = "spherical"
    environment: Literal["gas_phase"] = "gas_phase"
    nbf: int = Field(gt=0)
    nmo: int = Field(gt=0)
    index_base: Literal[0] = 0
    orbitals: list[Orbital]
    homo: float
    lumo: float
    gap: float
    homo_indices: list[int]
    lumo_indices: list[int]
    source: Literal["orbital_energy_table", "mo_header"]
    energy_resolution_ev: float = Field(gt=0)
    scf_evidence: dict
    convergence_policy: str
    engine: str
    engine_version: str
    ase_version: str
    artifacts: dict[str, dict[str, str]]
    working_directory: str
    native_settings: dict
    molecule_id: str | None = None
    conformer_id: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_spectrum(self):
        if not self.scf_evidence or not self.convergence_policy:
            raise ValueError("SCF convergence evidence and policy are required")
        if not {"input", "output"} <= self.artifacts.keys():
            raise ValueError("Native input/output provenance is required")
        for artifact in self.artifacts.values():
            if not artifact.get("path") or not re.fullmatch(
                r"[0-9a-f]{64}", artifact.get("sha256", "")
            ):
                raise ValueError("Artifact provenance requires a path and SHA256")
        if self.nmo != self.nbf or [o.index for o in self.orbitals] != list(
            range(self.nbf)
        ):
            raise ValueError("Spectrum must contain unique contiguous 0..nbf-1 indices")
        if any(
            min(abs(o.occupation), abs(o.occupation - 2)) > 1e-6 for o in self.orbitals
        ):
            raise ValueError("Restricted occupations must be 0 or 2")
        if (
            self.electron_count % 2
            or abs(sum(o.occupation for o in self.orbitals) - self.electron_count)
            > 1e-6
        ):
            raise ValueError("Orbital occupations disagree with electron count")
        occupied = [o for o in self.orbitals if o.occupation > 1]
        virtual = [o for o in self.orbitals if o.occupation < 1]
        if not occupied or not virtual:
            raise ValueError("Both occupied and virtual orbitals are required")
        homo, lumo = max(o.energy for o in occupied), min(o.energy for o in virtual)
        if any(
            abs(a - b) > 1e-8
            for a, b in ((homo, self.homo), (lumo, self.lumo), (lumo - homo, self.gap))
        ):
            raise ValueError("Cached frontier summaries disagree with orbitals")
        tolerance = max(1e-4, self.energy_resolution_ev)
        if self.homo_indices != [
            o.index for o in occupied if abs(o.energy - homo) <= tolerance
        ] or self.lumo_indices != [
            o.index for o in virtual if abs(o.energy - lumo) <= tolerance
        ]:
            raise ValueError("Frontier index groups disagree with orbitals")
        return self


class ASEFailureRecord(BaseModel):
    """Readable even when validation failed before an ASE input/geometry existed."""

    result_type: Literal["ase_failure"] = "ase_failure"
    schema_version: Literal[1] = 1
    success: Literal[False] = False
    status: Literal["failure"] = "failure"
    driver: str | None
    stage: str
    error_type: str
    message: str
    artifacts: list[str] = Field(default_factory=list)
    simulation_input: dict | None = None
    final_structure: dict | None = None
    provenance: dict = Field(default_factory=dict)
