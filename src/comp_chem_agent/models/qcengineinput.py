from pydantic import BaseModel, Field
from typing import Dict, Optional
from comp_chem_agent.models.atomsdata import AtomsData


class AtomicInputWrapper(BaseModel):
    atomsdata: AtomsData = Field(
        description="The atomsdata object to be used for the simulation."
    )
    driver: str = Field(
        default="energy",
        description="Type of quantum chemistry calculation. Supported values: 'energy', 'gradient', 'hessian', 'properties'.",
    )
    model: Optional[Dict[str, str]] = Field(
        default={"method": "SCF", "basis": "sto-3g"},
        description="Specification of the computational model. Contains 'method' (e.g., 'B3LYP') and 'basis' (e.g., '6-31G').",
    )
    keywords: Dict = Field(
        default={}, description="Additional keywords for simulation."
    )
    program: str = Field(
        default="psi4", description="The software to perform simulation."
    )
