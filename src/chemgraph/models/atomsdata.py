from pydantic import BaseModel, Field
from typing import List, Optional, Union


class AtomsData(BaseModel):
    """AtomsData object inherited from Pydantic BaseModel. Used to store atomic data (from ASE Atoms object or QCElemental Molecule object) that cannot be parsed via LLM Schema."""

    numbers: List[int] = Field(..., description="Atomic numbers")
    positions: List[List[float]] = Field(..., description="Atomic positions")
    cell: Optional[Union[List[List[float]], None]] = Field(
        default=None, description="Cell vectors or None"
    )
    pbc: Optional[Union[List[bool], None]] = Field(
        default=None, description="Periodic boundary conditions or None"
    )
