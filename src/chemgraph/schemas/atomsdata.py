from pydantic import BaseModel, Field
from typing import List, Optional 


class AtomsData(BaseModel):
    """AtomsData object inherited from Pydantic BaseModel. Used to store atomic data (from ASE Atoms object or QCElemental Molecule object) that cannot be parsed via LLM Schema."""
    
    # Optional is equivalent to Union[..., None], but more concise. 
    numbers: List[int] = Field(..., description="Atomic numbers")
    positions: List[List[float]] = Field(..., description="Atomic positions")
    cell: Optional[List[List[float]]] = Field(
        default=None, description="Cell vectors or None"
    )
    pbc: Optional[List[bool]] = Field(
        default=None, description="Periodic boundary conditions or None"
    )
