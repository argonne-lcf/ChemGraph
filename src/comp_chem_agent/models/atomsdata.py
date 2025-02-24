from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Annotated, Union

class AtomsData(BaseModel):
    numbers: List[int] = Field(..., description="Atomic numbers")
    positions: List[List[float]] = Field(..., description="Atomic positions")
    cell: Optional[Union[List[List[float]], None]] = Field(
        default=None, 
        description="Cell vectors or None"
    )
    pbc: Optional[Union[List[bool], None]] = Field(
        default=None, 
        description="Periodic boundary conditions or None"
    )

