from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Annotated

class AtomsData(BaseModel):
    numbers: List[int] = Field(..., description="Atomic numbers")
    positions: List[List[float]] = Field(..., description="Atomic positions")
    cell: Optional[List[List[float]]] = Field(default=[[0,0,0], [0,0,0], [0,0,0]], description="Cell vectors")
    pbc: List[bool] = Field(default=[False, False, False], description="Periodic boundary conditions")

