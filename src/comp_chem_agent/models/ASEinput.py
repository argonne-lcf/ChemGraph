from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import json

class ASESimulationInput(BaseModel):
    optimizer: str = Field(
        default="BFGS",
        description="The optimization algorithm used for geometry optimization. BFGS (Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton method for finding local minima."
    )
    calculator: str = Field(
        default="mace_mp",
        description="The potential energy surface calculator to be used. MACE-MP is a machine learning interatomic potential based on equivariant message passing."
    )
    fmax: float = Field(
        default=0.01,
        description="The convergence criterion for forces (in eV/Å). Optimization stops when all force components are smaller than this value."
    )
    steps: int = Field(
        default=100000,
        description="Maximum number of optimization steps. The optimization will terminate if this number is reached, even if forces haven't converged to fmax."
    )
