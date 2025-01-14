from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import json
from comp_chem_agent.models.atomsdata import AtomsData

class ASESimulationInput(BaseModel):
    atomsdata: AtomsData = Field(
        description="The atomsdata object to be used for the simulation."
    )
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
        default=10,
        description="Maximum number of optimization steps. The optimization will terminate if this number is reached, even if forces haven't converged to fmax."
    )

class ASESimulationOutput(BaseModel):
    converged: bool = Field(
        description="Whether the optimization converged."
    )
    final_structure: AtomsData = Field(
        description="The final structure after optimization."
    )
    simulation_input: ASESimulationInput = Field(
        description="The input used for the simulation."
    )
    gradients: List[float] = Field(
        description="The gradients at each 10th step of the optimization."
    )
