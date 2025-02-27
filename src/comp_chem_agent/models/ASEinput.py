from pydantic import BaseModel, Field
from typing import List, Optional, Union
from comp_chem_agent.models.atomsdata import AtomsData
from comp_chem_agent.models.calculators.tblite_calc import TBLiteCalc
from comp_chem_agent.models.calculators.emt_calc import EMTCalc
from comp_chem_agent.models.calculators.mace_calc import MaceCalc


class ASESimulationInput(BaseModel):
    atomsdata: AtomsData = Field(
        description="The atomsdata object to be used for the simulation."
    )
    optimizer: str = Field(
        default="BFGS",
        description="The optimization algorithm used for geometry optimization. BFGS (Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton method for finding local minima.",
    )
    calculator: Union[str, TBLiteCalc] = Field(
        default="mace_mp",
        description="The potential energy surface calculator to be used. MACE-MP is a machine learning interatomic potential based on equivariant message passing.",
    )
    fmax: float = Field(
        default=0.01,
        description="The convergence criterion for forces (in eV/Å). Optimization stops when all force components are smaller than this value.",
    )
    steps: int = Field(
        default=10,
        description="Maximum number of optimization steps. The optimization will terminate if this number is reached, even if forces haven't converged to fmax.",
    )


class ASESimulationOutput(BaseModel):
    converged: bool = Field(description="Whether the optimization converged.")
    final_structure: AtomsData = Field(
        description="The final structure after optimization."
    )
    simulation_input: ASESimulationInput = Field(
        description="The input used for the simulation."
    )
    gradients: Optional[List[float]] = Field(
        default=[], description="The gradients at each 10th step of the optimization."
    )
    frequencies: Optional[str] = Field(
        default="", description="Summary of vibrational frequency calculations"
    )


class ASEAtomicInput(BaseModel):
    atomsdata: AtomsData = Field(
        description="The atomsdata object to be used for the simulation."
    )
    driver: str = Field(
        default="opt",
        description="Type of simulation to run. Support: 'opt' for geometry optimization and 'vib' for vibrational frequency calculations.",
    )
    optimizer: str = Field(
        default="BFGS",
        description="The optimization algorithm used for geometry optimization. BFGS (Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton method for finding local minima.",
    )
    calculator: Union[str, TBLiteCalc] = Field(
        default="mace_mp",
        description="The potential energy surface calculator to be used. MACE-MP is a machine learning interatomic potential based on equivariant message passing.",
    )
    fmax: float = Field(
        default=0.01,
        description="The convergence criterion for forces (in eV/Å). Optimization stops when all force components are smaller than this value.",
    )
    steps: int = Field(
        default=1000,
        description="Maximum number of optimization steps. The optimization will terminate if this number is reached, even if forces haven't converged to fmax.",
    )


class ASESchema(BaseModel):
    atomsdata: AtomsData = Field(
        description="The atomsdata object to be used for the simulation."
    )
    driver: str = Field(
        default="opt",
        description="Type of simulation to run. Support: 'opt' for geometry optimization and 'vib' for vibrational frequency calculations.",
    )
    optimizer: str = Field(
        default="bfgs",
        description="The optimization algorithm used for geometry optimization. Options are 'bfgs', 'lbfgs', 'gpmin', 'fire', 'mdmin'",
    )
    calculator: Union[TBLiteCalc, MaceCalc, EMTCalc] = Field(
        default=None, description="The potential energy surface calculator to be used. "
    )
    fmax: float = Field(
        default=0.01,
        description="The convergence criterion for forces (in eV/Å). Optimization stops when all force components are smaller than this value.",
    )
    steps: int = Field(
        default=1000,
        description="Maximum number of optimization steps. The optimization will terminate if this number is reached, even if forces haven't converged to fmax.",
    )
