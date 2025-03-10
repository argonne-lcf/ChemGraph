from pydantic import BaseModel, Field
from typing import Optional, Union
from comp_chem_agent.models.atomsdata import AtomsData
from comp_chem_agent.models.calculators.mopac_calc import MopacCalc
from comp_chem_agent.models.calculators.psi4_calc import Psi4Calc
from qcelemental.models import OptimizationInput


class QCEngineInputSchema(BaseModel):
    atomsdata: AtomsData = Field(description="The atomsdata object to be used for the simulation.")
    driver: str = Field(
        default="energy",
        description="Type of quantum chemistry calculation. Supported values: 'energy' for single point calculation, 'gradient' for geometry optimization, 'hessian' for vibrational frequency calculations and 'properties' for properties calculations.",
    )
    calculator: Union[MopacCalc, Psi4Calc] = Field(
        default=None,
        description="The potential energy surface calculator to be used.",
    )
    program: str = Field(
        default=None, description="The software to perform simulation. Options are mopac and psi4."
    )


class QCEngineOutputSchema(BaseModel):
    converged: bool = Field(description="Whether the optimization converged.")
    final_structure: AtomsData = Field(description="The final structure after optimization.")
    simulation_input: Union[QCEngineInputSchema, dict] = Field(
        description="The input used for the simulation."
    )
    frequencies: Union[list[float], str] = Field(
        default="", description="Vibrational frequencies in cm-1."
    )
    thermochemistry: Optional[str] = Field(
        default="", description="Summary of thermochemistry calculations."
    )
