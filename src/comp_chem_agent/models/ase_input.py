from pydantic import BaseModel, Field
from typing import Union
from comp_chem_agent.models.atomsdata import AtomsData
from comp_chem_agent.models.calculators.tblite_calc import TBLiteCalc
from comp_chem_agent.models.calculators.emt_calc import EMTCalc
from comp_chem_agent.models.calculators.mace_calc import MaceCalc
from comp_chem_agent.models.calculators.nwchem_calc import NWChemCalc


class ASEInputSchema(BaseModel):
    atomsdata: AtomsData = Field(description="The atomsdata object to be used for the simulation.")
    driver: str = Field(
        default=None,
        description="Type of simulation to run. Support: 'energy' for single point energy calculation, 'opt' for geometry optimization, 'vib' for vibrational frequency calculations and 'thermo' for thermochemistry calculations.",
    )
    optimizer: str = Field(
        default="bfgs",
        description="The optimization algorithm used for geometry optimization. Options are 'bfgs', 'lbfgs', 'gpmin', 'fire', 'mdmin'",
    )
    calculator: Union[MaceCalc, EMTCalc, NWChemCalc, TBLiteCalc] = Field(
        default=None,
        description="The ASE calculator to be used. Support XTB, Mace, EMT and NWChem. ",
    )
    fmax: float = Field(
        default=0.01,
        description="The convergence criterion for forces (in eV/Å). Optimization stops when all force components are smaller than this value.",
    )
    steps: int = Field(
        default=1000,
        description="Maximum number of optimization steps. The optimization will terminate if this number is reached, even if forces haven't converged to fmax.",
    )
    temperature: float = Field(
        default=298.15, description="Temperature for thermochemistry calculations in Kelvin (K)."
    )
    pressure: float = Field(
        default=101325.0, description="Pressure for thermochemistry calculations in Pascal (Pa)."
    )


class ASEOutputSchema(BaseModel):
    converged: bool = Field(
        default=False, description="Indicates if the optimization successfully converged."
    )
    final_structure: AtomsData = Field(description="Final structure.")
    simulation_input: ASEInputSchema = Field(
        description="Simulation input for Atomic Simulation Environment."
    )
    single_point_energy: float = Field(
        default=None, description="Single-point energy/Potential energy"
    )
    vibrational_frequencies: dict = Field(
        default={}, description="Vibrational frequencies (in cm-1) and energies (in eV)."
    )
    thermochemistry: dict = Field(default={}, description="Thermochemistry data in eV.")
    success: bool = Field(
        default=False, description="Indicates if the simulation finished correctly."
    )
    error: str = Field(default="", description="Error captured during the simulation")
