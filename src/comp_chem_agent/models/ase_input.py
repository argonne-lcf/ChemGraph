from pydantic import BaseModel, Field
from typing import Union
from comp_chem_agent.models.atomsdata import AtomsData
from comp_chem_agent.models.calculators.tblite_calc import TBLiteCalc
from comp_chem_agent.models.calculators.emt_calc import EMTCalc
from comp_chem_agent.models.calculators.mace_calc import MaceCalc
from comp_chem_agent.models.calculators.nwchem_calc import NWChemCalc
from typing import Optional


class ASEInputSchema(BaseModel):
    """
    Schema for defining input parameters used in ASE-based molecular simulations.

    Attributes
    ----------
    atomsdata : AtomsData
        The atomic structure and associated metadata for the simulation.
    driver : str
        Specifies the type of simulation to perform. Options:
        - 'energy': Single-point electronic energy calculation.
        - 'opt': Geometry optimization.
        - 'vib': Vibrational frequency analysis.
        - 'thermo': Thermochemical property calculation (enthalpy, entropy, Gibbs free energy).
    optimizer : str
        Optimization algorithm for geometry optimization. Options:
        - 'bfgs', 'lbfgs', 'gpmin', 'fire', 'mdmin'.
    calculator : Union[MaceCalc, EMTCalc, NWChemCalc, TBLiteCalc]
        ASE-compatible calculator used for the simulation. Supported types include:
        - MACE, EMT, NWChem, and TBLite.
    fmax : float
        Force convergence criterion in eV/Å. Optimization stops when all force components fall below this threshold.
    steps : int
        Maximum number of steps for geometry optimization.
    temperature : Optional[float]
        Temperature in Kelvin, required for thermochemical calculations (e.g., when using 'thermo' as the driver).
    pressure : float
        Pressure in Pascal (Pa), used in thermochemistry calculations (default is 1 atm).
    """

    atomsdata: AtomsData = Field(description="The atomsdata object to be used for the simulation.")
    driver: str = Field(
        default=None,
        description="Specifies the type of simulation to run. Options: 'energy' for electronic energy calculations, 'opt' for geometry optimization, 'vib' for vibrational frequency analysis, and 'thermo' for thermochemical properties (including enthalpy, entropy, and Gibbs free energy). Use 'thermo' when the query involves enthalpy, entropy, or Gibbs free energy calculations.",
    )
    optimizer: str = Field(
        default="bfgs",
        description="The optimization algorithm used for geometry optimization. Options are 'bfgs', 'lbfgs', 'gpmin', 'fire', 'mdmin'",
    )
    calculator: Union[MaceCalc, EMTCalc, NWChemCalc, TBLiteCalc] = Field(
        default=None,
        description="The ASE calculator to be used. Support TBLite, Mace, EMT and NWChem. ",
    )
    fmax: float = Field(
        default=0.01,
        description="The convergence criterion for forces (in eV/Å). Optimization stops when all force components are smaller than this value.",
    )
    steps: int = Field(
        default=1000,
        description="Maximum number of optimization steps. The optimization will terminate if this number is reached, even if forces haven't converged to fmax.",
    )
    temperature: Optional[float] = Field(
        default=None, description="Temperature for thermochemistry calculations in Kelvin (K)."
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
    energy_unit: str = Field(default="eV", description="The unit of the energy reported.")
    vibrational_frequencies: dict = Field(
        default={}, description="Vibrational frequencies (in cm-1) and energies (in eV)."
    )
    thermochemistry: dict = Field(default={}, description="Thermochemistry data in eV.")
    success: bool = Field(
        default=False, description="Indicates if the simulation finished correctly."
    )
    error: str = Field(default="", description="Error captured during the simulation")
