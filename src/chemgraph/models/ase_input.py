from pydantic import BaseModel, Field, model_validator
from typing import Union, Optional, Any, List, Type
from chemgraph.models.atomsdata import AtomsData
from chemgraph.models.calculators.tblite_calc import TBLiteCalc
from chemgraph.models.calculators.emt_calc import EMTCalc
from chemgraph.models.calculators.nwchem_calc import NWChemCalc
from chemgraph.models.calculators.orca_calc import OrcaCalc


# Attempt to import optional calculators
try:
    from chemgraph.models.calculators.fairchem_calc import FAIRChemCalc
except ImportError:
    FAIRChemCalc = None

try:
    from chemgraph.models.calculators.mace_calc import MaceCalc
except ImportError:
    MaceCalc = None


# Define all possible calculator classes
_all_calculator_classes: List[Optional[Type[BaseModel]]] = [
    FAIRChemCalc,
    MaceCalc,
    NWChemCalc,
    TBLiteCalc,
    OrcaCalc,
    EMTCalc,
]

# Filter out unavailable calculators
available_calculator_classes: List[Type[BaseModel]] = [
    calc for calc in _all_calculator_classes if calc
]

# Create a union for type hinting
CalculatorUnion = Union[tuple(available_calculator_classes)]

# Determine default calculator and names string
if FAIRChemCalc:
    default_calculator = FAIRChemCalc
elif MaceCalc:
    default_calculator = MaceCalc
else:
    default_calculator = NWChemCalc

_calculator_names = ", ".join([calc.__name__ for calc in available_calculator_classes])


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
    calculator : Union[FAIRChemCalc, MaceCalc, NWChemCalc, OrcaCalc, TBLiteCalc, EMTCalc]
        ASE-compatible calculator used for the simulation. Supported types are determined
        by installed packages and may include:
        - FAIRChem, MACE, NWChem, Orca, TBLite and EMT. The order determines the priority of the calculators.
        - Use MACE or FAIRChem if the calculator is not specified.
    fmax : float
        Force convergence criterion in eV/Å. Optimization stops when all force components fall below this threshold.
    steps : int
        Maximum number of steps for geometry optimization.
    temperature : Optional[float]
        Temperature in Kelvin, required for thermochemical calculations (e.g., when using 'thermo' as the driver).
    pressure : float
        Pressure in Pascal (Pa), used in thermochemistry calculations (default is 1 atm).
    """

    atomsdata: AtomsData = Field(
        description="The atomsdata object to be used for the simulation."
    )
    driver: str = Field(
        default=None,
        description="Specifies the type of simulation to run. Options: 'energy' for electronic energy calculations, 'opt' for geometry optimization, 'vib' for vibrational frequency analysis, and 'thermo' for thermochemical properties (including enthalpy, entropy, and Gibbs free energy). Use 'thermo' when the query involves enthalpy, entropy, or Gibbs free energy calculations.",
    )
    optimizer: str = Field(
        default="bfgs",
        description="The optimization algorithm used for geometry optimization. Options are 'bfgs', 'lbfgs', 'gpmin', 'fire', 'mdmin'",
    )
    calculator: CalculatorUnion = Field(
        default_factory=default_calculator,
        description=f"The ASE calculator to be used. Support {_calculator_names}. Use {default_calculator.__name__} if not specified.",
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
        default=None,
        description="Temperature for thermochemistry calculations in Kelvin (K).",
    )
    pressure: float = Field(
        default=101325.0,
        description="Pressure for thermochemistry calculations in Pascal (Pa).",
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_calculator_type(cls, data: Any):
        if not isinstance(data, dict):
            return data

        calc = data.get("calculator")
        if calc is None:
            calc = default_calculator()
            data["calculator"] = calc

        available_calcs = [c.__name__[:4].lower() for c in available_calculator_classes]

        if isinstance(calc, dict):
            calc_name = calc.get("calculator_type")
            if not calc_name:
                raise ValueError(
                    "Calculator dictionary must have a 'calculator_type' key."
                )

            if calc_name[:4].lower() not in available_calcs:
                raise ValueError(
                    f"Calculator {calc_name} is not an allowed or available calculator. "
                    f"Available calculators are: {available_calcs}"
                )

            for c in available_calculator_classes:
                if c.__name__[:4].lower() == calc_name[:4].lower():
                    init_args = calc.copy()
                    init_args.pop("calculator_type", None)
                    data["calculator"] = c(**init_args)
                    return data

        elif hasattr(calc, "__class__"):
            calc_type_name = calc.__class__.__name__
            if calc_type_name[:4].lower() not in available_calcs:
                raise ValueError(
                    f"Calculator {calc_type_name} is not an allowed or available calculator. "
                    f"Available calculators are: {available_calcs}"
                )
        return data


class ASEOutputSchema(BaseModel):
    converged: bool = Field(
        default=False,
        description="Indicates if the optimization successfully converged.",
    )
    final_structure: AtomsData = Field(description="Final structure.")
    simulation_input: ASEInputSchema = Field(
        description="Simulation input for Atomic Simulation Environment."
    )
    single_point_energy: float = Field(
        default=None, description="Single-point energy/Potential energy"
    )
    energy_unit: str = Field(
        default="eV", description="The unit of the energy reported."
    )
    vibrational_frequencies: dict = Field(
        default={},
        description="Vibrational frequencies (in cm-1) and energies (in eV).",
    )
    thermochemistry: dict = Field(default={}, description="Thermochemistry data in eV.")
    success: bool = Field(
        default=False, description="Indicates if the simulation finished correctly."
    )
    error: str = Field(default="", description="Error captured during the simulation")
