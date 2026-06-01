import importlib.util
import json
import os
import shutil
from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Union, Optional, Any, List, Type
from chemgraph.schemas.atomsdata import AtomsData

from chemgraph.schemas.calculators.emt_calc import EMTCalc
from chemgraph.schemas.calculators.nwchem_calc import NWChemCalc
from chemgraph.schemas.calculators.orca_calc import OrcaCalc
from chemgraph.schemas.calculators.fairchem_calc import FAIRChemCalc
from chemgraph.schemas.calculators.mace_calc import MaceCalc
from chemgraph.schemas.calculators.tblite_calc import TBLiteCalc
from chemgraph.schemas.calculators.aimnet2_calc import AIMNET2Calc

# Gate optional calculators on whether their engine package is installed.
# Schema classes are always importable (internal to ChemGraph), so we must
# probe the underlying engine with importlib.util.find_spec().
# find_spec() can raise ModuleNotFoundError for sub-packages when the parent
# package is missing, so we guard with try/except.

def _engine_available(module_name: str) -> bool:
    """Return whether a Python calculator engine module is importable.

    Parameters
    ----------
    module_name : str
        Module name passed to ``importlib.util.find_spec``.

    Returns
    -------
    bool
        ``True`` when the module can be found.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _command_available(command_name: str, env_var_name: str) -> bool:
    """Return whether a calculator command is configured or on ``PATH``.

    Parameters
    ----------
    command_name : str
        Executable name to locate.
    env_var_name : str
        Environment variable that can provide the command.

    Returns
    -------
    bool
        ``True`` when the command is configured or discoverable.
    """
    return bool(os.environ.get(env_var_name)) or shutil.which(command_name) is not None


if not _engine_available("fairchem.core"):
    FAIRChemCalc = None

if not _engine_available("mace"):
    MaceCalc = None

if not _engine_available("tblite"):
    TBLiteCalc = None

if not _engine_available("aimnet2calc"):
    AIMNET2Calc = None

if not _command_available("nwchem", "ASE_NWCHEM_COMMAND"):
    NWChemCalc = None

if not _command_available("orca", "ASE_ORCA_COMMAND"):
    OrcaCalc = None


_CALCULATOR_ALIASES = {
    "xtb": "tbli",
    "gfn1xtb": "tbli",
    "gfn2xtb": "tbli",
}


def _calculator_key(name: str) -> str:
    """Return a normalized calculator lookup key.

    Parameters
    ----------
    name : str
        Calculator name or alias.

    Returns
    -------
    str
        Four-character normalized key used for calculator matching.
    """
    normalized = "".join(ch for ch in name.lower() if ch.isalnum())
    return _CALCULATOR_ALIASES.get(normalized, normalized[:4])


# Define all possible calculator classes
_all_calculator_classes: List[Optional[Type[BaseModel]]] = [
    FAIRChemCalc,
    MaceCalc,
    AIMNET2Calc,
    TBLiteCalc,
    EMTCalc,
    NWChemCalc,
    OrcaCalc,
]

# Filter out unavailable calculators
available_calculator_classes: List[Type[BaseModel]] = [
    calc for calc in _all_calculator_classes if calc
]

# Create a union for type hinting
CalculatorUnion = Union[tuple(available_calculator_classes)]

_calculator_name_items = [calc.__name__ for calc in available_calculator_classes]
_calculator_name_items = [
    "TBLiteCalc (aliases: xTB, GFN1-xTB, GFN2-xTB)"
    if name == "TBLiteCalc"
    else name
    for name in _calculator_name_items
]
_calculator_names = ", ".join(_calculator_name_items)

# Determine default calculator using only calculators detected as available.
default_calculator = available_calculator_classes[0]


def get_available_calculator_names() -> List[str]:
    """Return calculator class names detected as available in this environment."""
    return [calc.__name__ for calc in available_calculator_classes]


def get_default_calculator_name() -> str:
    """Return the default calculator class name selected for this environment."""
    return default_calculator.__name__


def get_calculator_selection_context() -> str:
    """Return prompt text describing available calculators and default choice."""
    return (
        "\n\nCalculator availability detected during ChemGraph initialization:\n"
        f"- Available ASE calculators: {_calculator_names}.\n"
        f"- Default calculator when the user does not specify one: "
        f"{default_calculator.__name__}.\n"
        "- When calling run_ase, choose only from the available calculators above. "
        "If the user requests an unavailable calculator, choose the default "
        "available calculator when that substitution is appropriate; otherwise "
        "ask for clarification or explain that the requested calculator is not "
        "available."
    )


class ASEInputSchema(BaseModel):
    """
    Schema for defining input parameters used in ASE-based molecular simulations.

    Attributes
    ----------
    input_structure_file : str
        Path to the input coordinate file (e.g., CIF, XYZ, POSCAR) containing
        the atomic structure for the simulation.
    output_results_file: str
        Path to a JSON file where simulation results will be saved.
    driver : str
        Specifies the type of simulation to perform. Options:
        - 'energy': Single-point electronic energy calculation.
        - 'opt': Geometry optimization.
        - 'vib': Vibrational frequency analysis.
        - 'ir': Infrared spectrum calculation.
        - 'thermo': Thermochemical property calculation (enthalpy, entropy, Gibbs free energy).
    optimizer : str
        Optimization algorithm for geometry optimization. Options:
        - 'bfgs', 'lbfgs', 'gpmin', 'fire', 'mdmin'.
    calculator : Union[FAIRChemCalc, MaceCalc, NWChemCalc, OrcaCalc, TBLiteCalc, EMTCalc, AIMNET2Calc]
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

    input_structure_file: str = Field(
        description="Path to the input coordinate file (e.g., CIF, XYZ, POSCAR) containing the atomic structure for the simulation."
    )
    output_results_file: str = Field(
        default="output.json",
        description="Path to a JSON file where simulation results will be saved.",
    )
    driver: str = Field(
        default=None,
        description="Specifies the type of simulation to run. Options: 'energy' for electronic energy calculations, 'dipole' for dipole moment calculation, 'opt' for geometry optimization, 'vib' for vibrational frequency analysis, 'ir' for calculating infrared spectrum, and 'thermo' for thermochemical properties (including enthalpy, entropy, and Gibbs free energy). Use 'thermo' when the query involves enthalpy, entropy, or Gibbs free energy calculations.",
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
        description="Maximum number of optimization steps. Internally 'vib', 'thermo' and 'ir' run geometry optimization before performing their respective calculations.",
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
        """Validate and coerce the calculator payload.

        Parameters
        ----------
        data : Any
            Raw ASE input payload before Pydantic validation.

        Returns
        -------
        Any
            Payload with calculator converted to an available calculator model.
        """
        if not isinstance(data, dict):
            return data

        calc = data.get("calculator")
        if calc is None:
            calc = default_calculator()
            data["calculator"] = calc

        available_calcs = {
            _calculator_key(c.__name__.removesuffix("Calc")): c
            for c in available_calculator_classes
        }
        available_calc_names = [c.__name__ for c in available_calculator_classes]

        if isinstance(calc, dict):
            calc_name = calc.get("calculator_type")
            if not calc_name:
                raise ValueError("Calculator dictionary must have a 'calculator_type' key.")

            calc_key = _calculator_key(calc_name)
            if calc_key not in available_calcs:
                raise ValueError(
                    f"Calculator {calc_name} is not an allowed or available calculator. "
                    f"Available calculators are: {available_calc_names}"
                )

            init_args = calc.copy()
            init_args.pop("calculator_type", None)
            data["calculator"] = available_calcs[calc_key](**init_args)
            return data

        elif hasattr(calc, "__class__"):
            calc_type_name = calc.__class__.__name__
            calc_key = _calculator_key(calc_type_name.removesuffix("Calc"))
            if calc_key not in available_calcs:
                raise ValueError(
                    f"Calculator {calc_type_name} is not an allowed or available calculator. "
                    f"Available calculators are: {available_calc_names}"
                )
        return data


class ASEOutputSchema(BaseModel):
    """
    Schema for defining outputs from ASE-based molecular simulations.
    """

    input_structure_file: str = Field(
        description=(
            "Path to the input coordinate file (e.g., CIF, XYZ, POSCAR) "
            "containing the initial atomic structure for the simulation."
        )
    )
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
    energy_unit: str = Field(default="eV", description="The unit of the energy reported.")
    dipole_value: List[Optional[float]] = Field(
        default=[None, None, None],
        description="The value of the dipole moment reported.",
    )
    dipole_unit: str = Field(
        default=" e * angstrom", description="The unit of the dipole moment reported."
    )
    vibrational_frequencies: dict = Field(
        default={},
        description="Vibrational frequencies (in cm-1) and energies (in eV).",
    )
    ir_data: dict = Field(
        default={},
        description="Infrared spectrum related data.",
    )
    thermochemistry: dict = Field(default={}, description="Thermochemistry data in eV.")
    success: bool = Field(
        default=False, description="Indicates if the simulation finished correctly."
    )
    error: str = Field(default="", description="Error captured during the simulation")
    wall_time: float = Field(
        default=None,
        description="Total wall time (in seconds) taken to complete the simulation.",
    )

    @field_validator("vibrational_frequencies", "ir_data", "thermochemistry", mode="before")
    @classmethod
    def _coerce_json_string_to_dict(cls, v: Any) -> dict:
        """Accept dict-like payloads serialized as JSON strings.

        Parameters
        ----------
        v : Any
            Raw field value.

        Returns
        -------
        dict
            Parsed dictionary or empty dictionary.
        """
        if v is None:
            return {}
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            text = v.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @field_validator("error", mode="before")
    @classmethod
    def _coerce_error_to_string(cls, v: Any) -> str:
        """Allow null/non-string error fields from intermediate tool payloads.

        Parameters
        ----------
        v : Any
            Raw error value.

        Returns
        -------
        str
            Normalized error string.
        """
        if v is None:
            return ""
        return v if isinstance(v, str) else str(v)
