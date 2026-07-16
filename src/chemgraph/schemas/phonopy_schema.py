import json
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Any
from chemgraph.schemas.atomsdata import AtomsData
from chemgraph.schemas.ase_input import CalculatorUnion, default_calculator, _calculator_key, available_calculator_classes

class PhonopyInputSchema(BaseModel):
    """
    Schema for defining input parameters used in Phonopy phonon calculations.
    """
    input_structure_file: str = Field(
        description="Path to the input coordinate file (e.g., CIF, XYZ, POSCAR) containing the relaxed atomic structure."
    )
    output_results_file: str = Field(
        default="phonopy_results.json",
        description="Path to a JSON file where phonon simulation results will be saved.",
    )
    supercell_matrix: Optional[List[int]] = Field(
        default=None,
        description="A list of 3 integers representing the diagonal of the supercell matrix (e.g., [2, 2, 2]). If not provided, it will be determined automatically to ensure dimensions are at least 10 Å.",
    )
    is_2d: bool = Field(
        default=False,
        description="Set to true if the material is 2D. This forces the supercell matrix in the Z-direction to be 1.",
    )
    minimize_structure: bool = Field(
        default=True,
        description="Whether to minimize (relax) the atomic structure before performing the phonon calculation.",
    )
    relaxation_mode: str = Field(
        default="full",
        description="Relaxation mode to use if minimize_structure is True. Options are 'full' (relax both atoms and cell), 'atoms' (relax only atomic positions), or 'cell' (relax only cell parameters).",
    )

    @field_validator("relaxation_mode")
    @classmethod
    def _validate_relaxation_mode(cls, v: str) -> str:
        v = v.lower()
        if v not in ("full", "atoms", "cell"):
            raise ValueError("relaxation_mode must be one of 'full', 'atoms', or 'cell'")
        return v
    calculator: CalculatorUnion = Field(
        default_factory=default_calculator,
        description="The ASE calculator to be used for force evaluations.",
    )
    mesh: List[int] = Field(
        default=[10, 10, 10],
        description="Sampling mesh for reciprocal space (e.g., [10, 10, 10]).",
    )
    calculate_dos: bool = Field(
        default=True,
        description="Whether to calculate and plot the Total Density of States (DOS).",
    )
    calculate_thermal_properties: bool = Field(
        default=False,
        description="Whether to calculate and plot Thermal Properties (Free energy, Entropy, Heat Capacity). Set to True only if explicitly requested.",
    )
    calculate_band_structure: bool = Field(
        default=True,
        description="Whether to calculate and plot the phonon band structure (dispersion curve).",
    )
    band_paths: Optional[List[List[List[float]]]] = Field(
        default=None,
        description="A list of q-point paths in reciprocal space (e.g., [[[0,0,0], [0.5,0,0], [0.333,0.333,0], [0,0,0]]]). If None but calculate_band_structure is True, it will automatically determine the standard high-symmetry paths using seekpath.",
    )
    band_labels: Optional[List[str]] = Field(
        default=None,
        description="List of labels for the q-points in band_paths (e.g., ['Gamma', 'M', 'K', 'Gamma']). Total number of labels should match the total number of unique q-points in the path.",
    )
    band_npoints: int = Field(
        default=51,
        description="Number of q-points to sample along the paths between high-symmetry points for the band structure.",
    )
    supercell_target_length: float = Field(
        default=10.0,
        description="Target minimum length (in Å) for the auto-generated supercell matrix dimensions.",
    )
    save_vasp_files: bool = Field(
        default=True,
        description="Whether to save the VASP-format FORCE_CONSTANTS and POSCAR-* files.",
    )
    symprec: float = Field(
        default=1e-5,
        description="Symmetry tolerance used in Phonopy.",
    )
    dft_phonon_file: Optional[str] = Field(
        default=None,
        description="Path to a DFT phonon data file (e.g., band.dat or results.dat) to compare with the calculated phonon dispersion.",
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

        available_calcs = {
            _calculator_key(c.__name__.removesuffix("Calc")): c
            for c in available_calculator_classes
        }

        if isinstance(calc, dict):
            calc_name = calc.get("calculator_type")
            if not calc_name:
                raise ValueError("Calculator dictionary must have a 'calculator_type' key.")

            calc_key = _calculator_key(calc_name)
            if calc_key not in available_calcs:
                raise ValueError(f"Calculator {calc_name} is not an allowed or available calculator.")

            init_args = calc.copy()
            init_args.pop("calculator_type", None)
            data["calculator"] = available_calcs[calc_key](**init_args)
            return data

        elif hasattr(calc, "__class__"):
            calc_type_name = calc.__class__.__name__
            calc_key = _calculator_key(calc_type_name.removesuffix("Calc"))
            if calc_key not in available_calcs:
                raise ValueError(f"Calculator {calc_type_name} is not an allowed or available calculator.")
        return data


class PhonopyOutputSchema(BaseModel):
    """
    Schema for defining outputs from Phonopy phonon simulations.
    """
    input_structure_file: str = Field(description="Path to the input coordinate file.")
    simulation_input: PhonopyInputSchema = Field(description="Input parameters used.")
    success: bool = Field(default=False, description="Indicates if the simulation finished correctly.")
    error: str = Field(default="", description="Error captured during the simulation.")
    supercell_matrix_used: List[int] = Field(default=[], description="The actual supercell matrix used.")
    thermal_properties_plot: Optional[str] = Field(default=None, description="Path to the saved thermal properties plot.")
    dos_plot: Optional[str] = Field(default=None, description="Path to the saved DOS plot.")
    band_structure_plot: Optional[str] = Field(default=None, description="Path to the saved band structure plot.")
    band_yaml: Optional[str] = Field(default=None, description="Path to the saved band.yaml file containing frequencies and k-points.")
    band_dat: Optional[str] = Field(default=None, description="Path to the saved band.dat file containing frequencies and kspace data in gnuplot format.")
    calculation_info_file: Optional[str] = Field(default=None, description="Path to the saved Phonon_Calculation_Info.md report file in English.")
    phonopy_yaml: Optional[str] = Field(default=None, description="Path to the generated phonopy.yaml file containing force constants.")
    force_constants_file: Optional[str] = Field(default=None, description="Path to the saved FORCE_CONSTANTS file.")
    poscar_files: Optional[List[str]] = Field(default=None, description="List of paths to the saved POSCAR-* supercell files.")
    minimized_structure_file: Optional[str] = Field(default=None, description="Path to the saved minimized structure file.")
    minimization_log_file: Optional[str] = Field(default=None, description="Path to the saved structural minimization log file.")
    wall_time: float = Field(default=None, description="Total wall time (in seconds) taken to complete the simulation.")


    @field_validator("error", mode="before")
    @classmethod
    def _coerce_error_to_string(cls, v: Any) -> str:
        if v is None:
            return ""
        return v if isinstance(v, str) else str(v)
