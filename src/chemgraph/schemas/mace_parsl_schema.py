"""Pydantic schemas for MACE Parsl simulations.

These schemas define the API contract for MACE single and ensemble
calculations dispatched via Parsl.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class mace_input_schema(BaseModel):
    input_structure_file: str = Field(
        description="Path to the input coordinate file (e.g., CIF, XYZ, POSCAR) containing the atomic structure for the simulation."
    )
    output_result_file: str = Field(
        default="output.json",
        description="Path to a JSON file where simulation results will be saved.",
    )
    driver: str = Field(
        default=None,
        description="Specifies the type of simulation to run. Options: 'energy' for single-point energy calculations, 'opt' for geometry optimization, 'vib' for vibrational frequency analysis, and 'thermo' for thermochemical properties (including enthalpy, entropy, and Gibbs free energy).",
    )
    model: str = Field(
        default="medium-mpa-0",
        description="Path to the model. Default is medium-mpa-0."
        "Options are 'small', 'medium', 'large', 'small-0b', 'medium-0b', 'small-0b2', 'medium-0b2','large-0b2', 'medium-0b3', 'medium-mpa-0', 'medium-omat-0', 'mace-matpes-pbe-0', 'mace-matpes-r2scan-0'",
    )
    device: str = Field(
        default="cpu",
        description="Device to run the MACE calculation on. Options are cpu, cuda or xpu.",
    )
    temperature: float = Field(
        default=298.15,
        description="Temperature for thermo property calculations in Kelvin (K).",
    )
    pressure: float = Field(
        default=101325.0,
        description="Pressure for thermo property calculations in Pascal (Pa).",
    )
    fmax: float = Field(
        default=0.01,
        description="The convergence criterion for forces (in eV/Å). Optimization stops when all force components are smaller than this value.",
    )
    steps: int = Field(
        default=1000,
        description="Maximum number of optimization steps. The optimization will terminate if this number is reached, even if forces haven't converged to fmax.",
    )
    optimizer: str = Field(
        default="lbfgs",
        description="The optimization algorithm used for geometry optimization. Options are 'bfgs', 'lbfgs', 'gpmin', 'fire', 'mdmin'",
    )


class mace_input_schema_ensemble(BaseModel):
    input_structure_directory: str = Field(
        description="Path to a folder of input structures containing the atomic structure for the simulations."
    )
    output_result_file: str = Field(
        default="output.json",
        description="Path to a JSON file where simulation results will be saved.",
    )
    driver: str = Field(
        default=None,
        description="Specifies the type of simulation to run. Options: 'energy' for single-point energy calculations, 'opt' for geometry optimization, 'vib' for vibrational frequency analysis, and 'thermo' for thermochemical properties (including enthalpy, entropy, and Gibbs free energy).",
    )
    model: str = Field(
        default="medium-mpa-0",
        description="Path to the model. Default is medium-mpa-0."
        "Options are 'small', 'medium', 'large', 'small-0b', 'medium-0b', 'small-0b2', 'medium-0b2','large-0b2', 'medium-0b3', 'medium-mpa-0', 'medium-omat-0', 'mace-matpes-pbe-0', 'mace-matpes-r2scan-0'",
    )
    device: str = Field(
        default="cpu",
        description="Device to run the MACE calculation on. Options are cpu, cuda or xpu.",
    )
    temperature: float = Field(
        default=298.15,
        description="Temperature for thermo property calculations in Kelvin (K).",
    )
    pressure: float = Field(
        default=101325.0,
        description="Pressure for thermo property calculations in Pascal (Pa).",
    )
    fmax: float = Field(
        default=0.01,
        description="The convergence criterion for forces (in eV/Å). Optimization stops when all force components are smaller than this value.",
    )
    steps: int = Field(
        default=1000,
        description="Maximum number of optimization steps. The optimization will terminate if this number is reached, even if forces haven't converged to fmax.",
    )
    optimizer: str = Field(
        default="lbfgs",
        description="The optimization algorithm used for geometry optimization. Options are 'bfgs', 'lbfgs', 'gpmin', 'fire', 'mdmin'",
    )


class mace_output_schema(BaseModel):
    final_structure_file: str = Field(
        description="Path to the final coordinate file (e.g., CIF, XYZ, POSCAR) containing the atomic structure for the simulation."
    )
    output_result_file: str = Field(
        description="Path to a JSON file where simulation results is saved.",
    )
    model: str = Field(
        default=None, description="Path to the model. Default is medium-mpa-0."
    )
    device: str = Field(
        default="cpu",
        description="Device to run the MACE calculation on. Options are cpu, cuda or xpu.",
    )
    temperature: float = Field(
        default=298.15,
        description="Temperature for thermo property calculations in Kelvin (K).",
    )
    pressure: float = Field(
        default=101325.0,
        description="Pressure for thermo property calculations in Pascal (Pa).",
    )
    fmax: float = Field(
        default=0.01,
        description="The convergence criterion for forces (in eV/Å). Optimization stops when all force components are smaller than this value.",
    )
    steps: int = Field(
        default=1000,
        description="Maximum number of optimization steps. The optimization will terminate if this number is reached, even if forces haven't converged to fmax.",
    )
    energy: float = Field(
        description="The electronic energy of the system in eV",
    )
    success: bool = Field(
        description="Status of the simulation",
    )
    vibrational_frequencies: dict = Field(
        default={},
        description="Vibrational frequencies (in cm-1) and energies (in eV).",
    )
    thermochemistry: dict = Field(
        default={},
        description="Thermochemistry data in eV.",
    )
    error: str = Field(
        default="",
        description="Error captured during the simulation",
    )
    wall_time: float = Field(
        default=None,
        description="Total wall time (in seconds) taken to complete the simulation.",
    )
