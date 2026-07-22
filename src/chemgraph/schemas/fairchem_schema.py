"""Pydantic schemas for FairChem/UMA simulations.

These schemas define the API contract for FairChem single and ensemble
calculations. They mirror :mod:`chemgraph.schemas.mace_parsl_schema` but
carry the FairChem/UMA-specific fields (model name, task, charge,
multiplicity, ...) drawn from
:class:`chemgraph.schemas.calculators.fairchem_calc.FAIRChemCalc`.
"""

from __future__ import annotations

from typing import Optional

import torch
from pydantic import BaseModel, Field


class fairchem_input_schema(BaseModel):
    input_structure_file: str = Field(
        description="Path to the input coordinate file (e.g., CIF, XYZ, POSCAR) containing the atomic structure for the simulation."
    )
    output_result_file: str = Field(
        default="output.json",
        description="Path to a JSON file where simulation results will be saved.",
    )
    driver: str | None = Field(
        default=None,
        description="Specifies the type of simulation to run. Options: 'energy' for single-point energy calculations, 'opt' for geometry optimization, 'vib' for vibrational frequency analysis, and 'thermo' for thermochemical properties (including enthalpy, entropy, and Gibbs free energy).",
    )
    model_name: str = Field(
        default="uma-s-1p1",
        description="FairChem/UMA inference model name (NOT the calculator type). "
        "Options: 'uma-s-1p1' and 'uma-m-1'. Default is 'uma-s-1p1'.",
    )
    task_name: Optional[str] = Field(
        default=None,
        description="Prediction task for the model head. Options: 'omol', 'omat', 'oc20', 'odac', or 'omc'.",
    )
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run the FairChem calculation on. Options are 'cpu' or 'cuda'.",
    )
    charge: Optional[int] = Field(
        default=0,
        description="Total system charge. Default is 0.",
    )
    multiplicity: Optional[int] = Field(
        default=1,
        description="Spin multiplicity (2S+1) of the system. Default is 1 (singlet).",
        ge=1,
    )
    inference_settings: str = Field(
        default="default",
        description="Settings for inference. Can be 'default' or 'turbo'.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for inference reproducibility.",
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


class fairchem_input_schema_ensemble(BaseModel):
    input_structure_directory: str = Field(
        default="",
        description="Path to a local folder of input structures. Required unless remote_structure_directory is provided.",
    )
    remote_structure_directory: str | None = Field(
        default=None,
        description=(
            "Path to pre-staged structure files on the remote HPC filesystem. "
            "When provided, workers read structures directly from this path "
            "instead of using inline structure embedding. Use the "
            "transfer_files tool to stage files first, then pass the "
            "remote directory here."
        ),
    )
    output_result_file: str = Field(
        default="output.json",
        description="Path to a JSON file where simulation results will be saved.",
    )
    driver: str | None = Field(
        default=None,
        description="Specifies the type of simulation to run. Options: 'energy' for single-point energy calculations, 'opt' for geometry optimization, 'vib' for vibrational frequency analysis, and 'thermo' for thermochemical properties (including enthalpy, entropy, and Gibbs free energy).",
    )
    model_name: str = Field(
        default="uma-s-1p1",
        description="FairChem/UMA inference model name (NOT the calculator type). "
        "Options: 'uma-s-1p1' and 'uma-m-1'. Default is 'uma-s-1p1'.",
    )
    task_name: Optional[str] = Field(
        default=None,
        description="Prediction task for the model head. Options: 'omol', 'omat', 'oc20', 'odac', or 'omc'.",
    )
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run the FairChem calculation on. Options are 'cpu' or 'cuda'.",
    )
    charge: Optional[int] = Field(
        default=0,
        description="Total system charge. Default is 0.",
    )
    multiplicity: Optional[int] = Field(
        default=1,
        description="Spin multiplicity (2S+1) of the system. Default is 1 (singlet).",
        ge=1,
    )
    inference_settings: str = Field(
        default="default",
        description="Settings for inference. Can be 'default' or 'turbo'.",
    )
    seed: int = Field(
        default=42,
        description="Random seed for inference reproducibility.",
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


class fairchem_output_schema(BaseModel):
    final_structure_file: str = Field(
        description="Path to the final coordinate file (e.g., CIF, XYZ, POSCAR) containing the atomic structure for the simulation."
    )
    output_result_file: str = Field(
        description="Path to a JSON file where simulation results is saved.",
    )
    model_name: str | None = Field(
        default=None,
        description="FairChem/UMA inference model name. Default is 'uma-s-1p1'.",
    )
    device: str = Field(
        default="cpu",
        description="Device to run the FairChem calculation on. Options are 'cpu' or 'cuda'.",
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
    wall_time: float | None = Field(
        default=None,
        description="Total wall time (in seconds) taken to complete the simulation.",
    )
