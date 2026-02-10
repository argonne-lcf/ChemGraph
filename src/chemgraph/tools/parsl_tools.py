from pathlib import Path
import time
import glob
import os

from pydantic import BaseModel, Field


from chemgraph.tools.mcp_helper import (
    get_symmetry_number,
    is_linear_molecule,
    atoms_to_atomsdata,
)


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


def run_mace_core(mace_input_schema: mace_input_schema):
    """Run a single MACE calculations using specified input parameters.

    Parameters
    ----------
    mace_input_schema : mace_input_schema
        Input parameters for the MACE calculation
    """
    from ase.io import read
    from ase.io import write as ase_write
    from ase.optimize import BFGS, LBFGS, GPMin, FIRE, MDMin

    from mace.calculators import mace_mp

    # Input validations
    if not os.path.isfile(mace_input_schema.input_structure_file):
        err = f"Input structure file {mace_input_schema.input_structure_file} does not exist."
        raise ValueError(err)
    # Validate the output results file (if provided)
    if not mace_input_schema.output_result_file.endswith(".json"):
        err = f"Output results file must end with '.json', got: {mace_input_schema.output_result_file}"
        raise ValueError(err)

    # Validate the input structure with ASE io
    try:
        atoms = read(mace_input_schema.input_structure_file)
    except Exception as e:
        err = f"Cannot read {mace_input_schema.input_structure_file} using ASE. Exception from ASE: {e}"
        raise ValueError(err)

    # Start time
    start_time = time.time()

    calc = mace_mp(model=mace_input_schema.model, device=mace_input_schema.device)
    atoms.calc = calc

    # Create parent directory for output file
    output_path = Path(mace_input_schema.output_result_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mace_input_schema.driver == "energy":
        try:
            energy = atoms.get_potential_energy()
        except Exception as e:
            err = f"Error encountered with the simulation. Exception: {e}"
            raise ValueError(err)

        final_structure_file = os.path.abspath(mace_input_schema.input_structure_file)

        end_time = time.time()
        wall_time = end_time - start_time
        simulation_output = mace_output_schema(
            final_structure_file=final_structure_file,
            success=True,
            energy=energy,
            wall_time=wall_time,
            output_result_file=os.path.abspath(mace_input_schema.output_result_file),
            model=mace_input_schema.model,
            driver=mace_input_schema.driver,
            device=mace_input_schema.device,
        )
        with open(mace_input_schema.output_result_file, "w") as wf:
            wf.write(simulation_output.model_dump_json(indent=4))
        return {
            "status": "success",
            "message": f"Simulation completed. Results saved to {os.path.abspath(mace_input_schema.output_result_file)}",
            "single_point_energy": energy,
            "unit": "eV",
        }
    else:
        OPTIMIZERS = {
            "bfgs": BFGS,
            "lbfgs": LBFGS,
            "gpmin": GPMin,
            "fire": FIRE,
            "mdmin": MDMin,
        }
        try:
            optimizer_class = OPTIMIZERS.get(mace_input_schema.optimizer.lower())
            if optimizer_class is None:
                raise ValueError(f"Unsupported optimizer: {optimizer_class}")

            # Do optimization only if number of atoms > 1 to avoid error.
            if len(atoms) > 1:
                dyn = optimizer_class(atoms)
                dyn.run(
                    fmax=mace_input_schema.fmax,
                    steps=mace_input_schema.steps,
                )

            # Get the single-point energy of the optimized structure
            opt_energy = float(atoms.get_potential_energy())

            # Write the optimized structure
            input_path = mace_input_schema.input_structure_file
            root, ext = os.path.splitext(input_path)
            opt_path = root + "_opt" + ext
            ase_write(opt_path, atoms)
            final_structure_file = os.path.abspath(opt_path)

            # Initiate thermo and vibrational data
            thermo_data = {}
            vib_data = {}

            if mace_input_schema.driver in {"vib", "thermo"}:
                from ase.vibrations import Vibrations
                from ase import units

                vib = Vibrations(atoms)

                vib.clean()
                vib.run()

                vib_data = {
                    "energies": [],
                    "energy_unit": "meV",
                    "frequencies": [],
                    "frequency_unit": "cm-1",
                }

                energies = vib.get_energies()

                for idx, e in enumerate(energies):
                    is_imag = abs(e.imag) > 1e-8
                    e_val = e.imag if is_imag else e.real
                    energy_meV = 1e3 * e_val
                    freq_cm1 = e_val / units.invcm
                    suffix = "i" if is_imag else ""
                    vib_data["energies"].append(f"{energy_meV}{suffix}")
                    vib_data["frequencies"].append(f"{freq_cm1}{suffix}")

                # Remove existing frequencies.txt and .traj files
                for traj_file in glob.glob("*.traj"):
                    os.remove(traj_file)

                # Write frequencies into frequencies.txt
                freq_file = Path("frequencies.csv")
                if freq_file.exists():
                    freq_file.unlink()

                with freq_file.open("w") as f:
                    for i, freq in enumerate(vib_data["frequencies"], start=0):
                        f.write(f"vib.{i}.traj,{freq}\n")

                # Write normal modes .traj files
                for i in range(len(energies)):
                    vib.write_mode(n=None, kT=units.kB * 300, nimages=30)

                if mace_input_schema.driver == "thermo":
                    # Approximation for a single atom system.
                    if len(atoms) == 1:
                        thermo_data = {
                            "enthalpy": opt_energy,
                            "entropy": 0.0,
                            "gibbs_free_energy": opt_energy,
                            "unit": "eV",
                        }
                    else:
                        from ase.thermochemistry import IdealGasThermo

                        final_structure = atoms_to_atomsdata(atoms)
                        linear = is_linear_molecule(final_structure)
                        geometry = "linear" if linear else "nonlinear"
                        symmetrynumber = get_symmetry_number(final_structure)

                        thermo = IdealGasThermo(
                            vib_energies=energies,
                            potentialenergy=opt_energy,
                            atoms=atoms,
                            geometry=geometry,
                            symmetrynumber=symmetrynumber,
                            spin=0,
                        )
                        thermo_data = {
                            "enthalpy": float(
                                thermo.get_enthalpy(
                                    temperature=mace_input_schema.temperature
                                )
                            ),
                            "entropy": float(
                                thermo.get_entropy(
                                    temperature=mace_input_schema.temperature,
                                    pressure=mace_input_schema.pressure,
                                )
                            ),
                            "gibbs_free_energy": float(
                                thermo.get_gibbs_energy(
                                    temperature=mace_input_schema.temperature,
                                    pressure=mace_input_schema.pressure,
                                )
                            ),
                            "unit": "eV",
                        }

            end_time = time.time()
            wall_time = end_time - start_time

            simulation_output = mace_output_schema(
                final_structure_file=final_structure_file,
                success=True,
                energy=opt_energy,
                wall_time=wall_time,
                output_result_file=os.path.abspath(
                    mace_input_schema.output_result_file
                ),
                model=mace_input_schema.model,
                driver=mace_input_schema.driver,
                device=mace_input_schema.device,
                vibrational_frequencies=vib_data,
                thermochemistry=thermo_data,
            )
            with open(mace_input_schema.output_result_file, "w") as wf:
                wf.write(simulation_output.model_dump_json(indent=4))

            # Return message based on driver. Keep the return output minimal.
            if mace_input_schema.driver == "opt":
                return {
                    "status": "success",
                    "message": f"Simulation completed. Results saved to {mace_input_schema.output_result_file}",
                    "single_point_energy": opt_energy,  # small payload for LLMs
                    "unit": "eV",
                }
            elif mace_input_schema.driver == "vib":
                return {
                    "status": "success",
                    "result": {
                        "vibrational_frequencies": vib_data,
                    },  # small payload for LLMs
                    "message": (
                        "Vibrational analysis completed; frequencies returned. "
                        f"Full results (structure, vibrations and metadata) saved to {mace_input_schema.output_result_file}."
                    ),
                }
            elif mace_input_schema.driver == "thermo":
                return {
                    "status": "success",
                    "result": {"thermochemistry": thermo_data},
                    "message": (
                        "Thermochemistry computed and returned. "
                        f"Full results (structure, vibrations, thermochemistry and metadata) saved to {mace_input_schema.output_result_file}"
                    ),
                }
        except Exception as e:
            err = f"ASE simulation gave an exception:{e}"
            return {
                "status": "failure",
                "error_type": type(e).__name__,
                "message": str(e),
            }

        return True
