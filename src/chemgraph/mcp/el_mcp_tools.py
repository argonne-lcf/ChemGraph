from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path
from typing import Literal

import pubchempy as pcp
from ase import Atoms
from mcp.server.fastmcp import FastMCP

from chemgraph.schemas.ase_input import ASEInputSchema, ASEOutputSchema
from chemgraph.tools.mcp_helper import (
    atoms_to_atomsdata,
    get_symmetry_number,
    is_linear_molecule,
    load_calculator,
)

try:
    import uuid

    from ensemble_launcher import EnsembleLauncher
    from ensemble_launcher.config import LauncherConfig, SystemConfig
    from ensemble_launcher.mcp import ELFastMCP

    EL_AVAILABLE = True
except Exception as e:
    print(f"Importing ensemble launcher failed with error {e}")
    EL_AVAILABLE = False


CHECKPOINT_DIR = os.path.join(os.getcwd(), f".ckpt_{uuid.uuid4()}")


def _resolve_path(path: str) -> str:
    """If CHEMGRAPH_LOG_DIR is set and path is relative, prepend it."""
    log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")
    if log_dir and not os.path.isabs(path):
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, path)
    return path


async def molecule_name_to_smiles(name: str) -> str:
    """
    Parameters
    ----------
    name : str
        The molecule/common name to resolve.

    Returns
    -------
    str
        Canonical SMILES string.

    Raises
    ------
    ValueError
        If no match is found on PubChem.
    """
    if not name or not str(name).strip():
        raise ValueError("Parameter 'name' must be a non-empty string.")

    comps = pcp.get_compounds(str(name).strip(), "name")
    if not comps:
        raise ValueError(f"No PubChem compound found for name: {name!r}")

    smiles = comps[0].canonical_smiles
    if not smiles:
        raise ValueError(f"PubChem returned an empty SMILES for {name!r}.")
    return smiles


async def smiles_to_coordinate_file(
    smiles: str,
    output_file: str = "molecule.xyz",
    seed: int = 2025,
    fmt: Literal["xyz"] = "xyz",
) -> dict:
    """Convert a SMILES string to a coordinate file.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.
    output_file : str, optional
        Path to save the output coordinate file (currently XYZ only).
    seed : int, optional
        Random seed for RDKit 3D structure generation, by default 2025.
    fmt : {"xyz"}, optional
        Output format. Only "xyz" supported for now.

    Returns
    -------
    str
        A single-line JSON string LLMs can parse, e.g.
        {
            "ok": true,
            "artifact": "coordinate_file",
            "format": "xyz",
            "path": "...",
            "smiles": "...",
            "natoms": 12
        }

    Raises
    ------
    ValueError
        If the SMILES string is invalid or if 3D structure generation fails.
    """
    from ase.io import write as ase_write
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Generate the molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    # Add hydrogens and optimize 3D structure
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise ValueError("Failed to generate 3D coordinates.")
    if AllChem.UFFOptimizeMolecule(mol) != 0:
        raise ValueError("Failed to optimize 3D geometry.")
    # Extract atomic information
    conf = mol.GetConformer()
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]

    # Create Atoms object
    atoms = Atoms(numbers=numbers, positions=positions)

    final_output_file = _resolve_path(output_file)
    ase_write(
        final_output_file,
        atoms,
    )

    # Return dict for LLM/tool chaining
    return {
        "ok": True,
        "artifact": "coordinate_file",
        "path": os.path.abspath(final_output_file),
        "smiles": smiles,
        "natoms": len(numbers),
    }


def extract_output_json(json_file: str) -> dict:
    """
    Load simulation results from a JSON file produced by run_ase.

    Parameters
    ----------
    json_file : str
        Path to the JSON file containing ASE simulation results.

    Returns
    -------
    dict
        Parsed results from the JSON file as a Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


async def run_ase(params: ASEInputSchema) -> dict:
    """Run ASE calculations using specified input parameters.

    Parameters
    ----------
    params : ASEInputSchema
        Input parameters for the ASE calculation

    Returns
    -------
    dict
        Output containing calculation status

    Raises
    ------
    ValueError
        If the calculator is not supported or if the calculation fails
    """
    import io
    from contextlib import redirect_stdout

    from ase.io import read
    from ase.optimize import BFGS, FIRE, LBFGS, GPMin, MDMin

    from chemgraph.schemas.atomsdata import AtomsData

    f = io.StringIO()
    with redirect_stdout(f):
        try:
            calculator = params.calculator.model_dump()
        except Exception as e:
            return f"Missing calculator parameter for the simulation. Raised exception: {str(e)}"

        # Calculate wall time.
        start_time = time.time()

        input_structure_file = params.input_structure_file
        output_results_file = _resolve_path(params.output_results_file)
        optimizer = params.optimizer
        fmax = params.fmax
        steps = params.steps
        driver = params.driver
        temperature = params.temperature
        pressure = params.pressure

        # # Validate that the input structure file exists
        if not os.path.isfile(input_structure_file):
            err = f"Input structure file {input_structure_file} does not exist."
            raise ValueError(err)

        # Validate the output results file (if provided)
        if not output_results_file.endswith(".json"):
            err = f"Output results file must end with '.json', got: {params.output_results_file}"
            raise ValueError(err)

        calc, system_info, calc_model = load_calculator(calculator)
        params.calculator = calc_model

        if calc is None:
            err = (
                f"Unsupported calculator: {calculator}. Available calculators are MACE"
                "(mace_mp, mace_off, mace_anicc), EMT, TBLite (GFN2-xTB, GFN1-xTB), NWChem and Orca"
            )
            raise ValueError(err)

        try:
            atoms = read(input_structure_file)
        except Exception as e:
            err = (
                f"Cannot read {input_structure_file} using ASE. Exception from ASE: {e}"
            )
            raise ValueError(err) from e

        atoms.info.update(system_info)
        atoms.calc = calc

        if driver == "energy" or driver == "dipole":
            energy = atoms.get_potential_energy()
            final_structure = atoms_to_atomsdata(atoms=atoms)

            dipole = [None, None, None]
            if driver == "dipole":
                # Catch exception if calculator doesn't have get_dipole_moment()
                try:
                    dipole = list(atoms.get_dipole_moment())
                except Exception:
                    pass

            end_time = time.time()
            wall_time = end_time - start_time

            simulation_output = ASEOutputSchema(
                input_structure_file=input_structure_file,
                converged=True,
                final_structure=final_structure,
                simulation_input=params,
                success=True,
                dipole_value=dipole,
                single_point_energy=energy,
                wall_time=wall_time,
            )
            with open(output_results_file, "w", encoding="utf-8") as wf:
                wf.write(simulation_output.model_dump_json(indent=4))
            return {
                "status": "success",
                "message": f"Simulation completed. Results saved to {os.path.abspath(output_results_file)}",
                "single_point_energy": energy,
                "unit": "eV",
            }

        OPTIMIZERS = {
            "bfgs": BFGS,
            "lbfgs": LBFGS,
            "gpmin": GPMin,
            "fire": FIRE,
            "mdmin": MDMin,
        }
        try:
            optimizer_class = OPTIMIZERS.get(optimizer.lower())
            if optimizer_class is None:
                raise ValueError(f"Unsupported optimizer: {optimizer_class}")

            # Do optimization only if number of atoms > 1 to avoid error.
            if len(atoms) > 1:
                dyn = optimizer_class(atoms)
                converged = dyn.run(fmax=fmax, steps=steps)
            else:
                converged = True

            single_point_energy = float(atoms.get_potential_energy())
            final_structure = AtomsData(
                numbers=atoms.numbers,
                positions=atoms.positions,
                cell=atoms.cell,
                pbc=atoms.pbc,
            )
            thermo_data = {}
            vib_data = {}
            ir_data = {}

            if driver in {"vib", "thermo", "ir"}:
                from ase import units
                from ase.vibrations import Vibrations

                vib_name = _resolve_path("vib")
                vib = Vibrations(atoms, name=vib_name)

                vib.clean()
                vib.run()

                vib_data = {
                    "energies": [],
                    "energy_unit": "meV",
                    "frequencies": [],
                    "frequency_unit": "cm-1",
                }

                energies = vib.get_energies()
                linear = is_linear_molecule(atomsdata=final_structure)

                for idx, e in enumerate(energies):
                    is_imag = abs(e.imag) > 1e-8
                    e_val = e.imag if is_imag else e.real
                    energy_meV = 1e3 * e_val
                    freq_cm1 = e_val / units.invcm
                    suffix = "i" if is_imag else ""
                    vib_data["energies"].append(f"{energy_meV}{suffix}")
                    vib_data["frequencies"].append(f"{freq_cm1}{suffix}")

                # Remove existing frequencies.txt and .traj files
                # Note: This glob might need adjustment if we are writing elsewhere,
                # but vib.clean() should handle its own files.
                # We will just remove any stray .traj files in the target dir if needed.
                # using the resolved name pattern
                for traj_file in glob.glob(f"{vib_name}.*.traj"):
                    os.remove(traj_file)

                # Write frequencies into frequencies.txt
                freq_file_path = _resolve_path("frequencies.csv")
                freq_file = Path(freq_file_path)
                if freq_file.exists():
                    freq_file.unlink()

                with freq_file.open("w", encoding="utf-8") as f:
                    for i, freq in enumerate(vib_data["frequencies"], start=0):
                        f.write(f"vib.{i}.traj,{freq}\n")

                # Write normal modes .traj files
                for i in range(len(energies)):
                    vib.write_mode(n=None, kT=units.kB * 300, nimages=30)

                if driver == "ir":
                    import matplotlib.pyplot as plt
                    from ase.vibrations import Infrared

                    ir_data["spectrum_frequencies"] = []
                    ir_data["spectrum_frequencies_units"] = "cm-1"

                    ir_data["spectrum_intensities"] = []
                    ir_data["spectrum_intensities_units"] = "D/Å^2 amu^-1"

                    ir_name = _resolve_path("ir")
                    ir = Infrared(atoms, name=ir_name)
                    ir.clean()
                    ir.run()

                    IR_SPECTRUM_START = 500  # Start of IR spectrum range
                    IR_SPECTRUM_END = 4000  # End of IR spectrum range
                    freq_intensity = ir.get_spectrum(
                        start=IR_SPECTRUM_START, end=IR_SPECTRUM_END
                    )
                    # Generate IR spectrum plot
                    fig, ax = plt.subplots()
                    ax.plot(freq_intensity[0], freq_intensity[1])
                    ax.set_xlabel("Frequency (cm⁻¹)")
                    ax.set_ylabel("Intensity (a.u.)")
                    ax.set_title("Infrared Spectrum")
                    ax.grid(True)
                    ir_plot_path = _resolve_path("ir_spectrum.png")
                    fig.savefig(ir_plot_path, format="png", dpi=300)

                    ir_data["IR Plot"] = f"Saved to {os.path.abspath(ir_plot_path)}"
                    ir_data["Normal mode data"] = (
                        f"Normal modes saved as individual .traj files in {os.path.abspath(ir_name)}"
                    )

                if driver == "thermo":
                    # Approximation for a single atom system.
                    if len(atoms) == 1:
                        thermo_data = {
                            "enthalpy": single_point_energy,
                            "entropy": 0.0,
                            "gibbs_free_energy": single_point_energy,
                            "unit": "eV",
                        }
                    else:
                        from ase.thermochemistry import IdealGasThermo

                        linear = is_linear_molecule(atomsdata=final_structure)
                        geometry = "linear" if linear else "nonlinear"
                        symmetrynumber = get_symmetry_number(atomsdata=final_structure)

                        thermo = IdealGasThermo(
                            vib_energies=energies,
                            potentialenergy=single_point_energy,
                            atoms=atoms,
                            geometry=geometry,
                            symmetrynumber=symmetrynumber,
                            spin=0,  # Only support spin=0
                        )
                        thermo_data = {
                            "enthalpy": float(
                                thermo.get_enthalpy(temperature=temperature)
                            ),
                            "entropy": float(
                                thermo.get_entropy(
                                    temperature=temperature, pressure=pressure
                                )
                            ),
                            "gibbs_free_energy": float(
                                thermo.get_gibbs_energy(
                                    temperature=temperature, pressure=pressure
                                )
                            ),
                            "unit": "eV",
                        }

            end_time = time.time()
            wall_time = end_time - start_time

            simulation_output = ASEOutputSchema(
                input_structure_file=input_structure_file,
                converged=converged,
                final_structure=final_structure,
                simulation_input=params,
                vibrational_frequencies=vib_data,
                thermochemistry=thermo_data,
                success=True,
                ir_data=ir_data,
                single_point_energy=single_point_energy,
                wall_time=wall_time,
            )

            with open(output_results_file, "w", encoding="utf-8") as wf:
                wf.write(simulation_output.model_dump_json(indent=4))

            # Return message based on driver. Keep the return output minimal.
            if driver == "opt":
                return {
                    "status": "success",
                    "message": f"Simulation completed. Results saved to {os.path.abspath(output_results_file)}",
                    "single_point_energy": single_point_energy,  # small payload for LLMs
                    "unit": "eV",
                }
            elif driver == "vib":
                return {
                    "status": "success",
                    "result": {
                        "vibrational_frequencies": vib_data,
                    },  # small payload for LLMs
                    "message": (
                        "Vibrational analysis completed; frequencies returned. "
                        f"Full results (structure, vibrations and metadata) saved to {os.path.abspath(output_results_file)}."
                    ),
                }
            elif driver == "thermo":
                return {
                    "status": "success",
                    "result": {
                        "thermochemistry": thermo_data
                    },  # small payload for LLMs
                    "message": (
                        "Thermochemistry computed and returned. "
                        f"Full results (structure, vibrations, thermochemistry and metadata) saved to {os.path.abspath(output_results_file)}"
                    ),
                }
            elif driver == "ir":
                return {
                    "status": "success",
                    "result": {
                        "vibrational_frequencies": vib_data
                    },  # small payload for LLMs,  # small payload for LLMs
                    "message": (
                        "Infrared computer and returned"
                        f"Full results (structure, vibrations, thermochemistry and metadata) saved to {os.path.abspath(output_results_file)}. "
                        f"IR plot Saved to {os.path.abspath(ir_plot_path)}. Normal modes saved as individual .traj files"
                    ),
                }

        except Exception as e:
            err = f"ASE simulation gave an exception:{e}"
            raise ValueError(err) from e


def start_el():
    sys_config = SystemConfig(
        name="local", ncpus=os.cpu_count(), cpus=list(range(os.cpu_count()))
    )
    launch_config = LauncherConfig(
        return_stdout=True,
        worker_logs=True,
        master_logs=True,
        cpu_binding_option="",
        cluster=True,
        checkpoint_dir=CHECKPOINT_DIR,
    )
    el = EnsembleLauncher(
        ensemble_file={}, system_config=sys_config, launcher_config=launch_config
    )
    el.start()
    time.sleep(20.0)
    return el


def create_mcp():
    if EL_AVAILABLE:
        print("Using EL fast mcp")
        mcp = ELFastMCP(
            checkpoint_dir=CHECKPOINT_DIR,
            name="ChemGraph General Tools",
            instructions="""
            You provide chemistry tools for converting molecule names to SMILES,
            building 3D coordinates, running ASE simulations (geometry optimization, thermochemistry, vibrational calculations), and reading results. "
            Each tool has its own description — follow those to decide when to use them.\n\n
            General guidance:\n
            • Keep outputs compact; large results are written to files.\n
            • Do not invent data. If a tool raises an error, report it as-is.\n
            • Use absolute file paths when returning artifacts.\n
            • Energies are in eV, vibrational frequencies in cm⁻¹, wall times in seconds.
        """,
        )
        mcp.init_client()
        return mcp
    else:
        raise ModuleNotFoundError("EL not found")


def register_tools(mcp: ELFastMCP):
    mcp.tool(
        molecule_name_to_smiles,
        name="molecule_name_to_smiles",
        description="Convert a molecule name to a canonical SMILES string using PubChem.",
    )
    mcp.tool(
        smiles_to_coordinate_file,
        name="smiles_to_coordinate_file",
        description="Convert a SMILES string to a coordinate file",
    )
    mcp.tool(
        extract_output_json,
        name="extract_output_json",
        description="Load simulation results from a JSON file produced by run_ase.",
    )
    mcp.tool(
        run_ase,
        name="run_ase",
        description="Run ASE calculations using specified input parameters.",
    )
