from pathlib import Path
import time
import json
import numpy as np
from typing import Any, Dict

from langchain_core.tools import tool
from chemgraph.models.atomsdata import AtomsData
from chemgraph.models.ase_input import (
    ASEInputSchema,
    ASEOutputSchema,
)


@tool
def extract_output_json(json_file: str) -> Dict[str, Any]:
    """
    Load simulation results from a JSON file produced by run_ase.

    Parameters
    ----------
    json_file : str
        Path to the JSON file containing ASE simulation results.

    Returns
    -------
    Dict[str, Any]
        Parsed results from the JSON file as a Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def extract_ase_atoms_from_tool_result(tool_result: dict):
    """Extract ASE atoms data from tool result dictionary.

    Parameters
    ----------
    tool_result : dict
        Dictionary containing tool result data

    Returns
    -------
    tuple
        (atomic_numbers, positions) or (None, None) if extraction fails
    """
    for keyset in (
        {"numbers", "positions"},
        {"atomic_numbers", "positions"},
    ):
        if keyset.issubset(tool_result.keys()):
            return tool_result[keyset.pop()], tool_result["positions"]

    if "atoms" in tool_result:
        atoms_data = tool_result["atoms"]
        if {"numbers", "positions"}.issubset(atoms_data):
            return atoms_data["numbers"], atoms_data["positions"]

    return None, None


def atoms_to_atomsdata(atoms):
    """Convert ASE Atoms object to AtomsData.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object

    Returns
    -------
    AtomsData
        ChemGraph AtomsData object
    """
    return AtomsData(
        numbers=atoms.numbers.tolist(),
        positions=atoms.positions.tolist(),
        cell=atoms.cell.tolist(),
        pbc=atoms.pbc.tolist(),
    )


def atomsdata_to_atoms(atomsdata: AtomsData):
    """Convert AtomsData to ASE Atoms object.

    Parameters
    ----------
    atomsdata : AtomsData
        ChemGraph AtomsData object

    Returns
    -------
    ase.Atoms
        ASE Atoms object
    """
    from ase import Atoms

    return Atoms(
        numbers=atomsdata.numbers,
        positions=atomsdata.positions,
        cell=atomsdata.cell,
        pbc=atomsdata.pbc,
    )


@tool
def file_to_atomsdata(fname: str) -> AtomsData:
    """Convert a structure file to AtomsData format using ASE.

    Parameters
    ----------
    fname : str
        Path to the input structure file (supports various formats like xyz, pdb, cif, etc.)

    Returns
    -------
    AtomsData
        Object containing the atomic structure information

    Raises
    ------
    FileNotFoundError
        If the input file doesn't exist
    ValueError
        If the file format is not supported or file is corrupted
    """
    from ase.io import read

    try:
        atoms = read(fname)
        # Create AtomsData object from ASE Atoms object
        atoms_data = AtomsData(
            numbers=atoms.numbers.tolist(),
            positions=atoms.positions.tolist(),
            cell=atoms.cell.tolist(),
            pbc=atoms.pbc.tolist(),
        )
        return atoms_data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {fname}")
    except Exception as e:
        raise ValueError(f"Failed to read structure file: {str(e)}")


@tool
def save_atomsdata_to_file(atomsdata: AtomsData, fname: str = "output.xyz") -> str:
    """Save an AtomsData object to a file using ASE.

    Parameters
    ----------
    atomsdata : AtomsData
        AtomsData object to save
    fname : str, optional
        Path to the output file, by default "output.xyz"

    Returns
    -------
    str
        Success message or error message

    Raises
    ------
    ValueError
        If saving the file fails
    """
    from ase.io import write
    from ase import Atoms

    try:
        atoms = Atoms(
            numbers=atomsdata.numbers,
            positions=atomsdata.positions,
            cell=atomsdata.cell,
            pbc=atomsdata.pbc,
        )
        write(fname, atoms)
        return f"Successfully saved atomsdata to {fname}"
    except Exception as e:
        raise ValueError(f"Failed to save atomsdata to file: {str(e)}")


@tool
def get_symmetry_number(atomsdata: AtomsData) -> int:
    """Get the rotational symmetry number of a molecule using Pymatgen.

    Parameters
    ----------
    atomsdata : AtomsData
        AtomsData object containing the molecular structure

    Returns
    -------
    int
        Rotational symmetry number of the molecule
    """
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    from ase import Atoms
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = Atoms(
        numbers=atomsdata.numbers,
        positions=atomsdata.positions,
        cell=atomsdata.cell,
        pbc=atomsdata.pbc,
    )

    aaa = AseAtomsAdaptor()
    molecule = aaa.get_molecule(atoms)
    pga = PointGroupAnalyzer(molecule)
    symmetrynumber = pga.get_rotational_symmetry_number()

    return symmetrynumber


@tool
def is_linear_molecule(atomsdata: AtomsData, tol=1e-3) -> bool:
    """Determine if a molecule is linear or not.

    Parameters
    ----------
    atomsdata : AtomsData
        AtomsData object containing the molecular structure
    tol : float, optional
        Tolerance to check for linear molecule, by default 1e-3

    Returns
    -------
    bool
        True if the molecule is linear, False otherwise
    """
    coords = np.array(atomsdata.positions)
    # Center the coordinates.
    centered = coords - np.mean(coords, axis=0)
    # Singular value decomposition.
    U, s, Vt = np.linalg.svd(centered)
    # For a linear molecule, only one singular value is significantly nonzero.
    if s[0] == 0:
        return False  # degenerate case (all atoms at one point)
    return (s[1] / s[0]) < tol


def load_calculator(calculator: dict) -> tuple[object, dict, dict]:
    """Load an ASE calculator based on the provided configuration.

    Parameters
    ----------
    calculator : dict
        Dictionary containing calculator configuration parameters

    Returns
    -------
    object
        ASE calculator instance

    Raises
    ------
    ValueError
        If the calculator type is not supported
    """
    calc_type = calculator["calculator_type"].lower()

    if "emt" in calc_type:
        from chemgraph.models.calculators.emt_calc import EMTCalc

        calc = EMTCalc(**calculator)
    elif "tblite" in calc_type:
        from chemgraph.models.calculators.tblite_calc import TBLiteCalc

        calc = TBLiteCalc(**calculator)
    elif "orca" in calc_type:
        from chemgraph.models.calculators.orca_calc import OrcaCalc

        calc = OrcaCalc(**calculator)

    elif "nwchem" in calc_type:
        from chemgraph.models.calculators.nwchem_calc import NWChemCalc

        calc = NWChemCalc(**calculator)

    elif "fairchem" in calc_type:
        from chemgraph.models.calculators.fairchem_calc import FAIRChemCalc

        calc = FAIRChemCalc(**calculator)

    elif "mace" in calc_type:
        from chemgraph.models.calculators.mace_calc import MaceCalc

        calc = MaceCalc(**calculator)

    elif "aimnet2" in calc_type:
        from chemgraph.models.calculators.aimnet2_calc import AIMNET2Calc

        calc = AIMNET2Calc(**calculator)

    else:
        raise ValueError(
            f"Unsupported calculator: {calculator}. Available calculators are EMT, TBLite (GFN2-xTB, GFN1-xTB), Orca and FAIRChem or MACE or AIMNET2."
        )
    # Extract additional args like spin/charge if the model defines it
    extra_info = {}
    if hasattr(calc, "get_atoms_properties"):
        extra_info = calc.get_atoms_properties()

    return calc.get_calculator(), extra_info, calc


@tool
def run_ase(params: ASEInputSchema) -> ASEOutputSchema:
    """Run ASE calculations using specified input parameters.

    Parameters
    ----------
    params : ASEInputSchema
        Input parameters for the ASE calculation

    Returns
    -------
    ASEOutputSchema
        Output containing calculation results and status

    Raises
    ------
    ValueError
        If the calculator is not supported or if the calculation fails
    """
    import os
    from ase.io import read
    from ase.optimize import BFGS, LBFGS, GPMin, FIRE, MDMin

    try:
        calculator = params.calculator.model_dump()
    except Exception as e:
        return f"Missing calculator parameter for the simulation. Raised exception: {str(e)}"

    # Calculate wall time.
    start_time = time.time()

    input_structure_file = params.input_structure_file
    output_results_file = params.output_results_file
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

    if calc is None:
        err = f"Unsupported calculator: {calculator}. Available calculators are MACE (mace_mp, mace_off, mace_anicc), EMT, TBLite (GFN2-xTB, GFN1-xTB), NWChem and Orca"
        raise ValueError(err)

    try:
        atoms = read(input_structure_file)
    except Exception as e:
        err = f"Cannot read {input_structure_file} using ASE. Exception from ASE: {e}"
        raise ValueError(err)

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
            except Exception as e:
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
        with open(output_results_file, "w") as wf:
            wf.write(simulation_output.model_dump_json(indent=4))
        return {
            "status": "success",
            "message": f"Simulation completed. Results saved to {output_results_file}",
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
            linear = is_linear_molecule.invoke({"atomsdata": final_structure})

            for idx, e in enumerate(energies):
                is_imag = abs(e.imag) > 1e-8
                e_val = e.imag if is_imag else e.real
                energy_meV = 1e3 * e_val
                freq_cm1 = e_val / units.invcm
                suffix = "i" if is_imag else ""
                vib_data["energies"].append(f"{energy_meV}{suffix}")
                vib_data["frequencies"].append(f"{freq_cm1}{suffix}")

            # Remove existing frequencies.txt and .traj files
            import os, glob

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

            if driver == "ir":
                from ase.vibrations import Infrared
                import matplotlib.pyplot as plt

                ir_data["spectrum_frequencies"] = []
                ir_data["spectrum_frequencies_units"] = "cm-1"

                ir_data["spectrum_intensities"] = []
                ir_data["spectrum_intensities_units"] = "D/Å^2 amu^-1"

                ir = Infrared(atoms)
                ir.clean()
                ir.run()

                IR_SPECTRUM_START = 500  # Start of IR spectrum range
                IR_SPECTRUM_END = 4000  # End of IR spectrum range
                freq_intensity = ir.get_spectrum(
                    start=IR_SPECTRUM_START, end=IR_SPECTRUM_END
                )
                """
                for f, inten in zip(freq_intensity[0], freq_intensity[1]):
                    ir_data["spectrum_frequencies"].append(f"{f}")
                    ir_data["spectrum_intensities"].append(f"{inten}")
                """
                # Generate IR spectrum plot
                fig, ax = plt.subplots()
                ax.plot(freq_intensity[0], freq_intensity[1])
                ax.set_xlabel("Frequency (cm⁻¹)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.set_title("Infrared Spectrum")
                ax.grid(True)
                fig.savefig("ir_spectrum.png", format="png", dpi=300)

                ir_data["IR Plot"] = "Saved to ir_spectrum.png"
                ir_data["Normal mode data"] = (
                    "Normal modes saved as individual .traj files"
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

                    linear = is_linear_molecule.invoke({"atomsdata": final_structure})
                    geometry = "linear" if linear else "nonlinear"
                    symmetrynumber = get_symmetry_number.invoke(
                        {"atomsdata": final_structure}
                    )

                    thermo = IdealGasThermo(
                        vib_energies=energies,
                        potentialenergy=single_point_energy,
                        atoms=atoms,
                        geometry=geometry,
                        symmetrynumber=symmetrynumber,
                        spin=0,  # Only support spin=0
                    )
                    thermo_data = {
                        "enthalpy": float(thermo.get_enthalpy(temperature=temperature)),
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
        with open(output_results_file, "w") as wf:
            wf.write(simulation_output.model_dump_json(indent=4))

        # Return message based on driver. Keep the return output minimal.
        if driver == "opt":
            return {
                "status": "success",
                "message": f"Simulation completed. Results saved to {output_results_file}",
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
                    f"Full results (structure, vibrations and metadata) saved to {output_results_file}."
                ),
            }
        elif driver == "thermo":
            return {
                "status": "success",
                "result": {"thermochemistry": thermo_data},  # small payload for LLMs
                "message": (
                    "Thermochemistry computed and returned. "
                    f"Full results (structure, vibrations, thermochemistry and metadata) saved to {output_results_file}"
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
                    f"Full results (structure, vibrations, thermochemistry and metadata) saved to {output_results_file}. "
                    "IR plot Saved to ir_spectrum.png. Normal modes saved as individual .traj files"
                ),
            }

    except Exception as e:
        err = f"ASE simulation gave an exception:{e}"
        return {
            "status": "failure",
            "error_type": type(e).__name__,
            "message": str(e),
        }


def create_ase_atoms(atomic_numbers, positions):
    """Create an ASE Atoms object from atomic numbers and positions.

    Parameters
    ----------
    atomic_numbers : list or array
        List of atomic numbers
    positions : list or array
        List of atomic positions (3D coordinates)

    Returns
    -------
    ase.Atoms
        ASE Atoms object
    """
    from ase import Atoms

    try:
        atoms = Atoms(numbers=atomic_numbers, positions=positions)
        return atoms
    except Exception as e:
        print(f"Error creating ASE Atoms object: {e}")
        return None


def create_xyz_string(atomic_numbers, positions):
    """Create an XYZ format string from atomic numbers and positions.

    Parameters
    ----------
    atomic_numbers : list or array
        List of atomic numbers
    positions : list or array
        List of atomic positions (3D coordinates)

    Returns
    -------
    str
        XYZ format string
    """
    from ase import Atoms

    try:
        atoms = Atoms(numbers=atomic_numbers, positions=positions)

        # Create XYZ string manually
        xyz_lines = [str(len(atoms))]
        xyz_lines.append("Generated by ChemGraph")

        for i, (symbol, pos) in enumerate(
            zip(atoms.get_chemical_symbols(), atoms.positions)
        ):
            xyz_lines.append(
                f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}"
            )

        return "\n".join(xyz_lines)
    except Exception as e:
        print(f"Error creating XYZ string: {e}")
        return None
