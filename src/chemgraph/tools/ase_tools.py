import numpy as np
import time
from langchain_core.tools import tool
from chemgraph.models.atomsdata import AtomsData
from chemgraph.models.ase_input import (
    ASEInputSchema,
    ASEOutputSchema,
)


def create_ase_atoms(atomic_numbers, positions, cell=None, pbc=None):
    """Create ASE Atoms object from atomic numbers and positions.

    Parameters
    ----------
    atomic_numbers : list
        List of atomic numbers
    positions : list
        List of atomic positions
    cell : list, optional
        List of cell vectors
    pbc : list, optional
        List of periodic boundary conditions
    Returns
    -------
    ase.Atoms or None
        ASE Atoms object or None if creation fails
    """
    try:
        from ase import Atoms

        return Atoms(numbers=atomic_numbers, positions=positions, cell=cell, pbc=pbc)
    except Exception as exc:
        # Use print for general use, but allow caller to handle error display
        print(f"Error creating ASE Atoms object: {exc}")
        return None


def create_xyz_string(atomic_numbers, positions):
    """Create XYZ format string from atomic numbers and positions.

    Parameters
    ----------
    atomic_numbers : list
        List of atomic numbers
    positions : list
        List of atomic positions

    Returns
    -------
    str or None
        XYZ format string or None if creation fails
    """
    atoms = create_ase_atoms(atomic_numbers, positions)
    if atoms is None:
        return None

    lines = [str(len(atoms)), "ChemGraph Structure"]
    for atom in atoms:
        x, y, z = atom.position
        lines.append(f"{atom.symbol} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(lines)


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


# -----------------------------------------------------------------------------
# Original tool functions
# -----------------------------------------------------------------------------


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
    try:
        calculator = params.calculator.model_dump()
    except Exception as e:
        return f"Missing calculator parameter for the simulation. Raised exception: {str(e)}"

    # Calculate wall time.
    start_time = time.time()

    atomsdata = params.atomsdata
    optimizer = params.optimizer
    fmax = params.fmax
    steps = params.steps
    driver = params.driver
    pressure = params.pressure

    calc, system_info, calc_model = load_calculator(calculator)
    params.calculator = calc_model

    if calc is None:
        e = f"Unsupported calculator: {calculator}. Available calculators are MACE (mace_mp, mace_off, mace_anicc), EMT, TBLite (GFN2-xTB, GFN1-xTB), NWChem and Orca"
        raise ValueError(e)
        simulation_output = ASEOutputSchema(
            converged=False,
            final_structure=atomsdata,
            simulation_input=params,
            error=str(e),
            success=False,
        )
        return simulation_output

    from ase import Atoms
    from ase.optimize import BFGS, LBFGS, GPMin, FIRE, MDMin

    atoms = Atoms(
        numbers=atomsdata.numbers,
        positions=atomsdata.positions,
        cell=atomsdata.cell,
        pbc=atomsdata.pbc,
    )
    atoms.info.update(system_info)
    atoms.calc = calc

    if driver == "energy" or driver == "dipole":
        energy = atoms.get_potential_energy()

        dipole = [None,None,None]
        if  driver == "dipole":
            dipole = list(atoms.get_dipole_moment())

        end_time = time.time()
        wall_time = end_time - start_time
        simulation_output = ASEOutputSchema(
            converged=True,
            final_structure=atomsdata,
            simulation_input=params,
            success=True,
            dipole_value=dipole,
            single_point_energy=energy,
            wall_time=wall_time,
        )
        return simulation_output

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
            raise ValueError(f"Unsupported optimizer: {params['optimizer']}")
        # Do optimization only if number of atoms > 1 to avoid error.
        if len(atoms) > 1:
            dyn = optimizer_class(atoms)
            converged = dyn.run(fmax=fmax, steps=steps)
        else:
            converged = True

        final_coords = atoms.positions
        single_point_energy = atoms.get_potential_energy()
        final_structure = AtomsData(
            numbers=atoms.numbers,
            positions=final_coords,
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
        vib_data = {}
        thermo_data = {}
        # Infrared
        ir_data = {}

        if driver == "vib" or driver == "thermo" or driver == "ir":
            temperature = params.temperature

            from ase.vibrations import Vibrations
            from ase import units

            vib_data["energies"] = []
            vib_data["energy_unit"] = "meV"

            vib_data["frequencies"] = []
            vib_data["frequency_unit"] = "cm-1"

            vib = Vibrations(atoms)
            vib.clean()
            vib.run()

            energies = vib.get_energies()
            linear = is_linear_molecule.invoke({"atomsdata": final_structure})

            for idx, e in enumerate(energies):
                if abs(e.imag) > 1e-8:
                    c = "i"
                    e_val = e.imag
                else:
                    c = ""
                    e_val = e.real

                energy_meV = 1e3 * e_val
                freq_cm1 = e_val / units.invcm

                vib_data["energies"].append(f"{energy_meV}{c}")
                vib_data["frequencies"].append(f"{freq_cm1}{c}")

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

                IR_SPECTRUM_START = 800  # Start of IR spectrum range
                IR_SPECTRUM_END = 4000  # End of IR spectrum range
                freq_intensity = ir.get_spectrum(start=IR_SPECTRUM_START, end=IR_SPECTRUM_END)

                for f, inten in zip(freq_intensity[0], freq_intensity[1]):
                    ir_data["spectrum_frequencies"].append(f"{f}")
                    ir_data["spectrum_intensities"].append(f"{inten}")

                # Generate IR spectrum plot
                fig, ax = plt.subplots()
                ax.plot(freq_intensity[0], freq_intensity[1])
                ax.set_xlabel("Frequency (cm⁻¹)")
                ax.set_ylabel("Intensity (a.u.)")
                ax.set_title("Infrared Spectrum")
                ax.grid(True)
                fig.savefig("ir_spectrum.png", format="png", dpi=300)

            if driver == "thermo":
                # Approximation for system with a single atom.
                if len(atoms) == 1:
                    thermo_data["enthalpy"] = atoms.get_potential_energy()
                    thermo_data["entropy"] = 0
                    thermo_data["gibbs_free_energy"] = float(atoms.get_potential_energy())
                    thermo_data["unit"] = "eV"
                else:
                    from ase.thermochemistry import IdealGasThermo

                    potentialenergy = atoms.get_potential_energy()
                    vib_energies = vib.get_energies()

                    linear = is_linear_molecule.invoke({"atomsdata": final_structure})
                    symmetrynumber = get_symmetry_number.invoke({"atomsdata": final_structure})

                    if linear:
                        geometry = "linear"
                    else:
                        geometry = "nonlinear"
                    thermo = IdealGasThermo(
                        vib_energies=vib_energies,
                        potentialenergy=potentialenergy,
                        atoms=atoms,
                        geometry=geometry,
                        symmetrynumber=symmetrynumber,
                        spin=0,  # Only support spin=0
                    )

                    thermo_data["enthalpy"] = float(
                        thermo.get_enthalpy(
                            temperature=temperature,
                        )
                    )
                    thermo_data["entropy"] = float(
                        thermo.get_entropy(temperature=temperature, pressure=pressure)
                    )
                    thermo_data["gibbs_free_energy"] = float(
                        thermo.get_gibbs_energy(temperature=temperature, pressure=pressure)
                    )
                    thermo_data["unit"] = "eV"

        end_time = time.time()
        wall_time = end_time - start_time

        simulation_output = ASEOutputSchema(
            converged=converged,
            final_structure=final_structure,
            simulation_input=params,
            vibrational_frequencies=vib_data,
            thermochemistry=thermo_data,
            success=True,
            single_point_energy=single_point_energy,
            wall_time=wall_time,
        )
        return simulation_output

    except Exception as e:
        end_time = time.time()
        wall_time = end_time - start_time

        simulation_output = ASEOutputSchema(
            converged=False,
            final_structure=atomsdata,
            simulation_input=params,
            error=str(e),
            success=False,
            wall_time=wall_time,
        )
        return simulation_output
