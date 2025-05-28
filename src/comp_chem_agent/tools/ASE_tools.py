from langchain_core.tools import tool
from comp_chem_agent.models.atomsdata import AtomsData
from comp_chem_agent.models.ase_input import (
    ASEInputSchema,
    ASEOutputSchema,
)
import numpy as np


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


def load_calculator(calculator: dict):
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

    if "mace" in calc_type:
        from comp_chem_agent.models.calculators.mace_calc import MaceCalc

        calc = MaceCalc(**calculator).get_calculator()
    elif "emt" == calc_type:
        from comp_chem_agent.models.calculators.emt_calc import EMTCalc

        calc = EMTCalc(**calculator).get_calculator()
    elif "tblite" in calc_type:
        from comp_chem_agent.models.calculators.tblite_calc import TBLiteCalc

        calc = TBLiteCalc(**calculator).get_calculator()
    elif "orca" == calc_type:
        from comp_chem_agent.models.calculators.orca_calc import OrcaCalc

        calc = OrcaCalc(**calculator).get_calculator()

    elif "nwchem" == calc_type:
        from comp_chem_agent.models.calculators.nwchem_calc import NWChemCalc

        calc = NWChemCalc(**calculator).get_calculator()
    else:
        raise ValueError(
            f"Unsupported calculator: {calculator}. Available calculators are MACE (mace_mp, mace_off, mace_anicc), EMT, TBLite (GFN2-xTB, GFN1-xTB) and Orca"
        )
        return None
    return calc


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

    atomsdata = params.atomsdata
    optimizer = params.optimizer
    fmax = params.fmax
    steps = params.steps
    driver = params.driver
    pressure = params.pressure

    calc = load_calculator(calculator)
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
    atoms.calc = calc

    if driver == "energy":
        energy = atoms.get_potential_energy()
        simulation_output = ASEOutputSchema(
            converged=True,
            final_structure=atomsdata,
            simulation_input=params,
            success=True,
            single_point_energy=energy,
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

        final_structure = AtomsData(
            numbers=atoms.numbers,
            positions=final_coords,
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
        vib_data = {}
        thermo_data = {}

        if driver == "vib" or driver == "thermo":
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

            if driver == "thermo":
                # Approximation for system with a single atom.
                if len(atoms) == 1:
                    thermo_data["enthalpy"] = atoms.get_potential_energy()
                    thermo_data["entropy"] = 0
                    thermo_data["gibbs_free_energy"] = atoms.get_potential_energy()
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

                    thermo_data["enthalpy"] = thermo.get_enthalpy(temperature=temperature)
                    thermo_data["entropy"] = thermo.get_entropy(
                        temperature=temperature, pressure=pressure
                    )
                    thermo_data["gibbs_free_energy"] = thermo.get_gibbs_energy(
                        temperature=temperature, pressure=pressure
                    )
                    thermo_data["unit"] = "eV"

        simulation_output = ASEOutputSchema(
            converged=converged,
            final_structure=final_structure,
            simulation_input=params,
            vibrational_frequencies=vib_data,
            thermochemistry=thermo_data,
            success=True,
        )
        return simulation_output

    except Exception as e:
        simulation_output = ASEOutputSchema(
            converged=False,
            final_structure=atomsdata,
            simulation_input=params,
            error=str(e),
            success=False,
        )
        return simulation_output
