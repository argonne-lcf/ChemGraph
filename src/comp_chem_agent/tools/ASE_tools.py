from langchain_core.tools import tool
from comp_chem_agent.models.atomsdata import AtomsData
import pubchempy
from comp_chem_agent.models.ase_input import (
    ASEInputSchema,
    ASEOutputSchema,
)
import json
import numpy as np
from langchain_core.messages import HumanMessage

from comp_chem_agent.state.state import MultiAgentState


@tool
def molecule_name_to_smiles(name: str) -> str:
    """
    Convert a molecule name SMILES format.

    Args:
        name: Molecule name.

    Returns:
        SMILES string.
    """
    return pubchempy.get_compounds(str(name), "name")[0].canonical_smiles


@tool
def smiles_to_atomsdata(smiles: str) -> AtomsData:
    """
    Convert a SMILES string to AtomsData format.

    Args:
        smiles: Input SMILES string.

    Returns:
        coords: coordinates of the atoms.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Generate the molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    # Add hydrogens and optimize 3D structure
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol) != 0:
        raise ValueError("Failed to generate 3D coordinates.")
    if AllChem.UFFOptimizeMolecule(mol) != 0:
        raise ValueError("Failed to optimize 3D geometry.")

    # Extract atomic information
    conf = mol.GetConformer()
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]

    # Create AtomsData object
    atoms_data = AtomsData(
        numbers=numbers,
        positions=positions,
        cell=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        pbc=[False, False, False],  # No periodic boundary conditions
    )
    return atoms_data


@tool
def file_to_atomsdata(fname: str) -> AtomsData:
    """
    Convert a structure file to AtomsData format using ASE.

    Args:
        fname: Path to the input structure file (supports various formats like xyz, pdb, cif, etc.)

    Returns:
        AtomsData: Object containing the atomic structure information

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If the file format is not supported or file is corrupted
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
def save_atomsdata_to_file(atomsdata: AtomsData, fname: str = "output.xyz") -> None:
    """
    Save an AtomsData object to a file using ASE.

    Args:
        atomsdata: AtomsData object to save
        fname: Path to the output file
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

    Args:
        atomsdata (AtomsData): an AtomsData object.

    Returns:
        int: rotational symmetry number.
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

    Args:
        atomsdata (AtomsData): AtomsData object
        tol (float, optional): Tolerance to check for linear molecule. Defaults to 1e-3.

    Returns:
        bool: True if the molecule is linear, False otherwise.
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
def run_geometry_optimization(params: ASEInputSchema) -> ASEOutputSchema:
    """Run geometry optimization using Atomic Simulation Environment (ASE).

    Args:
        params (ASEInputSchema): ASEInputSchema object.

    Returns:
        ASEOutputSchema: ASEOutputSchema object.
    """
    calculator = params.calculator.model_dump()
    atomsdata = params.atomsdata
    optimizer = params.optimizer
    fmax = params.fmax
    steps = params.steps

    calc = load_calculator(calculator)
    if calc is None:
        e = f"Unsupported calculator: {calculator}. Available calculators are MACE (mace_mp, mace_off, mace_anicc), EMT, TBLite (GFN2-xTB, GFN1-xTB) and Orca"
        raise ValueError(
            f"Unsupported calculator: {calculator}. Available calculators are MACE (mace_mp, mace_off, mace_anicc), EMT, TBLite (GFN2-xTB, GFN1-xTB) and Orca"
        )
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

        dyn = optimizer_class(atoms)
        converged = dyn.run(fmax=fmax, steps=steps)

        final_structure = AtomsData(
            numbers=atoms.numbers,
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
        simulation_output = ASEOutputSchema(
            converged=converged,
            final_structure=final_structure,
            simulation_input=params,
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


@tool
def run_vibrational_frequency(params: ASEInputSchema) -> ASEOutputSchema:
    """Run vibrational frequency calculation using Atomic Simulation Environment (ASE).

    Args:
        params (ASEInputSchema): ASEInputSchema object.

    Returns:
        ASEOutputSchema: ASEOutputSchema object.
    """

    from ase import Atoms
    from ase.vibrations import Vibrations
    import ase.units as units

    calculator = params.calculator.model_dump()
    atomsdata = params.atomsdata
    calc = load_calculator(calculator)
    if calc is None:
        e = f"Unsupported calculator: {calculator}. Available calculators are MACE (mace_mp, mace_off, mace_anicc), EMT, TBLite (GFN2-xTB, GFN1-xTB) and Orca"
        raise ValueError(e)
        simulation_output = ASEOutputSchema(
            converged=False,
            final_structure=atomsdata,
            simulation_input=params,
            error=str(e),
            success=False,
        )
        return simulation_output
    try:
        atoms = Atoms(
            numbers=atomsdata.numbers,
            positions=atomsdata.positions,
            cell=atomsdata.cell,
            pbc=atomsdata.pbc,
        )
        atoms.calc = calc

        vib_data = {}
        vib_data['energies'] = []
        vib_data['energy_unit'] = 'meV'

        vib_data['frequencies'] = []
        vib_data['frequency_unit'] = 'cm-1'

        vib = Vibrations(atoms)
        vib.clean()
        vib.run()

        energies = vib.get_energies()
        for idx, e in enumerate(energies):
            if abs(e.imag) > 1e-8:
                c = 'i'
                e = e.imag
            else:
                c = ''
                e = e.real
            vib_data['energies'].append(str(1e3 * e) + c)
            vib_data['frequencies'].append(str(e / units.invcm) + c)
        simulation_output = ASEOutputSchema(
            converged=True,
            final_structure=atomsdata,
            simulation_input=params,
            vibrational_frequencies=vib_data,
            success=True,
        )
        return simulation_output
    except Exception as e:
        simulation_output = ASEOutputSchema(
            converged=False,
            final_structure=atomsdata,
            simulation_input=params,
            error=str(e),
            success=True,
        )
        return simulation_output


@tool
def run_ase(params: ASEInputSchema) -> ASEOutputSchema:
    """Run ASE calculations using specified input parameters.

    Args:
        params (ASEInputSchema): ASEInputSchema object.

    Returns:
        ASEOutputSchema: ASEOutputSchema object.
    """
    calculator = params.calculator.model_dump()
    atomsdata = params.atomsdata
    optimizer = params.optimizer
    fmax = params.fmax
    steps = params.steps
    driver = params.driver
    temperature = params.temperature
    pressure = params.pressure

    calc = load_calculator(calculator)
    if calc is None:
        e = f"Unsupported calculator: {calculator}. Available calculators are MACE (mace_mp, mace_off, mace_anicc), EMT, TBLite (GFN2-xTB, GFN1-xTB) and Orca"
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

    if driver == 'energy':
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

        dyn = optimizer_class(atoms)
        converged = dyn.run(fmax=fmax, steps=steps)

        final_structure = AtomsData(
            numbers=atoms.numbers,
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
        vib_data = {}
        thermo_data = {}

        if driver == "vib" or driver == "thermo":
            from ase.vibrations import Vibrations
            from ase import units

            vib_data['energies'] = []
            vib_data['energy_unit'] = 'meV'

            vib_data['frequencies'] = []
            vib_data['frequency_unit'] = 'cm-1'

            vib = Vibrations(atoms)
            vib.clean()
            vib.run()

            energies = vib.get_energies()
            for idx, e in enumerate(energies):
                if abs(e.imag) > 1e-8:
                    c = 'i'
                    e = e.imag
                else:
                    c = ''
                    e = e.real
                vib_data['energies'].append(str(1e3 * e) + c)
                vib_data['frequencies'].append(str(e / units.invcm) + c)

            if driver == "thermo":
                from ase.thermochemistry import IdealGasThermo

                potentialenergy = atoms.get_potential_energy()
                vib_energies = vib.get_energies()

                linear = is_linear_molecule.invoke({'atomsdata': final_structure})
                symmetrynumber = get_symmetry_number.invoke({'atomsdata': final_structure})

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

                thermo_data['enthalpy'] = thermo.get_enthalpy(temperature=temperature)
                thermo_data['entropy'] = thermo.get_entropy(
                    temperature=temperature, pressure=pressure
                )
                thermo_data['gibbs_free_energy'] = thermo.get_gibbs_energy(
                    temperature=temperature, pressure=pressure
                )
                thermo_data['unit'] = 'eV'

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


@tool
def run_single_point(params: ASEInputSchema) -> float:
    """Run single point calculation and return the energy.

    Args:
        params (ASEInputSchema): _description_

    Returns:
        : _description_
    """
    from ase import Atoms

    calculator = params.calculator.model_dump()
    atomsdata = params.atomsdata
    calc = load_calculator(calculator)
    if calc is None:
        e = f"Unsupported calculator: {calculator}. Available calculators are MACE (mace_mp, mace_off, mace_anicc), EMT, TBLite (GFN2-xTB, GFN1-xTB) and Orca"
        raise ValueError(e)
        simulation_output = ASEOutputSchema(
            converged=False,
            final_structure=atomsdata,
            simulation_input=params,
            error=str(e),
            success=False,
        )
        return simulation_output

    atoms = Atoms(
        numbers=atomsdata.numbers,
        positions=atomsdata.positions,
        cell=atomsdata.cell,
        pbc=atomsdata.pbc,
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()

    return energy


@tool
def calculate_thermochemistry(params: ASEInputSchema) -> ASEOutputSchema:
    """Calculate thermochemistry after performing vibrational frequency calculation

    Args:
        params (ASEInputSchema): ASEInputSchema object.

    Returns:
        ASEOutputSchema: ASEOutputSchema object.
    """
    calculator = params.calculator.model_dump()
    atomsdata = params.atomsdata
    optimizer = params.optimizer
    fmax = params.fmax
    steps = params.steps
    temperature = params.temperature
    pressure = params.pressure

    calc = load_calculator(calculator)
    if calc is None:
        e = f"Unsupported calculator: {calculator}. Available calculators are MACE (mace_mp, mace_off, mace_anicc), EMT, TBLite (GFN2-xTB, GFN1-xTB) and Orca"
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

        dyn = optimizer_class(atoms)
        converged = dyn.run(fmax=fmax, steps=steps)

        final_structure = AtomsData(
            numbers=atoms.numbers,
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc,
        )

        from ase.vibrations import Vibrations
        from ase import units

        vib_data = {}
        vib_data['energies'] = []
        vib_data['energy_unit'] = 'meV'

        vib_data['frequencies'] = []
        vib_data['frequency_unit'] = 'cm-1'

        vib = Vibrations(atoms)
        vib.clean()
        vib.run()

        energies = vib.get_energies()
        for idx, e in enumerate(energies):
            if abs(e.imag) > 1e-8:
                c = 'i'
                e = e.imag
            else:
                c = ''
                e = e.real
            vib_data['energies'].append(str(1e3 * e) + c)
            vib_data['frequencies'].append(str(e / units.invcm) + c)

        from ase.thermochemistry import IdealGasThermo

        thermo_data = {}
        potentialenergy = atoms.get_potential_energy()
        vib_energies = vib.get_energies()

        linear = is_linear_molecule.invoke({'atomsdata': final_structure})
        symmetrynumber = get_symmetry_number.invoke({'atomsdata': final_structure})

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

        thermo_data = {}
        thermo_data['enthalpy'] = thermo.get_enthalpy(temperature=temperature)
        thermo_data['entropy'] = thermo.get_entropy(temperature=temperature, pressure=pressure)
        thermo_data['gibbs_free_energy'] = thermo.get_gibbs_energy(
            temperature=temperature, pressure=pressure
        )
        thermo_data['unit'] = 'eV'

        simulation_output = ASEOutputSchema(
            converged=converged,
            final_structure=final_structure,
            simulation_input=params,
            vibrational_frequencies=vib_data,
            thermochemistry=thermo_data,
        )
        return simulation_output

    except Exception as e:
        simulation_output = ASEOutputSchema(
            converged=False, final_structure=atomsdata, simulation_input=params, error=str(e)
        )
        return simulation_output


def run_ase_with_state(state: MultiAgentState):
    # Get parameters to run ASE from state
    parameters = state["parameter_response"][-1]
    params = json.loads(parameters.content)

    calculator = params["calculator"]

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

    atomsdata = params["atomsdata"]
    optimizer = params["optimizer"]
    fmax = params["fmax"]
    steps = params["steps"]
    driver = params["driver"]
    temperature = params["temperature"]
    pressure = params["pressure"]

    from ase import Atoms
    from ase.optimize import BFGS, LBFGS, GPMin, FIRE, MDMin

    atoms = Atoms(
        numbers=atomsdata["numbers"],
        positions=atomsdata["positions"],
        cell=atomsdata["cell"],
        pbc=atomsdata["pbc"],
    )
    atoms.calc = calc
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

        dyn = optimizer_class(atoms)
        converged = dyn.run(fmax=fmax, steps=steps)

        final_structure = AtomsData(
            numbers=atoms.numbers,
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
        thermo_data = {}
        vib_data = {}
        if driver == "vib" or driver == "thermo":
            from ase.vibrations import Vibrations
            from ase import units

            vib_data = {}
            vib_data['energies'] = []
            vib_data['energy_unit'] = 'meV'

            vib_data['frequencies'] = []
            vib_data['frequency_unit'] = 'cm-1'

            vib = Vibrations(atoms)
            vib.clean()
            vib.run()

            energies = vib.get_energies()
            for idx, e in enumerate(energies):
                if abs(e.imag) > 1e-8:
                    c = 'i'
                    e = e.imag
                else:
                    c = ''
                    e = e.real
                vib_data['energies'].append(str(1e3 * e) + c)
                vib_data['frequencies'].append(str(e / units.invcm) + c)

            if driver == "thermo":
                from ase.thermochemistry import IdealGasThermo

                potentialenergy = atoms.get_potential_energy()
                vib_energies = vib.get_energies()

                linear = is_linear_molecule.invoke({'atomsdata': final_structure})
                symmetrynumber = get_symmetry_number.invoke({'atomsdata': final_structure})

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

                thermo_data['enthalpy'] = thermo.get_enthalpy(temperature=temperature)
                thermo_data['entropy'] = thermo.get_entropy(
                    temperature=temperature, pressure=pressure
                )
                thermo_data['gibbs_free_energy'] = thermo.get_gibbs_energy(
                    temperature=temperature, pressure=pressure
                )
                thermo_data['unit'] = 'eV'

        simulation_output = ASEOutputSchema(
            converged=converged,
            final_structure=final_structure,
            simulation_input=ASEInputSchema(**params),
            vibrational_frequencies=vib_data,
            thermochemistry=thermo_data,
        )
        output = []
        output.append(HumanMessage(role="system", content=simulation_output.model_dump_json()))
        return {"opt_response": output}

    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        return {"opt_response": str(e)}
