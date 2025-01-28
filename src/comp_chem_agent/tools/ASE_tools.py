from langchain_core.tools import tool
from typing import List, Dict, Any, Optional, Annotated
from comp_chem_agent.models.atomsdata import AtomsData
import pubchempy
from comp_chem_agent.models.ASEinput import ASESimulationInput, ASESimulationOutput
import os
import numpy as np
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
    positions = [
        list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())
    ]
    
    # Create AtomsData object
    atoms_data = AtomsData(
        numbers=numbers,
        positions=positions,
        cell=[[0,0,0], [0,0,0], [0,0,0]],
        pbc=[False, False, False],  # No periodic boundary conditions
    )
    #print(type(atoms_data))
    return atoms_data

@tool
def geometry_optimization(atomsdata: AtomsData, calculator: str="mace_mp", optimizer: str="BFGS", fmax: float=0.01, steps: int=10) -> ASESimulationOutput:
    """
    Run geometry optimization using ASE with specified calculator and optimizer.
    
    Args:
        atomsdata: AtomsData object containing initial geometry
        aseinput: ASE simulation parameters

    Returns:    
        ASESimulationOutput object containing:
            - converged: True if optimization converged successfully, False otherwise
            - final_structure: Final structure from the simulation
            - simulation_input: ASESimulationInput object containing the input used for the simulation
            - gradients: List of tuples containing the maximum force at each 10th step of the optimization
    """
    from ase import Atoms
    from ase.optimize import BFGS, LBFGS, GPMin, FIRE, MDMin
    
    max_force_at_steps = []
    def capture_max_force(optimizer):
        """Callback function to capture the maximum force at each step."""
        current_step = optimizer.nsteps  # Access the iteration count from the optimizer instance
        forces = atoms.get_forces()  # Get the forces on atoms
        max_force = np.max(np.linalg.norm(forces, axis=1))  # Compute the maximum force (norm of the forces)
        max_force_at_steps.append(max_force)

    OPTIMIZERS = {
        "bfgs": BFGS,
        "lbfgs": LBFGS,
        "gpmin": GPMin,
        "fire": FIRE,
        "mdmin": MDMin
    }
    
    #calculator = calculator.lower()
    if calculator == "mace_mp":
        from mace.calculators import mace_mp
        calc = mace_mp(model="medium", dispersion=True, default_dtype="float32")
    elif calculator == "emt":
        from ase.calculators.emt import EMT
        calc = EMT()
    else:
        raise ValueError(f"Unsupported calculator: {calculator}. Available calculators are mace_mp and emt.")

    atoms = Atoms(numbers=atomsdata.numbers, positions=atomsdata.positions, cell=atomsdata.cell, pbc=atomsdata.pbc)    
    atoms.calc = calc

    try:
        optimizer_class = OPTIMIZERS.get(optimizer.lower())
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        dyn = optimizer_class(atoms)
        dyn.attach(lambda: capture_max_force(dyn), interval=10)  # Call every 10th step (interval=10)
        converged = dyn.run(fmax=fmax, steps=steps)
        final_structure = AtomsData(
            numbers=atoms.numbers,
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc
        )
        simulation_output = ASESimulationOutput(
            converged=converged,
            final_structure=final_structure,
            simulation_input=ASESimulationInput(
                atomsdata=atomsdata,
                calculator=calculator,
                optimizer=optimizer,
                fmax=fmax,
                steps=steps
            ),
            gradients=max_force_at_steps
        )
        return simulation_output
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        return "Error" 

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
            pbc=atoms.pbc.tolist()
        )
        return atoms_data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {fname}")
    except Exception as e:
        raise ValueError(f"Failed to read structure file: {str(e)}")

@tool            
def save_atomsdata_to_file(atomsdata: AtomsData, fname: str="output.xyz") -> None:
    """
    Save an AtomsData object to a file using ASE.
    
    Args:
        atomsdata: AtomsData object to save
        fname: Path to the output file
    """
    from ase.io import write
    from ase import Atoms
    try:
        atoms = Atoms(numbers=atomsdata.numbers, positions=atomsdata.positions, cell=atomsdata.cell, pbc=atomsdata.pbc)
        write(fname, atoms)
        return f"Successfully saved atomsdata to {fname}"
    except Exception as e:
        raise ValueError(f"Failed to save atomsdata to file: {str(e)}")