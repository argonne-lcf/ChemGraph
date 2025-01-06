from langchain_core.tools import tool
from typing import List, Dict, Any, Optional, Annotated
from comp_chem_agent.models.atomsdata import AtomsData
import pubchempy
from comp_chem_agent.models.ASEinput import ASESimulationInput

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
        pbc=[False, False, False],  # No periodic boundary conditions
    )
    #print(type(atoms_data))
    return atoms_data

@tool
def geometry_optimization(atomsdata: AtomsData, aseinput: ASESimulationInput) -> tuple[bool, AtomsData]:
    """
    Run geometry optimization using ASE with specified calculator and optimizer.
    
    Args:
        atomsdata: AtomsData object containing initial geometry
        aseinput: ASE simulation parameters

    Returns:
        tuple containing:
            - bool: True if optimization converged successfully, False otherwise
            - AtomsData: Final optimized structure
    """
    from ase import Atoms
    from ase.optimize import BFGS, LBFGS, GPMin, FIRE, MDMin
    
    OPTIMIZERS = {
        "bfgs": BFGS,
        "lbfgs": LBFGS,
        "gpmin": GPMin,
        "fire": FIRE,
        "mdmin": MDMin
    }
    
    fmax = aseinput.fmax
    steps = aseinput.steps
    calculator = aseinput.calculator
    optimizer = aseinput.optimizer

    if calculator == "mace_mp":
        from mace.calculators import mace_mp
        calculator = mace_mp(model="medium", dispersion=True, default_dtype="float32")
    elif calculator == "emt":
        from ase.calculators.emt import EMT
        calculator = EMT()

    atoms = Atoms(numbers=atomsdata.numbers, positions=atomsdata.positions, cell=atomsdata.cell, pbc=atomsdata.pbc)    
    atoms.calc = calculator

    try:
        optimizer_class = OPTIMIZERS.get(optimizer.lower())
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        dyn = optimizer_class(atoms)
        dyn.run(fmax=fmax, steps=steps)
        final_structure = AtomsData(
            numbers=atoms.numbers,
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc
        )
        return True, final_structure
    except Exception as e:
        print(f"Optimization failed: {str(e)}")
        return False, atomsdata 