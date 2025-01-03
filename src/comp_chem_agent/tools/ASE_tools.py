from langchain_core.tools import tool
from typing import List, Dict, Any, Optional, Annotated
from comp_chem_agent.models.atomsdata import AtomsData
import pubchempy

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
def geometry_optimization_ase(atomsdata: AtomsData) -> AtomsData:
    """
    Run geometry optimization using ASE by reading AtomsData.
    
    Args:
        AtomsData: AtomsData object.

    Returns:
        optimized_structure (AtomsData): optimized structure in AtomsData format
    """
    from ase import Atoms
    from ase import Atoms
    from ase.optimize import BFGS
    from ase.calculators.emt import EMT

    ASEAtoms = Atoms(numbers=atomsdata.numbers, positions=atomsdata.positions, cell=atomsdata.cell, pbc=atomsdata.pbc, calculator=EMT())
    dyn = BFGS(ASEAtoms)
    print(dyn.run(fmax=0.05))

    new_atomsdata = AtomsData(numbers=ASEAtoms.numbers, positions=ASEAtoms.positions, cell=ASEAtoms.cell, pbc=ASEAtoms.pbc)
    #print(new_atomsdata)
    return new_atomsdata

