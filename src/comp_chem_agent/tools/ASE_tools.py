from langchain_core.tools import tool
from typing import List, Dict, Any, Optional, Annotated
from comp_chem_agent.models.atomsdata import AtomsData

@tool
def smiles_to_xyz(smiles: str):
    """
    Convert a SMILES string to a.
    
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
    atoms = mol.GetAtoms()

    coords = ''        
    for ida, atom in enumerate(atoms):
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords = coords + f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}"
        if ida < len(atoms):
            coords += ";"
        #xyz_file.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
    return coords

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
        cell=None,  # No unit cell for an isolated molecule
        pbc=[False, False, False],  # No periodic boundary conditions
    )
    #print(type(atoms_data))
    return atoms_data

@tool
def geometry_optimization(coords: str):
    """
    Run geometry optimization using PySCF by reading in the coordinates.
    
    Args:
        coords: coordinates of the atoms.

    Returns:
        optimized_structure: coordinates of the atoms after geometry optimization
    """
    from pyscf import gto, scf, geomopt
    from pyscf.geomopt.berny_solver import optimize

    mol = gto.M(
        atom=coords,
        basis='6-31g',
        charge=0,
        spin=0,
    )
    
    # Run geometry optimization
    mf = scf.RHF(mol)
    optimized_mol = optimize(mf)
    print("\nOptimized Geometry (Angstrom):")
    for atom in optimized_mol.atom:
        print(atom)
    
    # Print the total energy of the optimized geometry
    print("\nTotal Energy (Hartree):", mf.e_tot)

    return optimized_mol.atom

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

