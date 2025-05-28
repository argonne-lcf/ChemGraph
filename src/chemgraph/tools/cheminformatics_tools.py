import pubchempy
from langchain_core.tools import tool
from chemgraph.models.atomsdata import AtomsData


@tool
def molecule_name_to_smiles(name: str) -> str:
    """Convert a molecule name to SMILES format.

    Parameters
    ----------
    name : str
        The name of the molecule to convert.

    Returns
    -------
    str
        The SMILES string representation of the molecule.

    Raises
    ------
    IndexError
        If the molecule name is not found in PubChem.
    """
    return pubchempy.get_compounds(str(name), "name")[0].canonical_smiles


@tool
def smiles_to_atomsdata(smiles: str, randomSeed: int = 2025) -> AtomsData:
    """Convert a SMILES string to AtomsData format.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.
    randomSeed : int, optional
        Random seed for RDKit 3D structure generation, by default 2025.

    Returns
    -------
    AtomsData
        AtomsData object containing the molecular structure.

    Raises
    ------
    ValueError
        If the SMILES string is invalid or if 3D structure generation fails.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Generate the molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    # Add hydrogens and optimize 3D structure
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=randomSeed) != 0:
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
