from rdkit import Chem

def canonicalize_smiles(smiles: str) -> str:
    """Convert SMILES to its canonical form using RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)

def compare_smiles(smiles1: str, smiles2: str) -> bool:
    """Compare two SMILES strings by their canonical forms."""
    try:
        can1 = canonicalize_smiles(smiles1)
        can2 = canonicalize_smiles(smiles2)
        return can1 == can2
    except ValueError as e:
        print(f"Error: {e}")
        return False

# === Example Usage ===
if __name__ == "__main__":
    smi_a = "C(C(=O)O)N"      # Glycine
    smi_b = "NCC(=O)O"        # Another valid form of glycine
    smi_a = "N[S](=O)(=O)Cc1cc(NC(=O)c2c(Cc3ccccc3)c(S(=O)(=O)CC)nn2)ccc1"
    smi_b = "CCS(=O)(=O)N1C(CC(=N1)C2=CC(=CC=C2)NS(=O)(=O)C)C3=CC=CC=C3"
    smi_a = "CCc1ccc2nc(Cc3ccccc3)n2c1"
    smi_b = "CC(C)CCN1C2=CC=CC=C2N=C1CC3=CC=CC=C3"
    are_equal = compare_smiles(smi_a, smi_b)

    print(f"Canonical SMILES A: {canonicalize_smiles(smi_a)}")
    print(f"Canonical SMILES B: {canonicalize_smiles(smi_b)}")
    print("Are they the same molecule?", "Yes" if are_equal else "No")

