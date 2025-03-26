import pubchempy as pcp
import random
import time
import json
from comp_chem_agent.tools.ASE_tools import smiles_to_atomsdata, molecule_name_to_smiles


def get_random_molecule_names(n=2, cid_range=(0, 10000000), seed=42, natoms=30):
    """Get a list of random molecule names and smiles from PubChemPy.

    Args:
        n (int): Number of molecules to retrieve.
        cid_range (tuple): Range of PubChem CIDs to sample from.
        seed (int): Random seed for reproducibility.
        natoms (int): Maximum number of atoms per molecule.

    Returns:
        list: A list of dictionaries, each containing data for one molecule.
    """
    random.seed(seed)
    output = []
    tried = set()
    count = 0

    while len(output) < n:
        cid = random.randint(*cid_range)
        if cid in tried:
            continue
        tried.add(cid)

        try:
            compound = pcp.Compound.from_cid(cid)
            name = compound.iupac_name or (compound.synonyms[0] if compound.synonyms else None)
            if not name:
                continue

            smiles = molecule_name_to_smiles.invoke({"name": name})
            atomsdata = smiles_to_atomsdata.invoke({"smiles": smiles})

            if len(atomsdata.numbers) < natoms:
                molecule_info = {
                    "index": count,
                    "name": name,
                    "number_of_atoms": len(atomsdata.numbers),
                    "smiles": smiles,
                }
                output.append(molecule_info)
                count += 1
                print(count)
            else:
                print(f"Too many atoms in {name}, skipping...")

        except Exception:
            continue

        time.sleep(0.5)

    return output


def main():
    output = get_random_molecule_names(n=50)
    with open('pubchempy_molecule_max.json', 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
