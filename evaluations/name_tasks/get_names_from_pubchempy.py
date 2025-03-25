import pubchempy as pcp
import random
import time
from comp_chem_agent.tools.ASE_tools import molecule_name_to_smiles, smiles_to_atomsdata
import json


def get_random_molecule_names(n=2, cid_range=(0, 10000000), seed=42):
    """Get a list of random molecule names from PubChemPy.

    Args:
        n (int, optional): Number of molecules. Defaults to 2.
        cid_range (tuple, optional): CID range. Defaults to (0, 10000000).
        seed (int, optional): Seed number. Defaults to 42.
    """
    random.seed(seed)
    output = []
    names = []
    number_of_atoms = []
    list_of_smiles = []
    tried = set()
    count = 0

    while len(names) < n:
        cid = random.randint(*cid_range)
        if cid in tried:
            continue
        tried.add(cid)
        try:
            compound = pcp.Compound.from_cid(cid)
            name = compound.iupac_name or compound.synonyms[0]
            if name:
                smiles = molecule_name_to_smiles.invoke({"name": name})
                atomsdata = smiles_to_atomsdata.invoke({"smiles": smiles})
                if len(atomsdata.numbers) < 100:
                    names.append(name)
                    number_of_atoms.append(len(atomsdata.numbers))
                    list_of_smiles.append(smiles)

                    data = {}
                    data["index"] = count
                    data["name"] = name
                    data["number_of_atoms"] = len(atomsdata.numbers)
                    data["smiles"] = smiles

                    output.append(data)
                    count = count + 1
                    print(count)
                else:
                    print(f"Failed for {name}")
                    continue
        except Exception:
            continue
        time.sleep(0.5)

    # return names, number_of_atoms, list_of_smiles
    return output


# Example usage
# molecule_names, number_of_atoms, list_of_smiles = get_random_molecule_names(5)
output = get_random_molecule_names(100)
"""
print(len(molecule_names), len(number_of_atoms))
for i, name in enumerate(molecule_names):
    print(f"{number_of_atoms[i]}, {name}, {list_of_smiles[i]}")
"""

print(output)
with open('test.json', 'w') as f:
    json.dump(output, f, indent=4)
