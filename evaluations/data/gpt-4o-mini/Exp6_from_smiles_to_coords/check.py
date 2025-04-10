import json

with open("llm_workflow.json", "r") as f:
    data = json.load(f)

for mol in data:
    if mol["smiles"] == "C1C(C(CN1)C(=O)O)C2=CC(=CC=C2)C(F)(F)F.Cl":
        print(len(mol["llm_workflow"]["result"]["positions"]))
        print(len(mol["llm_workflow"]["result"]["numbers"]))

