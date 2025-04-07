import json

with open("llm_workflow.json", "r") as f:
    data = json.load(f)

for mol in data:
    if mol["name"] == "N-(3-cyano-6-methyl-4,5,6,7-tetrahydro-1-benzothiophen-2-yl)-4-nitrobenzamide":
        print(len(mol["llm_workflow"]["result"]["positions"]))
        print(len(mol["llm_workflow"]["result"]["numbers"]))

