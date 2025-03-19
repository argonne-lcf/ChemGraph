from comp_chem_agent.tools.ASE_tools import run_ase, smiles_to_atomsdata, molecule_name_to_smiles
from comp_chem_agent.models.ase_input import ASEInputSchema

reactants = ["carbon monoxide", "water"]
products = ["carbon dioxide", "hydrogen"]
prop = "gibbs_free_energy"

parameters_to_run = {
    "reactants": reactants,
    "products": products,
    "prop": "gibbs_free_energy",
    "temperature": 298,
    "pressure": 100000,
}
input_dict = {
    "atomsdata": None,
    "driver": "thermo",
    "calculator": {"calculator_type": "nwchem", "basis": "3-21G"},
}
workflow = {
    "reactants": reactants,
    "products": products,
    "tool_calls": {
        "molecule_name_to_smiles": {"name": reactants + products},
        "smiles_to_atomsdata": {"smiles": []},
        "run_ase": {"atomsdata": []},
    },
    f"reactants_{prop}": [],
    f"product_{prop}": [],
    prop: 0,
}


for r in reactants:
    smiles = molecule_name_to_smiles.invoke({"name": r})
    workflow["tool_calls"]["smiles_to_atomsdata"]["smiles"].append(smiles)
    atomsdata = smiles_to_atomsdata.invoke({"smiles": smiles})
    workflow["tool_calls"]["run_ase"]["atomsdata"].append(atomsdata)
    input_dict["atomsdata"] = atomsdata
    params = ASEInputSchema(**input_dict)
    aseoutput = run_ase.invoke({"params": params})
    workflow[prop] = workflow[prop] - aseoutput.thermochemistry[prop]
    workflow[f"reactants_{prop}"].append(aseoutput.thermochemistry[prop])

for p in products:
    smiles = molecule_name_to_smiles.invoke({"name": p})
    workflow["tool_calls"]["smiles_to_atomsdata"]["smiles"].append(smiles)
    atomsdata = smiles_to_atomsdata.invoke({"smiles": smiles})
    workflow["tool_calls"]["run_ase"]["atomsdata"].append(atomsdata)

    input_dict["atomsdata"] = atomsdata
    params = ASEInputSchema(**input_dict)
    aseoutput = run_ase.invoke({"params": params})
    workflow[prop] = workflow[prop] + aseoutput.thermochemistry[prop]
    workflow[f"product_{prop}"].append(aseoutput.thermochemistry[prop])

print(workflow)
