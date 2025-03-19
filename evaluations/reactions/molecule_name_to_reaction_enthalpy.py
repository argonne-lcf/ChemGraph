from comp_chem_agent.tools.ASE_tools import run_ase, smiles_to_atomsdata, molecule_name_to_smiles
from comp_chem_agent.models.ase_input import ASEInputSchema
from reactions_dataset import reaction_list

reactions = {
    "reactants": [{"name": "water", "coefficient": 2}],
    "products": [{"name": "hydrogen", "coefficient": 2}, {"name": "oxygen", "coefficient": 1}],
}

reactions = reaction_list[13]
prop = "enthalpy"

input_dict = {
    "atomsdata": None,
    "driver": "thermo",
    # "calculator": {"calculator_type": "mace_mp"},
    "calculator": {"calculator_type": "nwchem", "basis": "3-21G", "mult": 1, "odft": True},
}
workflow = {
    "reaction": reactions,
    "tool_calls": {
        "molecule_name_to_smiles": {"name": []},
        "smiles_to_atomsdata": {"smiles": []},
        "run_ase": {"atomsdata": []},
    },
    f"reactants_{prop}": [],
    f"product_{prop}": [],
    prop: 0,
}

print(workflow)
for r in reactions["reactants"]:
    name = r["name"]
    smiles = molecule_name_to_smiles.invoke({"name": name})
    workflow["tool_calls"]["smiles_to_atomsdata"]["smiles"].append(smiles)
    atomsdata = smiles_to_atomsdata.invoke({"smiles": smiles})
    workflow["tool_calls"]["run_ase"]["atomsdata"].append(atomsdata)
    input_dict["atomsdata"] = atomsdata
    params = ASEInputSchema(**input_dict)
    aseoutput = run_ase.invoke({"params": params})
    print(aseoutput)
    workflow[prop] = workflow[prop] - aseoutput.thermochemistry[prop] * r["coefficient"]
    workflow[f"reactants_{prop}"].append(aseoutput.thermochemistry[prop])

for p in reactions["products"]:
    name = p["name"]
    smiles = molecule_name_to_smiles.invoke({"name": name})
    workflow["tool_calls"]["smiles_to_atomsdata"]["smiles"].append(smiles)
    atomsdata = smiles_to_atomsdata.invoke({"smiles": smiles})
    workflow["tool_calls"]["run_ase"]["atomsdata"].append(atomsdata)

    input_dict["atomsdata"] = atomsdata
    params = ASEInputSchema(**input_dict)
    aseoutput = run_ase.invoke({"params": params})
    workflow[prop] = workflow[prop] + aseoutput.thermochemistry[prop] * p["coefficient"]
    workflow[f"product_{prop}"].append(aseoutput.thermochemistry[prop])

print(workflow)
