from comp_chem_agent.tools.ASE_tools import run_ase, smiles_to_atomsdata, molecule_name_to_smiles
from comp_chem_agent.models.ase_input import ASEInputSchema


def get_manual_workflow_result(
    reaction: dict,
    prop: str = "enthalpy",
    temperature: float = 298,
    pressure: float = 101325,
    calculator: dict = {},
):
    workflow = {
        "reaction": reaction,
        "tool_calls": {
            "molecule_name_to_smiles": {"name": []},
            "smiles_to_atomsdata": {"smiles": []},
            "run_ase": {"params": []},
        },
        prop: 0,
    }
    try:
        for r in reaction["reactants"]:
            input_dict = {
                "atomsdata": None,
                "driver": "thermo",  # Right now just thermochemistry in this subset.
                "calculator": calculator,
            }

            name = r["name"]
            smiles = molecule_name_to_smiles.invoke({"name": name})
            atomsdata = smiles_to_atomsdata.invoke({"smiles": smiles})
            input_dict["atomsdata"] = atomsdata
            params = ASEInputSchema(**input_dict)
            aseoutput = run_ase.invoke({"params": params})

            # Populate worklfow to return
            workflow["tool_calls"]["molecule_name_to_smiles"]["name"].append(name)
            workflow["tool_calls"]["smiles_to_atomsdata"]["smiles"].append(smiles)
            workflow["tool_calls"]["run_ase"]["params"].append(params)
            workflow[prop] = workflow[prop] - aseoutput.thermochemistry[prop] * r["coefficient"]

        for p in reaction["products"]:
            input_dict = {
                "atomsdata": None,
                "driver": "thermo",  # Right now just thermochemistry in this subset.
                "calculator": calculator,
            }
            name = p["name"]
            smiles = molecule_name_to_smiles.invoke({"name": name})
            atomsdata = smiles_to_atomsdata.invoke({"smiles": smiles})
            input_dict["atomsdata"] = atomsdata
            params = ASEInputSchema(**input_dict)
            aseoutput = run_ase.invoke({"params": params})

            # Populate worklfow to return
            workflow["tool_calls"]["molecule_name_to_smiles"]["name"].append(name)
            workflow["tool_calls"]["smiles_to_atomsdata"]["smiles"].append(smiles)
            workflow["tool_calls"]["run_ase"]["params"].append(params)
            workflow[prop] = workflow[prop] + aseoutput.thermochemistry[prop] * r["coefficient"]

        return workflow
    except Exception as e:
        print(reaction, e)
        return e
