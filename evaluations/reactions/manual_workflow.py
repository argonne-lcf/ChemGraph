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
        "tool_calls": [],
        "result": {"value": 0, "property": prop, "unit": "eV"},
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
            workflow["tool_calls"].append({"molecule_name_to_smiles": {"name": name}})
            workflow["tool_calls"].append({"smiles_to_atomsdata": {"smiles": smiles}})
            input_dict["atomsdata"] = input_dict["atomsdata"].model_dump()
            workflow["tool_calls"].append({"run_ase": {"params": input_dict}})

            workflow["result"]["value"] = (
                workflow["result"]["value"] - aseoutput.thermochemistry[prop] * r["coefficient"]
            )

        for p in reaction["products"]:
            input_dict = {
                "atomsdata": None,
                "driver": "thermo",
                "calculator": calculator,
            }
            name = p["name"]
            smiles = molecule_name_to_smiles.invoke({"name": name})
            atomsdata = smiles_to_atomsdata.invoke({"smiles": smiles})
            input_dict["atomsdata"] = atomsdata
            params = ASEInputSchema(**input_dict)
            aseoutput = run_ase.invoke({"params": params})

            # Populate worklfow to return
            workflow["tool_calls"].append({"molecule_name_to_smiles": {"name": name}})
            workflow["tool_calls"].append({"smiles_to_atomsdata": {"smiles": smiles}})
            input_dict["atomsdata"] = input_dict["atomsdata"].model_dump()
            workflow["tool_calls"].append({"run_ase": {"params": input_dict}})
            workflow["result"]["value"] = (
                workflow["result"]["value"] + aseoutput.thermochemistry[prop] * r["coefficient"]
            )

        return workflow
    except Exception as e:
        print(reaction, e)
        return e
