from comp_chem_agent.tools.ASE_tools import molecule_name_to_smiles, smiles_to_atomsdata, run_ase
from comp_chem_agent.models.ase_input import ASEInputSchema


def get_atomsdata_from_molecule_name(name: str) -> dict:
    """Return a workflow of converting a molecule name to an atomsdata.

    Args:
        name (str): a molecule name.

    Returns:
        dict: a workflow details including input parameters and results.
    """
    workflow = {
        "tool_calls": [],
        "result": None,
    }
    try:
        smiles = molecule_name_to_smiles.invoke({"name": name})
        result = smiles_to_atomsdata.invoke({"smiles": smiles})

        # Populate workflow with relevant data.
        workflow["tool_calls"].append({"molecule_name_to_smiles": {"name": name}})
        workflow["tool_calls"].append({"smiles_to_atomsdata": {"smiles": smiles}})
        workflow["result"] = result.model_dump_json()
        return workflow
    except Exception as e:
        return f"Error message: {e}"


def get_geometry_optimization_from_molecule_name(name: str, calculator: dict) -> dict:
    """Run and return a workflow of geometry optimization using a molecule name and a calculator as input.

    Args:
        smiles (str): SMILES string.
        calculator (dict): details of input calculator/method.

    Returns:
        dict: Workflow details including input parameters and results.
    """

    workflow = {
        "tool_calls": [],
        "result": None,
    }
    smiles = molecule_name_to_smiles.invoke({"name": name})
    atomsdata = smiles_to_atomsdata.invoke({"smiles": smiles})
    input_dict = {
        "atomsdata": atomsdata,
        "driver": "opt",
        "calculator": calculator,
    }
    try:
        params = ASEInputSchema(**input_dict)
        aseoutput = run_ase.invoke({"params": params})

        result = aseoutput.final_structure.model_dump()

        # Populate workflow with relevant data.
        workflow["tool_calls"].append({"molecule_name_to_smiles": {"name": name}})
        workflow["tool_calls"].append({"smiles_to_atomsdata": {"smiles": smiles}})
        input_dict["atomsdata"] = input_dict["atomsdata"].model_dump()
        workflow["tool_calls"].append({"run_ase": {"params": input_dict}})
        workflow["result"] = result.model_dump_json()

        return workflow
    except Exception as e:
        return f"Error message: {e}"


def get_vibrational_frequencies_from_molecule_name(name: str, calculator: dict) -> dict:
    """Run and return a workflow of calculating vibrational frequencies using molecule name and a calculator as input

    Args:
        smiles (str): SMILES string.
        calculator (dict): details of input calculator/method.

    Returns:
        dict: Workflow details including input parameters and results.
    """

    workflow = {
        "tool_calls": [],
        "result": {},
    }
    smiles = molecule_name_to_smiles.invoke({"name": name})
    atomsdata = smiles_to_atomsdata.invoke({"smiles": smiles})
    input_dict = {
        "atomsdata": atomsdata,
        "driver": "vib",
        "calculator": calculator,
    }
    try:
        params = ASEInputSchema(**input_dict)
        aseoutput = run_ase.invoke({"params": params})

        result = aseoutput.vibrational_frequencies['frequencies']
        # Populate workflow with relevant data.
        workflow["tool_calls"].append({"molecule_name_to_smiles": {"name": name}})
        workflow["tool_calls"].append({"smiles_to_atomsdata": {"smiles": smiles}})
        input_dict["atomsdata"] = input_dict["atomsdata"].model_dump()
        workflow["tool_calls"].append({"run_ase": {"params": input_dict}})
        workflow["result"]["frequency_cm1"] = result
        return workflow
    except Exception as e:
        return f"Error message: {e}"
