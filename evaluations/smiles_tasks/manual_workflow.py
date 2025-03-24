from comp_chem_agent.tools.ASE_tools import smiles_to_atomsdata, run_ase
from comp_chem_agent.models.ase_input import ASEInputSchema


def get_atomsdata_from_smiles(smiles: str) -> dict:
    """Return a workflow of converting smiles to atomsdata.

    Args:
        smiles (str): SMILES string.

    Returns:
        dict: Workflow details including input parameters and results.
    """
    workflow = {
        "tool_calls": [
            {
                "smiles_to_atomsdata": {"smiles": smiles},
            }
        ],
        "result": None,
    }
    try:
        result = smiles_to_atomsdata.invoke({"smiles": smiles})

        # Populate workflow with relevant data.
        workflow["result"] = result.model_dump_json()
        return workflow
    except Exception as e:
        return f"Error message: {e}"


def get_geometry_optimization_from_smiles(smiles: str, calculator: dict) -> dict:
    """Run and return a workflow of geometry optimization using SMILES and a calculator as input.s

    Args:
        smiles (str): SMILES string.
        calculator (dict): details of input calculator/method.

    Returns:
        dict: Workflow details including input parameters and results.
    """

    workflow = {
        "tool_calls": [{"smiles_to_atomsdata": {"smiles": smiles}}],
        "result": None,
    }
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
        workflow["result"] = result
        dat = {"run_ase": {"params": params.model_dump()}}
        workflow["tool_calls"].append(dat)

        return workflow
    except Exception as e:
        return f"Error message: {e}"


def get_vibrational_frequencies_from_smiles(smiles: str, calculator: dict) -> dict:
    """Run and return a workflow of calculating vibrational frequencies using SMILES and a calculator as input.s

    Args:
        smiles (str): SMILES string.
        calculator (dict): details of input calculator/method.

    Returns:
        dict: Workflow details including input parameters and results.
    """

    workflow = {
        "tool_calls": [{"smiles_to_atomsdata": {"smiles": smiles}}],
        "result": None,
    }
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
        workflow["result"] = {}
        workflow["result"]["frequency_cm1"] = result
        dat = {"run_ase": {"params": params.model_dump()}}
        workflow["tool_calls"].append(dat)
        return workflow

    except Exception as e:
        return f"Error message: {e}"
