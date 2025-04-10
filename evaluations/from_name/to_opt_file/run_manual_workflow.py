import json
import argparse
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
        workflow["result"] = result.model_dump()
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
        workflow["result"] = result

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


def main(fname: str, start_index: int, end_index: int):
    """
    Run a manual geometry optimization workflow on a subset of molecules
    from the input SMILES dataset.

    Args:
        fname (str): Path to the JSON file containing SMILES data.
        n_structures (int): Number of molecules to process from the dataset.
    """
    # Load SMILES data from the specified JSON file
    with open(fname, "r") as f:
        smiles_data = json.load(f)

    combined_data = []

    # Iterate through the first n_structures molecules
    for idx, molecule in enumerate(smiles_data[start_index : end_index + 1]):
        smiles = molecule["smiles"]
        index = molecule["index"]
        name = molecule["name"]

        try:
            # Run the geometry optimization workflow
            manual_workflow = get_atomsdata_from_molecule_name(name)
        except Exception as e:
            print(f"ERROR running workflow for {smiles}. Error message: {e}")
            continue

        # Store results in a structured dictionary
        molecule_data = {
            "name": name,
            "smiles": smiles,
            "index": index,
            "manual_workflow": manual_workflow,
        }
        combined_data.append(molecule_data)

    # Save the results to a JSON file
    with open("manual_workflow.json", "w") as f:
        json.dump(combined_data, f, indent=4)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run geometry optimization using a molecule name.")
    parser.add_argument(
        "--fname",
        type=str,
        default="data_from_pubchempy.json",
        help="Path to the input SMILES JSON file (e.g., smiles_data.json)",
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="First molecule to process (default: 0)"
    )
    parser.add_argument(
        "--end_index", type=int, default=1, help="Last molecule to process (default: 1)"
    )

    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.fname, args.start_index, args.end_index)
