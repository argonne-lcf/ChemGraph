import json
import argparse
from comp_chem_agent.tools.ASE_tools import smiles_to_atomsdata, run_ase
from comp_chem_agent.models.ase_input import ASEInputSchema
import datetime
import subprocess


def get_atomsdata_from_smiles(smiles: str) -> dict:
    """Return a workflow of converting smiles to atomsdata.

    Args:
        smiles (str): SMILES string.

    Returns:
        dict: Workflow details including input parameters and results.
    """
    workflow = {
        "tool_calls": [],
        "result": None,
    }
    try:
        result = smiles_to_atomsdata.invoke({"smiles": smiles})

        # Populate workflow with relevant data.
        workflow["tool_calls"].append({"smiles_to_atomsdata": {"smiles": smiles}})
        workflow["result"] = result.model_dump()
        return workflow
    except Exception as e:
        workflow["tool_calls"].append({"smiles_to_atomsdata": {"smiles": smiles}})
        workflow["result"] = f"ERROR - {str(e)}"
        return workflow


def get_geometry_optimization_from_smiles(smiles: str, calculator: dict) -> dict:
    """Run and return a workflow of geometry optimization using SMILES and a calculator as input.

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
        workflow["tool_calls"].append({"smiles_to_atomsdata": {"smiles": smiles}})
        input_dict["atomsdata"] = input_dict["atomsdata"].model_dump()
        workflow["tool_calls"].append({"run_ase": {"params": input_dict}})
        workflow["result"] = result

        return workflow
    except Exception as e:
        workflow["tool_calls"].append({"smiles_to_atomsdata": {"smiles": smiles}})
        input_dict["atomsdata"] = input_dict["atomsdata"].model_dump()
        workflow["tool_calls"].append({"run_ase": {"params": input_dict}})
        workflow["result"] = f"ERROR - {e}"
        return workflow


def get_vibrational_frequencies_from_smiles(smiles: str, calculator: dict) -> dict:
    """Run and return a workflow of calculating vibrational frequencies using SMILES and a calculator as input.

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
        workflow["tool_calls"].append({"smiles_to_atomsdata": {"smiles": smiles}})
        input_dict["atomsdata"] = input_dict["atomsdata"].model_dump()
        workflow["tool_calls"].append({"run_ase": {"params": input_dict}})
        return workflow

    except Exception as e:
        return f"Error message: {e}"


def main(fname: str, n_structures: int):
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

    combined_data = {}

    # Iterate through the first n_structures molecules
    for idx, molecule in enumerate(smiles_data[:n_structures]):
        smiles = molecule["smiles"]

        try:
            # Run the geometry optimization workflow
            manual_workflow = get_geometry_optimization_from_smiles(
                smiles, calculator={"calculator_type": "mace_mp"}
            )
        except Exception as e:
            print(f"ERROR running workflow for {smiles}. Error message: {e}")
            continue

        # Store results in a structured dictionary
        combined_data[smiles] = {"manual_workflow": manual_workflow}

        # Get metadata
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        try:
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            )
        except subprocess.CalledProcessError:
            git_commit = "unknown"

        metadata = {"timestamp": timestamp, "git_commit": git_commit}
        combined_data[smiles]["metadata"] = metadata
    # Save the results to a JSON file
    with open("manual_workflow.json", "w") as f:
        json.dump(combined_data, f, indent=4)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run geometry optimization on SMILES molecules.")
    parser.add_argument(
        "--fname",
        type=str,
        default="data_from_pubchempy.json",
        help="Path to the input SMILES JSON file (e.g., smiles_data.json)",
    )
    parser.add_argument(
        "--n_structures", type=int, default=30, help="Number of molecules to process (default: 30)"
    )
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.fname, args.n_structures)
