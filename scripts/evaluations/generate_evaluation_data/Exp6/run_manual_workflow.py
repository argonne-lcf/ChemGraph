import json
import argparse
from chemgraph.tools.ase_tools import (
    run_ase,
    save_atomsdata_to_file,
)
from chemgraph.tools.cheminformatics_tools import molecule_name_to_smiles, smiles_to_atomsdata
from chemgraph.models.ase_input import ASEInputSchema
import datetime
import subprocess
import os


def get_smiles_from_molecule_name(name: str) -> dict:
    """Return a workflow of converting a molecule name to a SMILES string.

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
        result = molecule_name_to_smiles.invoke({"name": name})

        # Populate workflow with relevant data.
        workflow["tool_calls"].append({"molecule_name_to_smiles": {"name": name}})
        workflow["result"] = result
        return workflow
    except Exception as e:
        workflow["tool_calls"].append({"molecule_name_to_smiles": {"name": name}})
        workflow["result"] = f"ERROR - {str(e)}"


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


def get_gibbs_energy_from_molecule_name(name: str, calculator: dict, temperature: float) -> dict:
    """Run and return a workflow of calculating gibbs free energy using a molecule name, a calculator and temperature as input.

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
        "driver": "thermo",
        "calculator": calculator,
        "temperature": temperature,
    }
    try:
        params = ASEInputSchema(**input_dict)
        aseoutput = run_ase.invoke({"params": params})

        result = aseoutput.thermochemistry['gibbs_free_energy']
        # Populate workflow with relevant data.
        workflow["tool_calls"].append({"molecule_name_to_smiles": {"name": name}})
        workflow["tool_calls"].append({"smiles_to_atomsdata": {"smiles": smiles}})
        input_dict["atomsdata"] = input_dict["atomsdata"].model_dump()
        workflow["tool_calls"].append({"run_ase": {"params": input_dict}})
        workflow["result"]["value"] = result
        workflow["result"]["property"] = "Gibbs free energy"
        workflow["result"]["unit"] = "eV"
        return workflow
    except Exception as e:
        workflow["tool_calls"].append({"molecule_name_to_smiles": {"name": name}})
        workflow["tool_calls"].append({"smiles_to_atomsdata": {"smiles": smiles}})
        input_dict["atomsdata"] = input_dict["atomsdata"].model_dump()
        workflow["tool_calls"].append({"run_ase": {"params": input_dict}})
        workflow["result"] = f"ERROR - {str(e)}"
        return workflow


def get_saved_file_from_molecule_name(
    name: str, calculator: dict, output_path: str = "manual_files"
) -> dict:
    """Run and return a workflow of geometry optimization and saving file using a molecule name and a calculator as input.

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

        os.makedirs(output_path, exist_ok=True)
        opt_atomsdata = aseoutput.final_structure.model_dump()
        filepath = os.path.join(output_path, name + '.xyz')

        result = save_atomsdata_to_file.invoke({"atomsdata": opt_atomsdata, "fname": filepath})
        # Populate workflow with relevant data.
        workflow["tool_calls"].append({"molecule_name_to_smiles": {"name": name}})
        workflow["tool_calls"].append({"smiles_to_atomsdata": {"smiles": smiles}})
        input_dict["atomsdata"] = input_dict["atomsdata"].model_dump()
        workflow["tool_calls"].append({"run_ase": {"params": input_dict}})
        workflow["tool_calls"].append({
            "save_atomsdata_to_file": {"atomsdata": opt_atomsdata, "fname": filepath}
        })
        workflow["result"] = result

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
        name = molecule["name"]

        try:
            # Run the geometry optimization workflow
            manual_workflow = get_saved_file_from_molecule_name(
                name, calculator={"calculator_type": "mace_mp"}
            )
        except Exception as e:
            print(f"ERROR running workflow for {name}. Error message: {e}")
            continue

        # Store results in a structured dictionary
        combined_data[name] = {"manual_workflow": manual_workflow}

        # Get metadata
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        try:
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            )
        except subprocess.CalledProcessError:
            git_commit = "unknown"

        metadata = {"timestamp": timestamp, "git_commit": git_commit}
        combined_data[name]["metadata"] = metadata
    # Save the results to a JSON file
    with open("manual_workflow.json", "w") as f:
        json.dump(combined_data, f, indent=4)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run geometry optimization and save the optimized structure from a molecule name."
    )
    parser.add_argument(
        "--fname",
        type=str,
        default="data_from_pubchempy.json",
        help="Path to the input SMILES JSON file (e.g., smiles_data.json)",
    )
    parser.add_argument(
        "--n_structures", type=int, default=15, help="Number of molecules to process (default: 15)"
    )
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.fname, args.n_structures)
