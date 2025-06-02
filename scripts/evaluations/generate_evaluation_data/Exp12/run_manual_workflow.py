import json
import argparse
from chemgraph.tools.ase_tools import run_ase
from chemgraph.tools.cheminformatics_tools import smiles_to_atomsdata, molecule_name_to_smiles
from chemgraph.models.ase_input import ASEInputSchema
import datetime
import subprocess


def get_manual_workflow_result(
    reaction: dict,
    prop: str = "enthalpy",
    temperature: float = 298.15,
    pressure: float = 101325,
    calculator: dict = {},
):
    """
    Evaluate a reaction thermochemical property (e.g., enthalpy change).

    Args:
        reaction (dict): Dictionary with 'reactants' and 'products'. Each must contain
                         a 'name' and 'coefficient'.
        prop (str): Thermochemical property to compute. Default is 'enthalpy'.
        temperature (float): Temperature in Kelvin.
        pressure (float): Pressure in Pascals.
        calculator (dict): Optional ASE calculator parameters.

    Returns:
        dict: Workflow results with tool call logs and the computed reaction property.
    """

    def process_species(species_list, sign):
        total = 0
        for species in species_list:
            name = species["name"]
            coeff = species["coefficient"]

            smiles = molecule_name_to_smiles.invoke({"name": name})
            atomsdata = smiles_to_atomsdata.invoke({"smiles": smiles})

            input_dict = {
                "atomsdata": atomsdata,
                "driver": "thermo",
                "calculator": calculator,
                "temperature": temperature,
            }

            params = ASEInputSchema(**input_dict)
            aseoutput = run_ase.invoke({"params": params})

            workflow["tool_calls"].extend([
                {"molecule_name_to_smiles": {"name": name}},
                {"smiles_to_atomsdata": {"smiles": smiles}},
                {"run_ase": {"params": {**input_dict, "atomsdata": atomsdata.model_dump()}}},
            ])

            total += sign * aseoutput.thermochemistry[prop] * coeff
        return total

    workflow = {
        "tool_calls": [],
        "result": {"value": 0, "property": prop, "unit": "eV"},
    }

    try:
        result = 0
        result += process_species(reaction["reactants"], sign=-1)
        result += process_species(reaction["products"], sign=1)
        workflow["result"]["value"] = result
        return workflow
    except Exception as e:
        workflow["result"]["value"] = f"ERROR - {e}"
        return workflow


def main():
    parser = argparse.ArgumentParser(
        description="Run manual workflow for thermochemical property calculations."
    )
    parser.add_argument(
        "--reaction_fp",
        type=str,
        default="reactions_dataset.json",
        help="Path to the reactions dataset file.",
    )
    parser.add_argument(
        "--n_reactions", type=int, default=10, help="Number of reactions to process."
    )
    parser.add_argument(
        "--output_fp",
        type=str,
        default="manual_workflow.json",
        help="Path to save the output workflow results.",
    )
    args = parser.parse_args()

    combined_data = {}
    with open("reaction_dataset.json", "r") as rf:
        reactions = json.load(rf)

    for idx, reaction in enumerate(reactions[: args.n_reactions]):
        name = reaction["reaction_name"]
        calculator = {"calculator_type": "TBLite", "method": "GFN2-xTB"}
        # calculator = {"calculator_type": "mace_mp"}
        manual_workflow = get_manual_workflow_result(
            reaction, calculator=calculator, temperature=400
        )
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
    with open(args.output_fp, "w") as f:
        json.dump(combined_data, f, indent=4)


if __name__ == "__main__":
    main()
