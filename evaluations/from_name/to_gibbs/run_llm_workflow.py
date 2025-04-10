import json
from comp_chem_agent.agent.llm_graph import llm_graph
from comp_chem_agent.utils.get_workflow_from_llm import get_workflow_from_state
import argparse


def get_query(
    name: str,
    query_name: str,  # options: atomsdata, opt, vib
    method: str = "mace_mp",
) -> str:
    """Get query for a SMILES-related task for CompChemAgent

    Args:
        name (str): Molecule name.
        query_name (str, optional): Type of query. Defaults to "atomsdata". Options: "atomsdata", "opt", "vib", "opt_method" and "vib_method".
        method (str, optional): The method/level of theory for CompChemAgent to run simulation. Defaults to "mace_mp".

    Returns:
        str: formatted query.
    """
    query_dict = {
        "name_to_coord": f"Provide the XYZ coordinates corresponding to this molecule: {name}",
        "name_to_opt": f"Perform geometry optimization for a molecule {name} using {method}",
        "name_to_vib": f"Run vibrational frequency calculation for a molecule {name} using {method}",
        "name_to_enthalpy": f"Calculate the enthalpy of a molecule {name} using {method}",
        "name_to_gibbs": f"Calculate the Gibbs free energy of a molecule {name} using {method} at T=300K and P=1 bar",
        "name_to_opt_file": f"Perform geometry optimization for a molecule {name} using {method}. Save the optimized coordinate in a XYZ file.",
    }
    return query_dict.get(query_name, "Query not found")  # Returns the query or a default message


def main(fname: str, start_index: int, end_index: int):
    """
    Run an LLM geometry optimization workflow on a subset of molecules
    from the input SMILES dataset.

    Args:
        fname (str): Path to the JSON file containing SMILES data.
        n_structures (int): Number of molecules to process from the dataset.
    """
    # Load SMILES data from the specified JSON file
    with open(fname, "r") as f:
        smiles_data = json.load(f)

    combined_data = []

    cca = llm_graph(
        model_name='gpt-4o-mini',
        workflow_type="single_agent_ase",
        structured_output=True,
        return_option="state",
    )

    # Iterate through the first n_structures molecules
    for idx, molecule in enumerate(smiles_data[start_index:end_index]):
        print("********************************************")
        print(
            f"MOLECULE INDEX {molecule['index']}: MOLECULE SMILES: {molecule['smiles']} MOLECULE NAME: {molecule['name']}"
        )
        print("********************************************")

        smiles = molecule["smiles"]
        index = molecule["index"]
        name = molecule["name"]

        query = get_query(name, query_name="name_to_gibbs", method="mace_mp")
        state = cca.run(query, config={"configurable": {"thread_id": f"{str(idx)}"}})

        llm_workflow = get_workflow_from_state(state)

        # Store results in a structured dictionary
        molecule_data = {
            "name": name,
            "smiles": smiles,
            "index": index,
            "llm_workflow": llm_workflow,
        }
        combined_data.append(molecule_data)
        cca.write_state(config={"configurable": {"thread_id": f"{str(idx)}"}})

    # Save the results to a JSON file
    with open("llm_workflow.json", "w") as f:
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
    parser.add_argument("--start_index", type=int, default=0, help="Start index (default: 0)")
    parser.add_argument("--end_index", type=int, default=0, help="End index (default: 1)")
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.fname, args.start_index, args.end_index)
