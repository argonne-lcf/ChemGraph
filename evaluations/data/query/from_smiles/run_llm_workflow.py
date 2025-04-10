import json
from comp_chem_agent.agent.llm_graph import llm_graph
from comp_chem_agent.utils.get_workflow_from_llm import get_workflow_from_state
import argparse
from from_smiles import get_query


def main(fname: str, n_structures: int):
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
    for idx, molecule in enumerate(smiles_data[:n_structures]):
        print("********************************************")
        print(
            f"MOLECULE INDEX {molecule['index']}: MOLECULE SMILES: {molecule['smiles']} MOLECULE NAME: {molecule['name']}"
        )
        print("********************************************")

        smiles = molecule["smiles"]
        index = molecule["index"]
        name = molecule["name"]

        query = get_query(smiles, query_index=idx)
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
        default="pubchempy_molecules_max40.json",
        help="Path to the input SMILES JSON file (e.g., smiles_data.json)",
    )
    parser.add_argument(
        "--n_structures", type=int, default=6, help="Number of molecules to process (default: 10)"
    )
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.fname, args.n_structures)
