import json
from comp_chem_agent.agent.llm_graph import llm_graph
from comp_chem_agent.utils.get_workflow_from_llm import get_workflow_from_state
import argparse
from two_mol_per_task import get_query


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

    molecule_1 = smiles_data[0]
    molecule_2 = smiles_data[3]

    s1 = molecule_1["smiles"]
    n1 = molecule_1["name"]
    s2 = molecule_2["smiles"]
    n2 = molecule_2["name"]

    query = get_query(n1, n2, query_index=6)
    # query = get_query(n1, n2, query_index=1)
    state = cca.run(query, config={"configurable": {"thread_id": "1"}})

    llm_workflow = get_workflow_from_state(state)

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
