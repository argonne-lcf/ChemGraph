import json
from comp_chem_agent.agent.llm_graph import llm_graph
from comp_chem_agent.utils.get_workflow_from_llm import get_workflow_from_state
import argparse
import datetime
from inference_auth_token import get_access_token
from comp_chem_agent.prompt.qwen_prompt import single_agent_prompt, formatter_prompt

access_token = get_access_token()


def get_query(
    smiles: str,
    query_name: str = "smiles_to_coord",  # options: atomsdata, opt, vib
    method: str = "mace_mp",
) -> str:
    """Get query for a SMILES-related task for CompChemAgent

    Args:
        smiles (str): SMILES string.
        query_name (str, optional): Type of query. Defaults to "atomsdata". Options: "atomsdata", "opt", "vib", "opt_method" and "vib_method".
        method (str, optional): The method/level of theory for CompChemAgent to run simulation. Defaults to "mace_mp".

    Returns:
        str: formatted query.
    """
    query_dict = {
        "smiles_to_coord": f"Provide the XYZ coordinates corresponding to this SMILES string: {smiles}",
        "smiles_to_opt": f"Perform geometry optimization for this SMILES string {smiles} using {method}",
        "smiles_to_vib": f"Run vibrational frequency calculation for this SMILES string {smiles} using {method}",
        "smiles_to_enthalpy": f"Calculate the enthalpy of this SMILES string {smiles} using {method}",
        "smiles_to_gibbs": f"Calculate the Gibbs free energy of this SMILES string {smiles} using {method} at T=800K",
        "smiles_to_opt_file": f"Perform geometry optimization for this SMILES string {smiles} using {method}. Save the optimized coordinate in an XYZ file named {smiles}.xyz",
    }

    return query_dict.get(query_name, "Query not found")  # Returns the query or a default message


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

    combined_data = {}

    cca = llm_graph(
        model_name='Qwen/Qwen2.5-14B-Instruct',
        api_key=access_token,
        base_url="https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
        workflow_type="single_agent_ase",
        structured_output=True,
        return_option="state",
        system_prompt=single_agent_prompt,
        formatter_prompt=formatter_prompt,
    )
    # Iterate through the first n_structures molecules
    for idx, molecule in enumerate(smiles_data[:n_structures]):
        print("********************************************")
        print(
            f"MOLECULE INDEX {molecule['index']}: MOLECULE SMILES: {molecule['smiles']} MOLECULE NAME: {molecule['name']}"
        )
        print("********************************************")

        smiles = molecule["smiles"]

        query = get_query(smiles, query_name="smiles_to_opt_file", method="mace_mp")
        state = cca.run(query, config={"configurable": {"thread_id": f"{str(idx)}"}})

        llm_workflow = get_workflow_from_state(state)

        # Store results in a structured dictionary
        state_data = cca.write_state(config={"configurable": {"thread_id": f"{str(idx)}"}})

        combined_data[smiles] = {"llm_workflow": llm_workflow}
        combined_data[smiles]["metadata"] = state_data

    # Save the results to a JSON file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"llm_workflow_{timestamp}.json"

    # Save the results to a JSON file
    with open(filename, "w") as f:
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
        "--n_structures", type=int, default=15, help="Number of molecules to process (default: 30)"
    )
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.fname, args.n_structures)
