import json
from comp_chem_agent.agent.llm_graph import llm_graph
from comp_chem_agent.utils.get_workflow_from_llm import get_workflow_from_state
import argparse
from reactions_dataset import reactions


def get_query(
    reaction: dict,
    query_name: str = "enthalpy",
    temperature: float = 298,
    pressure: float = 101325,
    method: str = "mace_mp",
) -> str:
    """Get query for LLM.

    Returns:
        _type_: _description_
    """
    reactants_str = " + ".join([f"{r['coefficient']} {r['name']}" for r in reaction["reactants"]])
    products_str = " + ".join([f"{p['coefficient']} {p['name']}" for p in reaction["products"]])

    reaction_equation = f"{reactants_str} -> {products_str}"
    query_dict = {
        "enthalpy": f"What is the reaction enthalpy of {reaction_equation}",
        "enthalpy_method": f"Calculate the reaction enthalpy for this chemical reaction, {reaction_equation} using {method} potential",
        "gibbs_free_energy": f"What is the Gibbs free energy of reaction for {reaction_equation}?",
        "gibbs_free_energy_method": f"What is the Gibbs free energy of reaction for {reaction_equation} using {method}?",
        "gibbs_free_energy_method_temperature": f"What is the Gibbs free energy of reaction for {reaction_equation} using {method} at {temperature}K?",
    }

    return query_dict.get(query_name, "Query not found")  # Returns the query or a default message


def main(n_reactions: int):
    """ """
    # Load SMILES data from the specified JSON file
    combined_data = []

    cca = llm_graph(
        model_name='gpt-4o-mini',
        workflow_type="single_agent_ase",
        structured_output=True,
        return_option="state",
    )

    # Iterate through the first n_structures molecules
    for idx, reaction in enumerate(reactions[:n_reactions]):
        print("********************************************")
        print(
            f"REACTION INDEX {reaction['reaction_index']}: REACTION NAME: {reaction['reaction_name']}"
        )
        print("********************************************")

        name = reaction["reaction_name"]
        index = reaction["reaction_index"]
        reactants = reaction["reactants"]
        products = reaction["products"]

        query = get_query(reaction, query_name="enthalpy_method", method="mace_mp")
        state = cca.run(query, config={"configurable": {"thread_id": f"{str(idx)}"}})

        llm_workflow = get_workflow_from_state(state)

        # Store results in a structured dictionary
        reaction_data = {
            "name": name,
            "index": index,
            "reactants": reactants,
            "products": products,
            "llm_workflow": llm_workflow,
        }
        combined_data.append(reaction_data)
        cca.write_state(config={"configurable": {"thread_id": f"{str(idx)}"}})

    # Save the results to a JSON file
    with open("llm_workflow.json", "w") as f:
        json.dump(combined_data, f, indent=4)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculate properties of a reaction.")
    parser.add_argument(
        "--n_reactions", type=int, default=2, help="Number of molecules to process (default: 2)"
    )
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.n_reactions)
