import json
from comp_chem_agent.agent.llm_graph import llm_graph
from manual_workflow import (
    get_atomsdata_from_smiles,
    get_geometry_optimization_from_smiles,
    get_vibrational_frequencies_from_smiles,
)
from query import get_query
from comp_chem_agent.utils.get_workflow_from_llm import get_workflow_from_state

# Load a smiles string from smiles_data.json
with open("smiles_data.json", "r") as f:
    smiles_data = json.load(f)
smiles = smiles_data[5]["smiles"]

calculator = {"calculator_type": "mace_mp"}

# result = get_atomsdata_from_smiles(smiles)  # For evaluating conversion between smiles to atomsdata
# result = get_geometry_optimization_from_smiles(smiles, calculator)  # For evaluating running geometry optimization using SMILES
result = get_vibrational_frequencies_from_smiles(
    smiles,
    calculator,
)  # For evaluating running vibrational frequencies using SMILES

if isinstance(result, dict):
    # Save manual workflow in a json file
    with open("manual_result.json", "w") as f:
        json.dump(result, f, indent=4)
else:
    print("ERROR WITH RUNNING MANUAL WORKFLOW.")
    print(f"Error message:  {result}")
cca = llm_graph(
    model_name='gpt-4o',
    workflow_type="single_agent_ase",
    structured_output=True,
    return_option="state",
)
# Have to adjust query name and method manually
query = get_query(smiles, query_name="vib_method", method="mace_mp")
result = cca.run(query, config={"configurable": {"thread_id": "1"}})

workflow_dict = get_workflow_from_state(result)

# Store just the workflow
with open('llm_result.json', 'w') as f:
    json.dump(workflow_dict, f, indent=4)

# Store the state
cca.write_state(config={"configurable": {"thread_id": "1"}})
