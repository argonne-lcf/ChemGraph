import json
from comp_chem_agent.agent.llm_graph import llm_graph
from manual_workflow import get_manual_workflow_result
from reactions_dataset import reaction_list
from query import get_query

# Parameters
reaction = reaction_list[0]  # Pick a reaction from reaction_list

# Results from running workflow
calculator = {"calculator_type": "nwchem", "xc": "b3lyp", "basis": "3-21g"}
result = get_manual_workflow_result(reaction_list[0], calculator=calculator)

if isinstance(result, dict):
    # Save manual workflow in a json file
    with open("manual_result.json", "w") as f:
        json.dump(result, f)
else:
    print("ERROR WITH RUNNING MANUAL WORKFLOW.")
    print(f"Error message:  {result}")

# Results from running CompChemAgent

cca = llm_graph(model_name='gpt-4o-mini', workflow_type="single_agent_ase")
# Have to adjust query name and method manually
query = get_query(reaction_list[0], query_name="enthalpy_method", method="B3LYP/3-21G")

cca.run(query, config={"configurable": {"thread_id": "1"}})

# Write LLM workflow under run_logs folder.
cca.write_state(config={"configurable": {"thread_id": "1"}})
