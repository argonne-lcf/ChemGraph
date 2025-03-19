from comp_chem_agent.agent.llm_graph import llm_graph

cca = llm_graph(model_name='gpt-4o', workflow_type="single_agent_ase")
# query = "Run geometry optimization for water using mace_mp."
# query = "Calculate the Gibbs free energy for this reaction using mace_mp: water + carbon monoxide -> hydrogen + carbon dioxide"
query = "Calculate the Gibbs free energy for this reaction using mace_mp: Glucose -> 2*Ethanol + 2*Carbon dioxide"

cca.run(query, config={"configurable": {"thread_id": "2"}})
# cca.write_state(config={"configurable": {"thread_id": "1"}})
