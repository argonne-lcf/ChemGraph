from comp_chem_agent.agent.llm_graph import llm_graph

cca = llm_graph(model_name='gpt-4o-mini', workflow_type="single_agent_ase")
query = "Run geometry optimization for water using mace_mp."
cca.run(query, config={"configurable": {"thread_id": "1"}})
cca.write_state(config={"configurable": {"thread_id": "1"}})
