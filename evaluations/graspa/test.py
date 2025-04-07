from comp_chem_agent.agent.llm_graph import llm_graph

cca = llm_graph(
    model_name='gpt-4o-mini',
    workflow_type="graspa_agent",
    structured_output=False,
    return_option="state",
)


query = "Run gRASPA calculation for CO2 adsorption in MOF-1 at 323K and 10000 Pascal using 1000 cycles. The DDEC6 charge output of MOF-1 is stored in the current directory ('./'). Store the gRASPA output in a test/ folder."

state = cca.run(query, config={"configurable": {"thread_id": 1}})

print(state)
