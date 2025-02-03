from comp_chem_agent.agent.llm_graph import *

cca = llm_graph()
query = "Run geometry optimization for Toluene."
#query = "What is the capital of Vietnam?"
cca.run(query, workflow_type="multi_agent_ase")

#cca.run(query, workflow_type="geoopt")
