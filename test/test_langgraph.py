from comp_chem_agent.agent.llm_graph import *

cca = llm_graph()
query = "Run geometry optimization for Toluene using ASE and mace_mp."
#query = "What is the capital of Vietnam?"
cca.run(query, workflow_type="opt_vib")