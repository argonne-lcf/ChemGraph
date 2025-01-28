from comp_chem_agent.agent.llm_graph import *

cca = llm_graph()
query = "Run geometry optimization until reaching convergence for Naphthalene using EMT calculator and FIRE optimizer."
cca.run(query, workflow_type="ase")

#cca.run(query, workflow_type="geoopt")
