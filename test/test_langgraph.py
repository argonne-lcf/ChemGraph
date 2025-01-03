from comp_chem_agent.agent.llm_graph import *

cca = llm_graph()
query = "Run geometry optimization using ASE for acetic acid"
cca.run(query, workflow_type="geoopt")
