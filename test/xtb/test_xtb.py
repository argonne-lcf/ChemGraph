from comp_chem_agent.agent.llm_graph import *

cca = llm_graph()
#query = "Run geometry optimization using ASE for acetic acid"

query = "The directory already contains the test.xyz file. Run XTB calculations."
cca.run(query, workflow_type="xtb")
