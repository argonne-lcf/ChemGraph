from comp_chem_agent.agent.llm_agent import *
from comp_chem_agent.models.raspa import SimulationInput
cca = CompChemAgent()
#mess = cca.return_input("Create a simulation input file to calculate H2 adsorption in a MOF named IRMOF1.cif at 77K and 100 bar using a 2 3 4 unit cell")
query = "Create a simulation input file to calculate H2 adsorption in a MOF named IRMOF1.cif at 77K and 100 bar using a 2 3 4 unit cell"
mess = cca.return_input(query, SimulationInput)
#cca.run("Run geometry optimization using ASE for the molecule with the smiles c1ccccc1 using your available tools.")
#new_agent.run("Run geometry optimization using ASE for the molecule with the smiles c1ccccc1 using your available tools. Firt convert the SMILES string to AtomsData, then run geometry optimization using AtomsData input.")
"""
model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
base_url = "https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1"

# Load the access token
access_token_file = 'access_token.txt'
with open(access_token_file, 'r') as file:
    access_token = file.read().strip()

new_agent = llmagent(model_name=model_name, base_url=base_url, api_key=access_token)    
new_agent.run("Run geometry optimization for the molecule with the smiles c1ccccc1 using your available tools.")
#new_agent.run("What is the coordinates of the molecule with smiles c1ccccc1 using your tools.")
#new_agent.runq("Explain quantum computing")
"""

"""
model_name = "llama3.2"
new_agent = llmagent(model_name=model_name, temperature=0)    
#new_agent.runq("Explain quantum computing")
new_agent.run("Run geometry optimization for the molecule with the smiles c1ccccc1 using your available tools.")
"""
