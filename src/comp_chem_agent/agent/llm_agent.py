import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from comp_chem_agent.tools.local_model_loader import load_ollama_model
from comp_chem_agent.models.raspa import SimulationInput
from comp_chem_agent.tools.coremof_utils import *
from comp_chem_agent.tools.ASE_tools import *
from comp_chem_agent.tools.alcf_loader import load_alcf_model
from comp_chem_agent.tools.openai_loader import load_openai_model
class CompChemAgent:
    def __init__(
            self,
            model_name="gpt-3.5-turbo",
            tools = None,
            prompt = None,
            base_url = None,
            api_key = None,
            temperature= 0            
    ):
        try:
            if model_name in ["gpt-3.5-turbo"]:
                llm = load_openai_model(model_name=model_name, temperature=temperature)
                print(f"Loaded {model_name}")
            elif model_name in ['llama3.2', "llama3.1"]:
                llm = load_ollama_model(model_name=model_name, temperature=temperature)
                print(f"Loaded {model_name}")
            else:
                llm = load_alcf_model(model_name=model_name, base_url=base_url, api_key=api_key)
                print(f"Loaded {model_name}")

        except Exception as e:
            print(e)
            print(f"Error with loading {model_name}")
        if prompt == None:
            system_message = "You are a helpful assistant."
            #system_message =  "You are a helpful assistant at extracting data of material databases."
        #tools = [smiles_to_xyz, geometry_optimization]
        tools = [smiles_to_atomsdata, geometry_optimization_ase]
        #self.langgraph_agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
        #tools = [get_files_in_directories, search_file_by_keyword, extract_coreid_and_refcode]
        self.graph = create_react_agent(llm, tools, state_modifier=system_message)
        self.llm = llm

    def run(self, query):
        inputs = {"messages": [("user", query)]}
        for s in self.graph.stream(inputs, stream_mode="values"):
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

    def runq(self, query):
        messages = self.llm.invoke(query)
        print(messages)
        return messages

    def return_input(self, query, simulation_class):
        structured_llm = self.llm.with_structured_output(simulation_class)
        messages = structured_llm.invoke(query)
        return messages
    
