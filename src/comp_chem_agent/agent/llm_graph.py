from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from comp_chem_agent.tools.openai_loader import load_openai_model
from langchain_core.tools import tool

from comp_chem_agent.tools.ASE_tools import *
import json

from comp_chem_agent.graphs.geoopt_workflow import *
class llm_graph:
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
        tools = [smiles_to_atomsdata, geometry_optimization_ase]
        llm_with_tools = llm.bind_tools(tools)
        self.llm_with_tools = llm_with_tools

    def geo_opt_graph(self):        # Return the LLMs with tools graph
        tools = [smiles_to_atomsdata, geometry_optimization_ase]
        return construct_geoopt_graph(tools, self.llm_with_tools)
