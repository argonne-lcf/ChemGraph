from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from comp_chem_agent.tools.openai_loader import load_openai_model
from langchain_core.tools import tool

from comp_chem_agent.tools.ASE_tools import *
import json
from comp_chem_agent.tools.alcf_loader import load_alcf_model
from comp_chem_agent.tools.local_model_loader import load_ollama_model
from comp_chem_agent.graphs.geoopt_workflow import construct_geoopt_graph
from comp_chem_agent.tools.xtb_tools import *
from comp_chem_agent.graphs.xtb_workflow import construct_xtb_graph

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
        self.llm = llm

    def _bind_tools(self, tools):
        if not tools:
            return self.llm
        try:
            self.llm_with_tools = self.llm.bind_tools(tools)    
            return self.llm_with_tools
        except AttributeError:
            raise AttributeError("The LLM model doesn't support tool binding")
        
    def _construct_workflow(self, workflow_type: str, tools=None):
        """
        Constructs a workflow graph based on the specified type
        
        Args:
            workflow_type (str): Type of workflow to construct ('geoopt', 'xtb', etc.)
            tools (list, optional): Custom tools to use. If None, uses default tools for the workflow
        
        Returns:
            Graph: Constructed workflow graph
        """
        workflow_map = {
            'geoopt': {
                'default_tools': [molecule_name_to_smiles, smiles_to_atomsdata, geometry_optimization_ase],
                'constructor': construct_geoopt_graph
            },
            'xtb': {
                'default_tools': [molecule_name_to_smiles, smiles_to_atomsdata, run_xtb_calculation],
                'constructor': construct_xtb_graph
            }
            # Add more workflows here as needed
        }
        if workflow_type not in workflow_map:
            raise ValueError(f"Unsupported workflow type: {workflow_type}. Available types: {list(workflow_map.keys())}")

        workflow = workflow_map[workflow_type]
        tools_to_use = tools if tools is not None else workflow['default_tools']
        self.llm_with_tools = self._bind_tools(tools=tools_to_use)

        return workflow['constructor'](tools_to_use, self.llm_with_tools)

    def run(self, query, workflow_type='geoopt', tools=None):
        """
        Runs the specified workflow with the given query
        
        Args:
            query (str): The user's input query
            workflow_type (str): Type of workflow to run ('geoopt' or 'xtb')
            tools (list, optional): Custom tools to use. If None, uses default tools for the workflow
        """
        # Construct the workflow graph
        workflow = self._construct_workflow(workflow_type, tools=tools)
        
        # Prepare the input format expected by the graph
        inputs = {"messages": [("user", query)]}
        
        # Execute the workflow and stream results
        for s in workflow.stream(inputs, stream_mode="values"):
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
