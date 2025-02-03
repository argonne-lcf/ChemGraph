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
from comp_chem_agent.prompt.prompt import ase_prompt, geometry_input_prompt
from comp_chem_agent.graphs.ASE_geoopt import construct_ase_graph
class llm_graph:
    def __init__(
        self,
        model_name="gpt-4o-mini",
        tools = None,
        prompt = None,
        base_url = None,
        api_key = None,
        temperature= 0            
    ):
        try:
            if model_name in ["gpt-4o-mini"]:
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

        self.workflow_map = {
            'geoopt': {
                'default_tools': [molecule_name_to_smiles, smiles_to_atomsdata, geometry_optimization, save_atomsdata_to_file, file_to_atomsdata],
                'constructor': construct_geoopt_graph,
                'prompt': ase_prompt
            },
            'xtb': {
                'default_tools': [molecule_name_to_smiles, smiles_to_atomsdata, run_xtb_calculation],
                'constructor': construct_xtb_graph
            },
            'ase': {
                'default_tools': [molecule_name_to_smiles, smiles_to_atomsdata, file_to_atomsdata],
                'constructor': construct_ase_graph,
                prompt: geometry_input_prompt
            }
        }

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
        if workflow_type not in self.workflow_map:
            raise ValueError(f"Unsupported workflow type: {workflow_type}. Available types: {list(self.workflow_map.keys())}")

        workflow = self.workflow_map[workflow_type]
        tools_to_use = tools if tools is not None else workflow['default_tools']
        self.llm_with_tools = self._bind_tools(tools=tools_to_use)
        if workflow_type != 'ase':
            return workflow['constructor'](tools_to_use, self.llm_with_tools)
        else:
            return workflow['constructor'](self.llm)
        
    def visualize(self, workflow_type):
        if workflow_type == 'ase':
            workflow = self._construct_workflow(workflow_type)
        else:
            raise ValueError("Workflow {workflow} visualization is not supported yet.")
        
        import nest_asyncio
        from IPython.display import Image, display
        from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

        nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions

        display(
            Image(
                workflow.get_graph().draw_mermaid_png(
                    curve_style=CurveStyle.LINEAR,
                    node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
                    wrap_label_n_words=9,
                    output_file_path=None,
                    draw_method=MermaidDrawMethod.PYPPETEER,
                    background_color="white",
                    padding=10,
                )
            )
        )

    def run(self, query, workflow_type='geoopt', tools=None, config = {"configurable": {"thread_id": "1"}}):
        """
        Runs the specified workflow with the given query
        
        Args:
            query (str): The user's input query
            workflow_type (str): Type of workflow to run ('geoopt' or 'xtb')
            tools (list, optional): Custom tools to use. If None, uses default tools for the workflow
        """
        # Construct the workflow graph
        if workflow_type == 'ase':
            workflow = self._construct_workflow(workflow_type)

        else:
            workflow = self._construct_workflow(workflow_type, tools=tools)
        if workflow_type=="geoopt":
            system_message = self.workflow_map[workflow_type]['prompt']

            # Prepare the input format expected by the graph
            inputs = {"messages": [("user", query), ("system", system_message)]}
                    # Execute the workflow and stream results
            for s in workflow.stream(inputs, stream_mode="values", config=config):
                message = s["messages"][-1]
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()
        else:
            inputs = {"question": query, "geometry_response": query, "parameter_response": query, "opt_response": query}
            previous_lengths = {
                "planner_response": 0,
                "geometry_response": 0,
                "parameter_response": 0,
                "opt_response": 0,
                "feedback_response": 0,
                "router_response": 0,
                "end_response": 0,
                "regular_response": 0
            }

            for s in workflow.stream(inputs, stream_mode="values", config=config):
                # Check if the lengths of the message lists have changed
                for key in previous_lengths.keys():
                    current_length = len(s.get(key, []))
                    
                    if current_length > previous_lengths[key]:
                        # If the length has increased, process the newest message
                        new_message = s[key][-1]  # Get the newest message
                        print(f"New message in {key}:")
                        
                        if isinstance(new_message, tuple):
                            print(new_message)
                        else:
                            new_message.pretty_print()
                        
                        # Update the previous length
                        previous_lengths[key] = current_length
