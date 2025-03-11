from comp_chem_agent.tools.openai_loader import load_openai_model
from comp_chem_agent.tools.alcf_loader import load_alcf_model
from comp_chem_agent.tools.local_model_loader import load_ollama_model
from comp_chem_agent.graphs.single_agent import construct_geoopt_graph
from comp_chem_agent.models.supported_models import (
    supported_openai_models,
    supported_ollama_models,
)

from comp_chem_agent.graphs.multi_agent import construct_multi_framework_graph
import logging

logger = logging.getLogger(__name__)


class llm_graph:
    def __init__(
        self,
        model_name="gpt-4o-mini",
        workflow_type="single_agent_ase",
        base_url=None,
        api_key=None,
        temperature=0,
    ):
        try:
            if model_name in supported_openai_models:
                llm = load_openai_model(model_name=model_name, temperature=temperature)
            elif model_name in supported_ollama_models:
                llm = load_ollama_model(model_name=model_name, temperature=temperature)
            else:
                llm = load_alcf_model(model_name=model_name, base_url=base_url, api_key=api_key)

        except Exception as e:
            logger.error(f"Exception thrown when loading {model_name}.")
            raise e

        self.workflow_type = workflow_type
        self.workflow_map = {
            "single_agent_ase": {
                "constructor": construct_geoopt_graph,
            },
            "multi_framework": {"constructor": construct_multi_framework_graph},
        }

        if workflow_type not in self.workflow_map:
            raise ValueError(
                f"Unsupported workflow type: {workflow_type}. Available types: {list(self.workflow_map.keys())}"
            )

        self.workflow = self.workflow_map[workflow_type]["constructor"](llm)

    def visualize(self):
        workflow = self.workflow

        import nest_asyncio
        from IPython.display import Image, display
        from langchain_core.runnables.graph import (
            CurveStyle,
            MermaidDrawMethod,
            NodeStyles,
        )

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
                    padding=6,
                )
            )
        )

    def get_state(self, config={"configurable": {"thread_id": "1"}}):
        return self.workflow.get_state(config).values["messages"]

    def run(
        self,
        query,
        config={"configurable": {"thread_id": "1"}},
    ):
        """
        Runs the specified workflow with the given query

        Args:
            query (str): The user's input query
            workflow_type (str): Type of workflow to run ('geoopt' or 'xtb')
        """
        try:
            # Construct the workflow graph
            workflow = self.workflow

            if self.workflow_type == "single_agent_ase":
                inputs = {"messages": query}
                for s in workflow.stream(inputs, stream_mode="values", config=config):
                    message = s["messages"][-1]
                    if isinstance(message, tuple):
                        logger.info(message)
                        continue
                    else:
                        message.pretty_print()

            elif self.workflow_type == "multi_framework":
                inputs = {
                    "question": query,
                    "geometry_response": query,
                    "parameter_response": query,
                    "opt_response": query,
                }
                previous_lengths = {
                    "planner_response": 0,
                    "geometry_response": 0,
                    "parameter_response": 0,
                    "opt_response": 0,
                    "feedback_response": 0,
                    "router_response": 0,
                    "end_response": 0,
                    "regular_response": 0,
                }

                for s in workflow.stream(inputs, stream_mode="values", config=config):
                    # Check if the lengths of the message lists have changed
                    for key in previous_lengths.keys():
                        current_length = len(s.get(key, []))

                        if current_length > previous_lengths[key]:
                            # If the length has increased, process the newest message
                            new_message = s[key][-1]  # Get the newest message
                            logger.info(f"New message in {key}:")

                            if isinstance(new_message, tuple):
                                logger.info(new_message)
                            else:
                                new_message.pretty_print()

                            # Update the previous length
                            previous_lengths[key] = current_length

            else:
                logger.error(
                    f"Workflow {self.workflow_type} is not supported. Please select either multi_agent_ase or single_agent_ase"
                )
                raise ValueError(f"Workflow {self.workflow_type} is not supported")

        except Exception as e:
            logger.error(f"Error running workflow {self.workflow_type}: {str(e)}")
            raise
