from comp_chem_agent.tools.openai_loader import load_openai_model
from comp_chem_agent.tools.alcf_loader import load_alcf_model
from comp_chem_agent.tools.local_model_loader import load_ollama_model
from comp_chem_agent.graphs.single_agent import construct_geoopt_graph
from comp_chem_agent.models.supported_models import (
    supported_openai_models,
    supported_ollama_models,
)
from comp_chem_agent.prompt.single_agent_prompt import single_agent_prompt
from comp_chem_agent.graphs.multi_agent import construct_multi_framework_graph
from comp_chem_agent.graphs.python_relp_agent import construct_relp_graph
import logging

logger = logging.getLogger(__name__)


def serialize_state(state):
    """Convert non-serializable objects in state to a JSON-friendly format."""
    if isinstance(state, list):
        return [serialize_state(item) for item in state]
    elif isinstance(state, dict):
        return {key: serialize_state(value) for key, value in state.items()}
    elif hasattr(state, "__dict__"):
        return {key: serialize_state(value) for key, value in state.__dict__.items()}
    else:
        return str(state)


class llm_graph:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        workflow_type: str = "single_agent_ase",
        base_url: str = None,
        api_key: str = None,
        temperature: float = 0,
        system_prompt: str = single_agent_prompt,
        structured_output: bool = False,
        return_option: str = "last_message",
        recursion_limit: int = 25,
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
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.structured_output = structured_output
        self.return_option = return_option
        self.recursion_limit = recursion_limit
        self.workflow_map = {
            "single_agent_ase": {
                "constructor": construct_geoopt_graph,
            },
            "multi_framework": {"constructor": construct_multi_framework_graph},
            "python_relp": {"constructor": construct_relp_graph},
        }

        if workflow_type not in self.workflow_map:
            raise ValueError(
                f"Unsupported workflow type: {workflow_type}. Available types: {list(self.workflow_map.keys())}"
            )

        self.workflow = self.workflow_map[workflow_type]["constructor"](
            llm, self.system_prompt, self.structured_output
        )

    def visualize(self):
        """Visualize the LangGraph graph structure."""
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
        """Get the current state.

        Args:
            config (dict, optional): Config of the conversation. Defaults to {"configurable": {"thread_id": "1"}}.
        """

        return self.workflow.get_state(config).values["messages"]

    def write_state(self, config={"configurable": {"thread_id": "1"}}, output_dir="run_logs"):
        """Write log of CCA run to a file.

        Args:
            config (dict, optional): Config of the conversation to save. Defaults to "{"configurable": {"thread_id": "1"}}"
            output_dir (str, optional): Output directory to save log. Defaults to "run_logs".

        Returns:
            0: Save file successfully
            1: Encounter error
        """
        import datetime
        import os
        import json
        import subprocess

        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            os.makedirs(output_dir, exist_ok=True)
            thread_id = config["configurable"]["thread_id"]
            file_name = f"state_{thread_id}_{timestamp}.json"
            file_path = os.path.join(output_dir, file_name)

            state = self.get_state(config=config)

            serialized_state = serialize_state(state)
            try:
                git_commit = (
                    subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
                )
            except subprocess.CalledProcessError:
                git_commit = "unknown"

            output_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "model_name": self.model_name,
                "system_prompt": self.system_prompt,
                "state": serialized_state,
                "thread_id": thread_id,
                "git_commit": git_commit,
            }
            with open(file_path, "w", encoding="utf-8") as json_file:
                json.dump(output_data, json_file, indent=4)
            return output_data

        except Exception as e:
            print("Error with write_state: ", str(e))
            return "Error"

    def run(self, query: str, config=None):
        """
        Runs the specified workflow with the given query.

        Args:
            query (str): The user's input query
            config (dict, optional): Configuration dictionary.
        """
        try:
            if config is None:
                config = {}
            if not isinstance(config, dict):
                raise TypeError(f"`config` must be a dictionary, got {type(config).__name__}")
            config.setdefault("configurable", {}).setdefault("thread_id", "1")
            config["recursion_limit"] = self.recursion_limit

            # Construct the workflow graph
            workflow = self.workflow

            if self.workflow_type == "single_agent_ase" or self.workflow_type == "python_relp":
                inputs = {"messages": query}
                for s in workflow.stream(inputs, stream_mode="values", config=config):
                    message = s["messages"][-1]
                    if isinstance(message, tuple):
                        logger.info(message)
                        continue
                    else:
                        message.pretty_print()
                if self.return_option == "last_message":
                    return s["messages"][-1]
                elif self.return_option == "state":
                    return s["messages"]
                else:
                    raise ValueError(
                        f"Return option {self.return_option} is not supported. Only supports 'last_message' or 'state'."
                    )
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
