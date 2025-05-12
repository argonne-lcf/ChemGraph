from comp_chem_agent.tools.openai_loader import load_openai_model
from comp_chem_agent.tools.alcf_loader import load_alcf_model
from comp_chem_agent.tools.local_model_loader import load_ollama_model
from comp_chem_agent.tools.anthropic_loader import load_anthropic_model
from comp_chem_agent.graphs.single_agent import construct_geoopt_graph
from comp_chem_agent.models.supported_models import (
    supported_openai_models,
    supported_ollama_models,
    supported_anthropic_models,
    supported_alcf_models,
)
from comp_chem_agent.prompt.single_agent_prompt import (
    single_agent_prompt,
    formatter_prompt,
)
from comp_chem_agent.graphs.multi_agent import construct_multi_framework_graph
from comp_chem_agent.graphs.python_relp_agent import construct_relp_graph
from comp_chem_agent.graphs.chemgraph_manager_worker import (
    contruct_manager_worker_graph,
)
import logging

logger = logging.getLogger(__name__)


def serialize_state(state):
    """Convert non-serializable objects in state to a JSON-friendly format.

    Parameters
    ----------
    state : Any
        The state object to be serialized. Can be a list, dict, or object with __dict__

    Returns
    -------
    Any
        A JSON-serializable version of the input state
    """
    if isinstance(state, list):
        return [serialize_state(item) for item in state]
    elif isinstance(state, dict):
        return {key: serialize_state(value) for key, value in state.items()}
    elif hasattr(state, "__dict__"):
        return {key: serialize_state(value) for key, value in state.__dict__.items()}
    else:
        return str(state)


class llm_graph:
    """A graph-based workflow for LLM-powered computational chemistry tasks.

    This class manages different types of workflows for computational chemistry tasks,
    supporting various LLM models and workflow types.

    Parameters
    ----------
    model_name : str, optional
        Name of the language model to use, by default "gpt-4o-mini"
    workflow_type : str, optional
        Type of workflow to use. Options:
        - "single_agent_ase"
        - "multi_framework"
        - "multi_agent_ase"
        - "python_relp"
        - "manager_worker"
        by default "single_agent_ase"
    base_url : str, optional
        Base URL for API calls, by default None
    api_key : str, optional
        API key for authentication, by default None
    temperature : float, optional
        Temperature parameter for model generation, by default 0
    system_prompt : str, optional
        System prompt for the language model, by default single_agent_prompt
    formatter_prompt : str, optional
        Prompt for formatting output, by default formatter_prompt
    structured_output : bool, optional
        Whether to use structured output, by default False
    return_option : str, optional
        What to return from the workflow. Options:
        - "last_message"
        - "state"
        by default "last_message"
    recursion_limit : int, optional
        Maximum number of recursive steps in the workflow, by default 50

    Raises
    ------
    ValueError
        If the workflow_type is not supported
    Exception
        If there is an error loading the specified model
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        workflow_type: str = "single_agent_ase",
        base_url: str = None,
        api_key: str = None,
        temperature: float = 0,
        system_prompt: str = single_agent_prompt,
        formatter_prompt: str = formatter_prompt,
        structured_output: bool = False,
        return_option: str = "last_message",
        recursion_limit: int = 50,
    ):
        try:
            if model_name in supported_openai_models:
                llm = load_openai_model(model_name=model_name, temperature=temperature)
            elif model_name in supported_ollama_models:
                llm = load_ollama_model(model_name=model_name, temperature=temperature)
            elif model_name in supported_alcf_models:
                llm = load_alcf_model(
                    model_name=model_name, base_url=base_url, api_key=api_key
                )
            elif model_name in supported_anthropic_models:
                llm = load_anthropic_model(
                    model_name=model_name, api_key=api_key, temperature=temperature
                )

        except Exception as e:
            logger.error(f"Exception thrown when loading {model_name}.")
            raise e

        self.workflow_type = workflow_type
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.formatter_prompt = formatter_prompt
        self.structured_output = structured_output
        self.return_option = return_option
        self.recursion_limit = recursion_limit
        self.workflow_map = {
            "single_agent_ase": {
                "constructor": construct_geoopt_graph,
            },
            "multi_framework": {"constructor": construct_multi_framework_graph},
            "multi_agent_ase": {"constructor": construct_multi_framework_graph},
            "python_relp": {"constructor": construct_relp_graph},
            "manager_worker": {"constructor": contruct_manager_worker_graph},
        }

        if workflow_type not in self.workflow_map:
            raise ValueError(
                f"Unsupported workflow type: {workflow_type}. Available types: {list(self.workflow_map.keys())}"
            )

        if self.workflow_type != "manager_worker":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm, self.system_prompt, self.structured_output, self.formatter_prompt
            )
        else:
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm, structured_output=self.structured_output
            )

    def visualize(self):
        """Visualize the LangGraph graph structure.

        This method creates and displays a visual representation of the workflow graph
        using Mermaid diagrams. The visualization is shown in Jupyter notebooks.

        Notes
        -----
        Requires IPython and nest_asyncio to be installed.
        The visualization uses Mermaid diagrams with custom styling.
        """
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
                    node_colors=NodeStyles(
                        first="#ffdfba", last="#baffc9", default="#fad7de"
                    ),
                    wrap_label_n_words=9,
                    output_file_path=None,
                    draw_method=MermaidDrawMethod.PYPPETEER,
                    background_color="white",
                    padding=6,
                )
            )
        )

    def get_state(self, config={"configurable": {"thread_id": "1"}}):
        """Get the current state of the workflow.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary containing thread information,
            by default {"configurable": {"thread_id": "1"}}

        Returns
        -------
        list
            List of messages in the current state
        """
        return self.workflow.get_state(config).values["messages"]

    def write_state(
        self, config={"configurable": {"thread_id": "1"}}, output_dir="run_logs"
    ):
        """Write log of CCA run to a file.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary containing thread information,
            by default {"configurable": {"thread_id": "1"}}
        output_dir : str, optional
            Output directory to save log, by default "run_logs"

        Returns
        -------
        dict or str
            If successful, returns a dictionary containing:
            - timestamp: ISO format timestamp
            - model_name: Name of the model used
            - system_prompt: The system prompt used
            - state: Serialized state
            - thread_id: Thread identifier
            - git_commit: Current git commit hash
            If error occurs, returns "Error"

        Notes
        -----
        The output file is saved as a JSON file with format:
        state_{thread_id}_{timestamp}.json
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
                    subprocess.check_output(["git", "rev-parse", "HEAD"])
                    .decode("utf-8")
                    .strip()
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
        """Run the specified workflow with the given query.

        Parameters
        ----------
        query : str
            The user's input query
        config : dict, optional
            Configuration dictionary for the workflow run, by default None

        Returns
        -------
        Any
            The result depends on return_option:
            - If "last_message": returns the last message from the workflow
            - If "state": returns the complete message state

        Raises
        ------
        TypeError
            If config is not a dictionary
        ValueError
            If return_option is not supported
        Exception
            If there is an error running the workflow
        """
        try:
            if config is None:
                config = {}
            if not isinstance(config, dict):
                raise TypeError(
                    f"`config` must be a dictionary, got {type(config).__name__}"
                )
            config.setdefault("configurable", {}).setdefault("thread_id", "1")
            config["recursion_limit"] = self.recursion_limit

            # Construct the workflow graph
            workflow = self.workflow

            if (
                self.workflow_type == "single_agent_ase"
                or self.workflow_type == "python_relp"
                or self.workflow_type == "manager_worker"
            ):
                inputs = {"messages": query}
                for s in workflow.stream(inputs, stream_mode="values", config=config):
                    message = s["messages"][-1]
                    logger.info(message)
                if self.return_option == "last_message":
                    return s["messages"][-1]
                elif self.return_option == "state":
                    return s["messages"]
                else:
                    raise ValueError(
                        f"Return option {self.return_option} is not supported. Only supports 'last_message' or 'state'."
                    )
            elif (
                self.workflow_type == "multi_framework"
                or self.workflow_type == "multi_agent_ase"
            ):
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
                            logger.info(new_message)

                            # Update the previous length
                            previous_lengths[key] = current_length
                # Return based on option for multi-agent as well
                if self.return_option == "last_message":
                    # Need to determine what the 'last' meaningful message is for multi-agent
                    # Returning the full state for now as a placeholder
                    return s
                elif self.return_option == "state":
                    return s
                else:
                    raise ValueError(
                        f"Return option {self.return_option} is not supported. Only supports 'last_message' or 'state'."
                    )

            else:
                logger.error(
                    f"Workflow {self.workflow_type} is not supported. Please select either multi_agent_ase or single_agent_ase"
                )
                raise ValueError(f"Workflow {self.workflow_type} is not supported")

        except Exception as e:
            logger.error(f"Error running workflow {self.workflow_type}: {str(e)}")
            raise
