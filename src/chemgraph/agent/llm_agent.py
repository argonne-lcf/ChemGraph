import datetime
import os
from chemgraph.tools.openai_loader import load_openai_model
from chemgraph.tools.alcf_loader import load_alcf_model
from chemgraph.tools.local_model_loader import load_ollama_model
from chemgraph.tools.anthropic_loader import load_anthropic_model
from chemgraph.tools.gemini_loader import load_gemini_model
from chemgraph.models.supported_models import (
    supported_openai_models,
    supported_ollama_models,
    supported_anthropic_models,
    supported_alcf_models,
    supported_argo_models,
    supported_gemini_models,
)
from chemgraph.prompt.single_agent_prompt import (
    single_agent_prompt,
    formatter_prompt,
    report_prompt,
)
from chemgraph.prompt.multi_agent_prompt import (
    executor_prompt,
    formatter_multi_prompt,
    aggregator_prompt,
    planner_prompt,
)
from chemgraph.graphs.single_agent import construct_single_agent_graph
from chemgraph.graphs.python_relp_agent import construct_relp_graph
from chemgraph.graphs.multi_agent import contruct_multi_agent_graph
from chemgraph.graphs.graspa_agent import construct_graspa_graph
from chemgraph.graphs.mock_agent import construct_mock_agent_graph

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
    if isinstance(state, (int, float, bool)) or state is None:
        return state
    elif isinstance(state, list):
        return [serialize_state(item) for item in state]
    elif isinstance(state, dict):
        return {key: serialize_state(value) for key, value in state.items()}
    elif hasattr(state, "__dict__"):
        return {key: serialize_state(value) for key, value in state.__dict__.items()}
    else:
        return str(state)


class ChemGraph:
    """A graph-based workflow for LLM-powered computational chemistry tasks.

    This class manages different types of workflows for computational chemistry tasks,
    supporting various LLM models and workflow types.

    Parameters
    ----------
    model_name : str, optional
        Name of the language model to use, by default "gpt-4o-mini"
    workflow_type : str, optional
        Type of workflow to use. Options:
        - "single_agent"
        - "multi_agent"
        - "python_relp"
        - "graspa_agent"
        by default "single_agent"
    base_url : str, optional
        Base URL for API calls, by default None
    api_key : str, optional
        API key for authentication, by default None
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
        workflow_type: str = "single_agent",
        base_url: str = None,
        api_key: str = None,
        system_prompt: str = single_agent_prompt,
        formatter_prompt: str = formatter_prompt,
        structured_output: bool = False,
        return_option: str = "last_message",
        recursion_limit: int = 50,
        planner_prompt: str = planner_prompt,
        executor_prompt: str = executor_prompt,
        aggregator_prompt: str = aggregator_prompt,
        formatter_multi_prompt: str = formatter_multi_prompt,
        generate_report: bool = False,
        report_prompt: str = report_prompt,
    ):
        try:
            # Use hardcoded optimal values for tool calling
            temperature = 0.0  # Deterministic responses
            max_tokens = 4000  # Sufficient for most tasks
            top_p = 1.0  # No nucleus sampling filtering
            frequency_penalty = 0.0  # No repetition penalty
            presence_penalty = 0.0  # No presence penalty

            if (
                model_name in supported_openai_models
                or model_name in supported_argo_models
            ):
                llm = load_openai_model(
                    model_name=model_name, temperature=temperature, base_url=base_url
                )
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
            elif model_name in supported_gemini_models:
                llm = load_gemini_model(
                    model_name=model_name, api_key=api_key, temperature=temperature
                )

            else:  # Assume it might be a vLLM or other custom OpenAI-compatible endpoint
                import os

                # Use environment variables for vLLM base_url and a dummy api_key if not provided
                # These would be set by docker-compose for the jupyter_lab service
                vllm_base_url = os.getenv("VLLM_BASE_URL", base_url)
                # ChatOpenAI requires an api_key, even if the endpoint doesn't use it.
                vllm_api_key = os.getenv(
                    "OPENAI_API_KEY", api_key if api_key else "dummy_vllm_key"
                )

                if vllm_base_url:
                    logger.info(
                        f"Attempting to load model '{model_name}' from custom endpoint: {vllm_base_url}"
                    )
                    from langchain_openai import ChatOpenAI

                    llm = ChatOpenAI(
                        model=model_name,
                        temperature=temperature,
                        base_url=vllm_base_url,
                        api_key=vllm_api_key,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                    )
                    logger.info(
                        f"Successfully initialized ChatOpenAI for model '{model_name}' at {vllm_base_url}"
                    )
                else:
                    logger.error(
                        f"Model '{model_name}' is not in any supported list and no VLLM_BASE_URL/base_url provided."
                    )
                    raise ValueError(
                        f"Unsupported model or missing base URL for: {model_name}"
                    )

        except Exception as e:
            logger.error(f"Exception thrown when loading {model_name}: {str(e)}")
            raise e

        self.workflow_type = workflow_type
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.formatter_prompt = formatter_prompt
        self.structured_output = structured_output
        self.generate_report = generate_report
        self.report_prompt = report_prompt
        self.return_option = return_option
        self.recursion_limit = recursion_limit
        self.planner_prompt = planner_prompt
        self.executor_prompt = executor_prompt
        self.aggregator_prompt = aggregator_prompt
        self.formatter_multi_prompt = formatter_multi_prompt
        self.workflow_map = {
            "single_agent": {"constructor": construct_single_agent_graph},
            "multi_agent": {"constructor": contruct_multi_agent_graph},
            "python_relp": {"constructor": construct_relp_graph},
            "graspa": {"constructor": construct_graspa_graph},
            "mock_agent": {"constructor": construct_mock_agent_graph},
        }

        if workflow_type not in self.workflow_map:
            raise ValueError(
                f"Unsupported workflow type: {workflow_type}. Available types: {list(self.workflow_map.keys())}"
            )

        if self.workflow_type == "single_agent":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm,
                self.system_prompt,
                self.structured_output,
                self.formatter_prompt,
                self.generate_report,
                self.report_prompt,
            )
        elif self.workflow_type == "multi_agent":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm,
                planner_prompt=self.planner_prompt,
                aggregator_prompt=self.aggregator_prompt,
                executor_prompt=self.executor_prompt,
                formatter_prompt=self.formatter_multi_prompt,
                structured_output=self.structured_output,
            )
        elif self.workflow_type == "python_relp":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm,
                self.system_prompt,
            )
        elif self.workflow_type == "graspa":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm,
                self.system_prompt,
                self.structured_output,
                self.formatter_prompt,
            )
        elif self.workflow_type == "mock_agent":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm=llm,
                system_prompt=self.system_prompt,
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
                self.workflow.get_graph().draw_mermaid_png(
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
        return self.workflow.get_state(config).values

    def write_state(
        self,
        config={"configurable": {"thread_id": "1"}},
        file_path: str = None,
        file_name: str = None,
    ):
        """Write log of ChemGraph run to a JSON file, including workflow-specific prompts.

        Parameters
        ----------
        config : dict, optional
            Workflow config, must include 'configurable.thread_id'
        file_path : str, optional
            Full path to output file. If not provided, writes to 'run_logs/state_thread_<thread_id>_<timestamp>.json'
        file_name : str, optional
            Optional filename to use if file_path is not provided

        Returns
        -------
        dict or str
            Dictionary of metadata if successful, or "Error" if failed.
        """
        import json
        import subprocess

        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            thread_id = config["configurable"]["thread_id"]
            if not file_path:
                os.makedirs("run_logs", exist_ok=True)
                if not file_name:
                    file_name = f"state_thread_{thread_id}_{timestamp}.json"
                file_path = os.path.join("run_logs", file_name)

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

            # Base log info
            output_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "model_name": self.model_name,
                "thread_id": thread_id,
                "git_commit": git_commit,
                "state": serialized_state,
            }

            # Add prompts depending on workflow_type
            if self.workflow_type in {"single_agent", "graspa", "python_relp"}:
                output_data.update({
                    "system_prompt": self.system_prompt,
                    "formatter_prompt": self.formatter_prompt,
                })
            elif self.workflow_type == "mock_agent":
                output_data.update({
                    "system_prompt": self.system_prompt,
                })
            elif self.workflow_type == "multi_agent":
                output_data.update({
                    "planner_prompt": self.planner_prompt,
                    "executor_prompt": self.executor_prompt,
                    "aggregator_prompt": self.aggregator_prompt,
                    "formatter_prompt": self.formatter_multi_prompt,
                })
            else:
                output_data.update({
                    "system_prompt": "unknown",
                    "formatter_prompt": "unknown",
                })

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
        import uuid

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

            if self.workflow_type in {"single_agent", "graspa", "python_relp", "mock_agent"}:
                inputs = {"messages": query}

                prev_messages = []

                for s in workflow.stream(inputs, stream_mode="values", config=config):
                    if "messages" in s and s["messages"] != prev_messages:
                        new_message = s["messages"][-1]
                        new_message.pretty_print()
                        logger.info(new_message)
                        prev_messages = s["messages"]
                log_id = str(uuid.uuid4())
                log_dir = os.path.join("logs", log_id)
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, "state.json")
                self.write_state(config=config, file_path=log_path)

                if self.return_option == "last_message":
                    return s["messages"][-1]
                elif self.return_option == "state":
                    return serialize_state(self.get_state(config=config))
                else:
                    raise ValueError(
                        f"Return option {self.return_option} is not supported. Only supports 'last_message' or 'state'."
                    )
            elif self.workflow_type == "multi_agent":
                inputs = {"messages": query}
                prev_messages = []

                for s in workflow.stream(inputs, stream_mode="values", config=config):
                    if "messages" in s and s["messages"] != prev_messages:
                        new_message = s["messages"][-1]
                        new_message.pretty_print()
                        logger.info(new_message)
                        prev_messages = s["messages"]

                log_id = str(uuid.uuid4())
                log_dir = os.path.join("logs", log_id)
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, "state.json")
                self.write_state(config=config, file_path=log_path)

                if self.return_option == "last_message":
                    return s["messages"][-1]
                elif self.return_option == "state":
                    return serialize_state(self.get_state(config=config))
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
