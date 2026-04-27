import asyncio
import datetime
import os
from typing import Callable, List, Optional
import uuid

from chemgraph.memory.store import SessionStore
from chemgraph.memory.schemas import SessionMessage
from chemgraph.models.openai import load_openai_model
from chemgraph.models.alcf_endpoints import load_alcf_model
from chemgraph.models.local_model import load_ollama_model
from chemgraph.models.anthropic import load_anthropic_model
from chemgraph.models.gemini import load_gemini_model
from chemgraph.models.groq import load_groq_model
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
    get_single_agent_prompt,
    formatter_prompt as default_formatter_prompt,
    report_prompt as default_report_prompt,
)
from chemgraph.prompt.multi_agent_prompt import (
    executor_prompt as default_executor_prompt,
    formatter_multi_prompt as default_formatter_multi_prompt,
    aggregator_prompt as default_aggregator_prompt,
    planner_prompt as default_planner_prompt,
)
from langgraph.types import Command
from langgraph.errors import GraphInterrupt

from chemgraph.graphs.single_agent import construct_single_agent_graph


from chemgraph.graphs.python_relp_agent import construct_relp_graph
from chemgraph.graphs.multi_agent import construct_multi_agent_graph
from chemgraph.graphs.graspa_agent import construct_graspa_graph
from chemgraph.graphs.mock_agent import construct_mock_agent_graph
from chemgraph.graphs.single_agent_mcp import construct_single_agent_mcp_graph
from chemgraph.graphs.graspa_mcp import construct_graspa_mcp_graph
from chemgraph.graphs.rag_agent import construct_rag_agent_graph
from chemgraph.graphs.single_agent_xanes import construct_single_agent_xanes_graph
from chemgraph.prompt.rag_prompt import rag_agent_prompt
from chemgraph.prompt.xanes_prompt import (
    xanes_single_agent_prompt as default_xanes_single_agent_prompt,
    xanes_formatter_prompt as default_xanes_formatter_prompt,
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
    max_retries : int, optional
        Maximum number of LLM retry attempts when an agent
        fails to parse its output, by default 1
    human_input_handler : callable, optional
        A callback ``f(question: str) -> str`` invoked when the graph
        pauses for human input (via ``interrupt()``).  Receives the
        question text and must return the human's answer as a string.
        If ``None`` (default), interrupts will propagate as
        ``GraphInterrupt`` exceptions.  The handler may also be an
        ``async`` callable.
    human_supervised : bool, optional
        Whether to include the ``ask_human`` tool so the agent can
        pause and request human input.  When ``False`` the tool is
        excluded from the tool list and the corresponding instruction
        is removed from the default system prompt, by default True.

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
        argo_user: str = None,
        system_prompt: str = single_agent_prompt,
        formatter_prompt: str = default_formatter_prompt,
        structured_output: bool = False,
        return_option: str = "last_message",
        recursion_limit: int = 50,
        planner_prompt: str = default_planner_prompt,
        executor_prompt: str = default_executor_prompt,
        aggregator_prompt: str = default_aggregator_prompt,
        formatter_multi_prompt: str = default_formatter_multi_prompt,
        generate_report: bool = False,
        report_prompt: str = default_report_prompt,
        support_structured_output: bool = True,
        tools: List = None,
        data_tools: List = None,
        session_store: Optional[SessionStore] = None,
        enable_memory: bool = True,
        memory_db_path: Optional[str] = None,
        log_dir: Optional[str] = None,
        max_retries: int = 1,
        human_input_handler: Optional[Callable[[str], str]] = None,
        human_supervised: bool = True,
    ):
        # Always generate a unique identifier for this instance
        self.uuid = str(uuid.uuid4())[:8]

        # Initialize log directory.  Explicit ``log_dir`` argument takes
        # precedence over the ``CHEMGRAPH_LOG_DIR`` environment variable,
        # which in turn takes precedence over the auto-generated default.
        self.log_dir = log_dir or os.environ.get("CHEMGRAPH_LOG_DIR")
        if not self.log_dir:
            # Create a new session log directory under cg_logs/
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Use abspath to ensure tools getting this env var have a full path
            self.log_dir = os.path.join(
                os.getcwd(), "cg_logs", f"session_{timestamp}_{self.uuid}"
            )
            os.makedirs(self.log_dir, exist_ok=True)
            # Set env var for tools to pick up
            os.environ["CHEMGRAPH_LOG_DIR"] = self.log_dir

        # Initialize session memory store
        if session_store is not None:
            self.session_store = session_store
        elif enable_memory:
            self.session_store = SessionStore(db_path=memory_db_path)
        else:
            self.session_store = None

        # Track whether session has been registered in the memory store
        self._session_created: bool = False
        self._session_title: Optional[str] = None

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
                openai_load_kwargs = {
                    "model_name": model_name,
                    "temperature": temperature,
                    "base_url": base_url,
                }
                if argo_user is not None:
                    openai_load_kwargs["argo_user"] = argo_user
                llm = load_openai_model(
                    **openai_load_kwargs,
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
            elif model_name.startswith("groq:"):
                llm = load_groq_model(
                    model_name=model_name, api_key=api_key, temperature=temperature
                )

            else:  # Assume it might be a vLLM or other custom OpenAI-compatible endpoint
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
        self.tools = tools
        self.data_tools = data_tools
        self.max_retries = max_retries
        self.human_input_handler = human_input_handler
        self.human_supervised = human_supervised

        # When human supervision is disabled and the caller is using the
        # default system prompt, strip the ask_human instructions so the
        # LLM is not told to call a tool that is unavailable.
        if not self.human_supervised and self.system_prompt == single_agent_prompt:
            self.system_prompt = get_single_agent_prompt(human_supervised=False)

        if model_name in supported_argo_models:
            self.support_structured_output = False
        else:
            self.support_structured_output = support_structured_output

        self.workflow_map = {
            "single_agent": {"constructor": construct_single_agent_graph},
            "multi_agent": {"constructor": construct_multi_agent_graph},
            "python_relp": {"constructor": construct_relp_graph},
            "graspa": {"constructor": construct_graspa_graph},
            "mock_agent": {"constructor": construct_mock_agent_graph},
            "single_agent_mcp": {"constructor": construct_single_agent_mcp_graph},
            "graspa_mcp": {"constructor": construct_graspa_mcp_graph},
            "rag_agent": {"constructor": construct_rag_agent_graph},
            "single_agent_xanes": {"constructor": construct_single_agent_xanes_graph},
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
                self.tools,
                max_retries=self.max_retries,
                human_supervised=self.human_supervised,
            )
        elif self.workflow_type == "multi_agent":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm,
                planner_prompt=self.planner_prompt,
                executor_prompt=self.executor_prompt,
                executor_tools=self.tools,
                structured_output=self.structured_output,
                formatter_prompt=self.formatter_multi_prompt,
                max_retries=self.max_retries,
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
        elif self.workflow_type == "single_agent_mcp":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm=llm,
                system_prompt=self.system_prompt,
                tools=self.tools,
            )
        elif self.workflow_type == "graspa_mcp":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm=llm,
                executor_tools=self.tools,
                analysis_tools=self.data_tools,
            )
        elif self.workflow_type == "rag_agent":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm=llm,
                system_prompt=self.system_prompt
                if self.system_prompt != single_agent_prompt
                else rag_agent_prompt,
                tools=self.tools,
            )
        elif self.workflow_type == "single_agent_xanes":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm,
                system_prompt=self.system_prompt
                if self.system_prompt != single_agent_prompt
                else default_xanes_single_agent_prompt,
                structured_output=self.structured_output,
                formatter_prompt=self.formatter_prompt
                if self.formatter_prompt != default_formatter_prompt
                else default_xanes_formatter_prompt,
                tools=self.tools,
            )

    def visualize(self, method: str = "ascii"):
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

        if method == "ascii":
            return self.workflow.get_graph().draw_ascii()
        else:
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
        config: dict = None,
        file_path: str = None,
        file_name: str = None,
    ):
        """Write log of ChemGraph run to a JSON file, including workflow-specific prompts.

        Parameters
        ----------
        config : dict, optional
            Workflow config, must include 'configurable.thread_id'
        file_path : str, optional
            Full path to output file. If not provided, writes to 'cg_logs/state_thread_<thread_id>_<timestamp>.json'
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
            if config is None:
                config = {"configurable": {"thread_id": "1"}}
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            thread_id = config["configurable"]["thread_id"]
            if not file_path:
                log_dir = getattr(self, "log_dir", None) or os.environ.get(
                    "CHEMGRAPH_LOG_DIR", "cg_logs"
                )
                os.makedirs(log_dir, exist_ok=True)
                if not file_name:
                    file_name = f"state_thread_{thread_id}_{self.uuid}_{timestamp}.json"
                file_path = os.path.join(log_dir, file_name)

            state = self.get_state(config=config)
            serialized_state = serialize_state(state)

            try:
                git_commit = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                    )
                    .decode("utf-8")
                    .strip()
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
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
            if self.workflow_type in {
                "single_agent",
                "single_agent_xanes",
                "graspa",
                "python_relp",
                "rag_agent",
            }:
                output_data.update(
                    {
                        "system_prompt": self.system_prompt,
                        "formatter_prompt": self.formatter_prompt,
                    }
                )

            elif self.workflow_type == "graspa_mcp":
                output_data.update(
                    {
                        "system_prompt": self.system_prompt,
                    }
                )

            elif self.workflow_type == "mock_agent":
                output_data.update(
                    {
                        "system_prompt": self.system_prompt,
                    }
                )
            elif self.workflow_type == "multi_agent":
                output_data.update(
                    {
                        "planner_prompt": self.planner_prompt,
                        "executor_prompt": self.executor_prompt,
                        "formatter_prompt": self.formatter_multi_prompt,
                    }
                )
            else:
                output_data.update(
                    {
                        "system_prompt": "unknown",
                        "formatter_prompt": "unknown",
                    }
                )

            with open(file_path, "w", encoding="utf-8") as json_file:
                json.dump(output_data, json_file, indent=4)
            return output_data

        except Exception as e:
            print("Error with write_state: ", str(e))
            return "Error"

    @property
    def session_id(self) -> str:
        """Current session ID (always available, derived from self.uuid)."""
        return self.uuid

    def _ensure_session(self, query: str) -> None:
        """Create a session record on first run if memory is enabled."""
        if self.session_store is None:
            return
        if self._session_created:
            return

        self._session_title = SessionStore.generate_title(query)
        self.session_store.create_session(
            session_id=self.uuid,
            model_name=self.model_name,
            workflow_type=self.workflow_type,
            title=self._session_title,
            log_dir=self.log_dir,
        )
        self._session_created = True
        logger.info(f"Created session {self.uuid}: {self._session_title}")

    def _save_messages_to_store(self, last_state: dict, query: str) -> None:
        """Extract messages from workflow state and persist to session store."""
        if self.session_store is None or not self._session_created:
            return

        try:
            messages_to_save = []
            state_messages = last_state.get("messages", [])

            for msg in state_messages:
                role = None
                content = ""
                tool_name = None

                if hasattr(msg, "type"):
                    # LangChain message objects
                    if msg.type == "human":
                        role = "human"
                    elif msg.type == "ai":
                        role = "ai"
                    elif msg.type == "tool":
                        role = "tool"
                        tool_name = getattr(msg, "name", None)
                    content = getattr(msg, "content", str(msg))
                elif isinstance(msg, dict):
                    role = msg.get("type") or msg.get("role")
                    content = msg.get("content", "")
                    tool_name = msg.get("name")

                if role and content:
                    messages_to_save.append(
                        SessionMessage(
                            role=role,
                            content=content,
                            tool_name=tool_name,
                        )
                    )

            self.session_store.save_messages(
                session_id=self.uuid,
                messages=messages_to_save,
                title=self._session_title,
            )
            logger.info(
                f"Saved {len(messages_to_save)} messages to session {self.uuid}"
            )
        except Exception as e:
            logger.warning(f"Failed to save messages to session store: {e}")

    def load_previous_context(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
    ) -> str:
        """Load context from a previous session as a summary string.

        This can be injected into the conversation to give the agent
        awareness of prior work.

        Parameters
        ----------
        session_id : str
            Previous session ID (or unique prefix).
        max_messages : int, optional
            Limit the number of messages included.

        Returns
        -------
        str
            Formatted context summary, or empty string if not found.
        """
        if self.session_store is None:
            logger.warning("Memory is disabled; cannot load previous context.")
            return ""
        return self.session_store.build_context_summary(session_id)

    async def _call_human_input_handler(self, question: str) -> str:
        """Invoke the human_input_handler, supporting both sync and async callables.

        Raises :class:`HumanInputRequired` when no handler is configured,
        allowing external callers (CLI, UI) to catch it, prompt the user,
        and resume the graph.
        """
        handler = self.human_input_handler
        if handler is None:
            raise HumanInputRequired(question)
        if asyncio.iscoroutinefunction(handler):
            return await handler(question)
        return handler(question)

    async def run(self, query: str, config=None, resume_from: Optional[str] = None):
        """
        Async-only runner. Requires `self.workflow.astream(...)`.
        Streams values, logs new messages, writes state, and returns according to
        `self.return_option` ("last_message" or "state").

        When the graph pauses for human input (via ``interrupt()``), the
        ``human_input_handler`` callback is invoked to obtain the user's
        response, and the graph is automatically resumed.  If no handler
        is configured, the ``GraphInterrupt`` exception propagates to the
        caller.

        Parameters
        ----------
        query : str
            The user query to execute.
        config : dict, optional
            LangGraph config with thread_id, etc.
        resume_from : str, optional
            Session ID to load context from. The previous conversation
            summary is prepended to the query.
        """

        def _validate_config(cfg):
            if cfg is None:
                cfg = {}
            if not isinstance(cfg, dict):
                raise TypeError(
                    f"`config` must be a dictionary, got {type(cfg).__name__}"
                )

            # Support top-level thread_id for convenience
            if "thread_id" in cfg:
                if "configurable" not in cfg:
                    cfg["configurable"] = {}
                cfg["configurable"]["thread_id"] = str(cfg["thread_id"])

            cfg.setdefault("configurable", {}).setdefault("thread_id", "1")
            cfg["recursion_limit"] = self.recursion_limit
            return cfg

        def _save_state_and_select_return(last_state, cfg):
            log_dir = self.log_dir
            if not log_dir:
                log_dir = "cg_logs"

            os.makedirs(log_dir, exist_ok=True)
            log_path = None
            self.write_state(config=cfg, file_path=log_path)

            if self.return_option == "last_message":
                return last_state["messages"][-1]
            elif self.return_option == "state":
                return serialize_state(self.get_state(config=cfg))
            else:
                raise ValueError(
                    f"Unsupported return_option: {self.return_option}. Use 'last_message' or 'state'."
                )

        async def _stream_until_interrupt(stream_input, cfg):
            """Stream the workflow until completion or an interrupt.

            Returns ``(last_state, interrupt_value)`` where
            ``interrupt_value`` is ``None`` when the graph completed
            normally.

            LangGraph's ``astream(stream_mode="values")`` does **not**
            raise ``GraphInterrupt``.  Instead the stream emits a state
            containing an ``__interrupt__`` key and then ends.  We
            detect this in two ways:

            1. Check for the ``__interrupt__`` key in streamed states.
            2. After the stream ends, inspect the checkpoint snapshot
               for pending interrupt tasks.
            """
            prev_msgs: list = []
            last_st = None
            interrupt_val = None
            try:
                async for s in self.workflow.astream(
                    stream_input, stream_mode="values", config=cfg
                ):
                    # Detect inline interrupt marker emitted by astream.
                    if "__interrupt__" in s:
                        int_data = s["__interrupt__"]
                        if isinstance(int_data, (list, tuple)) and int_data:
                            interrupt_val = int_data[0].value
                        elif hasattr(int_data, "value"):
                            interrupt_val = int_data.value
                        else:
                            interrupt_val = {
                                "question": "The workflow needs your input."
                            }

                    if "messages" in s and s["messages"] != prev_msgs:
                        new_message = s["messages"][-1]
                        try:
                            new_message.pretty_print()
                        except Exception:
                            pass
                        logger.info(new_message)
                        prev_msgs = s["messages"]
                    last_st = s
            except GraphInterrupt as gi:
                # Fallback: some LangGraph versions may still raise.
                interrupts = gi.args[0] if gi.args else []
                if interrupts:
                    interrupt_val = interrupts[0].value
                else:
                    interrupt_val = {
                        "question": "The workflow needs your input."
                    }

            # Double-check the checkpoint for pending interrupts that
            # the stream may not have surfaced explicitly.
            if interrupt_val is None:
                try:
                    snapshot = self.workflow.get_state(cfg)
                    if snapshot and snapshot.tasks:
                        for t in snapshot.tasks:
                            t_interrupts = getattr(t, "interrupts", None)
                            if t_interrupts:
                                interrupt_val = t_interrupts[0].value
                                break
                except Exception:
                    pass

            if interrupt_val is not None:
                logger.info("Graph interrupted: %s", interrupt_val)
                # Refresh state from checkpoint for consistency.
                try:
                    snapshot = self.workflow.get_state(cfg)
                    if snapshot:
                        last_st = snapshot.values
                except Exception:
                    pass

            return last_st, interrupt_val

        logger.debug("run called with config=%s", config)
        config = _validate_config(config)
        logger.debug("validated config=%s", config)

        # Initialize logging directory before determining inputs or running workflow
        # Check if CHEMGRAPH_LOG_DIR is already set
        if not os.environ.get("CHEMGRAPH_LOG_DIR"):
            os.environ["CHEMGRAPH_LOG_DIR"] = self.log_dir

        # Ensure session exists in memory store
        self._ensure_session(query)

        # If resuming from a previous session, prepend context
        if resume_from and self.session_store:
            context = self.session_store.build_context_summary(resume_from)
            if context:
                query = (
                    f"{context}\n\n"
                    f"Now, continuing from the previous session above, "
                    f"please help with the following:\n\n{query}"
                )
                logger.info(f"Injected context from session {resume_from}")

        inputs = {"messages": query}

        try:
            last_state, interrupt_value = await _stream_until_interrupt(inputs, config)

            # --- Human-in-the-loop resume loop ---
            # When the graph pauses with an interrupt, ask the human and
            # resume.  This loop handles chains of multiple interrupts
            # (e.g., the agent asks a follow-up question after receiving
            # the first answer).
            max_interrupts = 10  # safety guard against infinite interrupt loops
            interrupt_count = 0
            while interrupt_value is not None:
                interrupt_count += 1
                if interrupt_count > max_interrupts:
                    logger.error(
                        "Exceeded maximum number of human interrupts (%d); "
                        "aborting workflow.",
                        max_interrupts,
                    )
                    raise RuntimeError(
                        f"Workflow exceeded maximum of {max_interrupts} "
                        f"human interrupts."
                    )

                # Extract the question text from the interrupt value.
                if isinstance(interrupt_value, dict):
                    question = interrupt_value.get(
                        "question",
                        interrupt_value.get("message", str(interrupt_value)),
                    )
                else:
                    question = str(interrupt_value)

                logger.info("Requesting human input: %s", question)
                human_answer = await self._call_human_input_handler(question)
                logger.info("Human responded: %s", human_answer)

                # Resume the graph from the checkpoint with the human's answer.
                resume_cmd = Command(resume=human_answer)
                last_state, interrupt_value = await _stream_until_interrupt(
                    resume_cmd, config
                )

            if last_state is None:
                raise RuntimeError("Workflow produced no states.")

            # Save messages to persistent session store
            self._save_messages_to_store(last_state, query)

            return _save_state_and_select_return(last_state, config)

        except HumanInputRequired:
            # No human_input_handler configured — propagate so the
            # caller (CLI / UI) can prompt the user and resume.
            raise
        except Exception as e:
            logger.error(f"Error running workflow {self.workflow_type}: {e}")
            raise

class HumanInputRequired(Exception):
    """Raised when the graph needs human input but no handler is configured.

    Carries the question text so that external callers (CLI, UI) can
    present it to the user and resume the graph with
    ``Command(resume=answer)``.
    """

    def __init__(self, question: str):
        self.question = question
        super().__init__(question)
