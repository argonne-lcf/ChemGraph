import asyncio
import datetime
import json
import os
import time
from pathlib import Path
from typing import Callable, Collection, List, Optional
import uuid

from chemgraph.agent.events import EventCallback, _AstreamEventCallback
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

from chemgraph.schemas.ase_input import (
    get_available_calculator_names,
    get_calculator_selection_context,
    get_default_calculator_name,
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
    get_planner_prompt,
)
from langgraph.types import Command
from langgraph.errors import GraphInterrupt

from chemgraph.graphs.single_agent import construct_single_agent_graph
from chemgraph.agent.turn import serialize_state


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


# Every cost-bearing tool takes a single pydantic parameter, so the LLM's
# tool-call ``args`` nest one level under a wrapper key (``params`` for most,
# ``graspa_input`` for gRASPA). The manifest reads ``driver`` / ``calculator`` /
# ``input_structure_file`` at the TOP level of the recorded args, so the hook
# must unwrap that inner dict before recording -- otherwise every field reads as
# ``?``. Some tools name their structure field differently; ``alias`` maps that
# field onto the canonical ``input_structure_file`` a resume reads.
_TOOL_ARG_SPEC = {
    "run_ase": {"wrapper": "params"},
    "run_xanes": {"wrapper": "params"},
    "run_mace_single": {"wrapper": "params"},
    "run_mace_ensemble": {"wrapper": "params", "alias": "input_structure_directory"},
    "run_graspa": {"wrapper": "graspa_input", "alias": "cif_path"},
}


def _unwrap_tool_args(name, args):
    """Return a cost tool's inner args dict with the structure field normalized.

    Cost tools wrap their real arguments under a single pydantic-param key; the
    manifest reads ``driver`` / ``calculator`` / ``input_structure_file`` at the
    top level, so return the inner dict. When the wrapper key is absent (a flat
    direct call, or a test feeding args already unwrapped), return ``args``
    unchanged. Any tool-specific structure-field alias is copied onto
    ``input_structure_file`` so a resume finds it under the canonical name.
    """
    if not isinstance(args, dict):
        return args
    spec = _TOOL_ARG_SPEC.get(name)
    if spec is None:
        return args
    wrapper = spec.get("wrapper")
    inner = args.get(wrapper) if wrapper else None
    if not isinstance(inner, dict):
        # Already-flat args (direct call / test shape) -> use as-is.
        inner = args
    alias = spec.get("alias")
    if alias and alias in inner and "input_structure_file" not in inner:
        inner = dict(inner)
        inner["input_structure_file"] = inner[alias]
    return inner


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
        is removed from the default system prompt, by default False.
    terminal_tool_names : Collection[str], optional
        Tool names that should terminate supported workflows after
        successful execution, by default empty.
    on_event : callable, optional
        Callback invoked with dashboard workflow events, by default None.

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
        human_supervised: bool = False,
        terminal_tool_names: Collection[str] = (),
        on_event: Optional[EventCallback] = None,
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

        # Durable run manifest (records executed cost-bearing steps + args +
        # result-file paths for crash-safe, cross-allocation resume). Lives in
        # the log dir on the shared filesystem, independent of the SessionStore
        # message layer which drops tool-call args.
        from chemgraph.memory.manifest import RunManifest

        self.run_manifest = RunManifest(
            os.path.join(self.log_dir, "run_manifest.json")
        )
        # Open cost-bearing steps, keyed by tool_call_id so parallel tool calls
        # in one AI message each close the correct manifest step. Each value is
        # {"idx": int, "tool": str, "args": dict}.
        self._pending_steps: dict[str, dict] = {}

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
        self.terminal_tool_names = tuple(terminal_tool_names)
        self.on_event = on_event

        # Record whether the caller relied on the default system prompt before
        # any mutation below rewrites it (e.g. stripping ask_human when
        # unsupervised). Downstream workflow branches use this to decide whether
        # to substitute their own default prompt.
        prompt_is_default = system_prompt == single_agent_prompt

        # When human supervision is disabled and the caller is using the
        # default system prompt, strip the ask_human instructions so the
        # LLM is not told to call a tool that is unavailable.
        if not self.human_supervised and self.system_prompt == single_agent_prompt:
            self.system_prompt = get_single_agent_prompt(human_supervised=False)
        if not self.human_supervised and self.planner_prompt == default_planner_prompt:
            self.planner_prompt = get_planner_prompt(human_supervised=False)

        self.available_calculators = get_available_calculator_names()
        self.default_calculator = get_default_calculator_name()
        self.calculator_selection_context = get_calculator_selection_context()

        def append_calculator_context(prompt: str) -> str:
            """Append calculator availability guidance to a prompt once.

            Parameters
            ----------
            prompt : str
                Prompt text to augment.

            Returns
            -------
            str
                Prompt with calculator-selection context appended.
            """
            if self.calculator_selection_context in prompt:
                return prompt
            return f"{prompt}{self.calculator_selection_context}"

        if self.workflow_type in {"single_agent", "mock_agent", "single_agent_mcp"}:
            self.system_prompt = append_calculator_context(self.system_prompt)
        elif self.workflow_type == "multi_agent":
            self.planner_prompt = append_calculator_context(self.planner_prompt)
            self.executor_prompt = append_calculator_context(self.executor_prompt)

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
                terminal_tool_names=self.terminal_tool_names,
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
                human_supervised=self.human_supervised,
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
                if not prompt_is_default
                else rag_agent_prompt,
                tools=self.tools,
            )
        elif self.workflow_type == "single_agent_xanes":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm,
                system_prompt=self.system_prompt
                if not prompt_is_default
                else default_xanes_single_agent_prompt,
                structured_output=self.structured_output,
                formatter_prompt=self.formatter_prompt
                if self.formatter_prompt != default_formatter_prompt
                else default_xanes_formatter_prompt,
                tools=self.tools,
            )

    def visualize(self):
        """Return an ASCII representation of the LangGraph workflow."""
        return self.workflow.get_graph().draw_ascii()

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

                # MCP tool messages may return content as a list of
                # content blocks (e.g. [{'type': 'text', 'text': '...'}])
                # instead of a plain string. Normalize to str.
                if isinstance(content, list):
                    content = "\n".join(
                        block.get("text", str(block))
                        if isinstance(block, dict)
                        else str(block)
                        for block in content
                    )
                elif not isinstance(content, str):
                    content = str(content)

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

    _COST_TOOLS = {
        "run_ase",
        "run_mace_single",
        "run_mace_ensemble",
        "run_graspa",
        "run_xanes",
    }

    def _manifest_observe(self, message) -> None:
        """Record cost-bearing tool starts/ends into the run manifest.

        Reads tool-call args from live graph state -- before
        ``_save_messages_to_store``'s empty-content filter drops pure tool-call
        messages -- so the durable manifest keeps the args (calculator, driver,
        input file) a resumed agent needs. Best-effort: never breaks the run.
        """
        try:
            from chemgraph.agent.turn import _message_tool_calls

            # AI message issuing a cost-bearing tool call -> record a step start.
            for call in _message_tool_calls(message) or []:
                name = (
                    call.get("name")
                    if isinstance(call, dict)
                    else getattr(call, "name", None)
                )
                if name in self._COST_TOOLS:
                    raw_args = (
                        call.get("args", {})
                        if isinstance(call, dict)
                        else getattr(call, "args", {})
                    )
                    # The LLM nests the real args under a single pydantic-param
                    # wrapper key; unwrap so the manifest records driver /
                    # calculator / input_structure_file at the top level.
                    args = _unwrap_tool_args(name, raw_args)
                    call_id = (
                        call.get("id")
                        if isinstance(call, dict)
                        else getattr(call, "id", None)
                    )
                    idx = self.run_manifest.record_step_start(name, args)
                    # Key by tool_call_id so the matching ToolMessage closes THIS
                    # step even when several tool calls are open at once. A missing
                    # id (rare) falls back to a synthetic key.
                    key = call_id if call_id is not None else f"__noid_{idx}"
                    self._pending_steps[key] = {
                        "idx": idx,
                        "tool": name,
                        "args": args,
                    }

            # ToolMessage returning a result -> record the step end.
            role = getattr(message, "type", None) or getattr(message, "role", None)
            if role == "tool" and self._pending_steps:
                # Match the ToolMessage to its open step by tool_call_id. If the
                # id is absent or unknown, fall back to the single open step when
                # exactly one is pending (unambiguous), else skip.
                call_id = getattr(message, "tool_call_id", None)
                if call_id in self._pending_steps:
                    key = call_id
                elif len(self._pending_steps) == 1:
                    key = next(iter(self._pending_steps))
                else:
                    key = None
                if key is None:
                    return
                pending = self._pending_steps.pop(key)
                pending_step_idx = pending["idx"]
                pending_step_tool = pending["tool"]
                pending_step_args = pending["args"]
                content = str(getattr(message, "content", ""))
                # LangGraph's ToolNode serializes a non-string tool return via
                # json.dumps, so in production `content` is the JSON of the
                # run_ase_core result dict. Parse it and read the resume contract
                # (wall_time_capped / result_file / restart_file /
                # resume_input_file / wall_time) as structured fields. Scraping the
                # human-readable message for a "saved to <path>" substring is
                # brittle (it breaks the moment the wording changes and mis-parses
                # a trailing '."' as part of the path), so JSON is the source of
                # truth; the string path below is only a fallback for a plain-text
                # ToolMessage (older tools / hand-written test content).
                is_error = getattr(message, "status", None) == "error"
                result_file = None
                wall_time = None
                wall_time_capped = False
                restart_file = None
                resume_input_file = None
                try:
                    payload = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    payload = None
                if isinstance(payload, dict):
                    is_error = is_error or payload.get("status") == "failure"
                    if not is_error:
                        result_file = payload.get("result_file")
                        wall_time = payload.get("wall_time")
                        wall_time_capped = bool(payload.get("wall_time_capped"))
                        restart_file = payload.get("restart_file")
                        resume_input_file = payload.get("resume_input_file")
                else:
                    # Plain-text fallback: recover the result-file path from the
                    # message and read the capped flags out of that JSON on disk.
                    is_error = (
                        is_error
                        or content.lstrip().startswith("Error")
                        or '"status": "failure"' in content
                    )
                    if not is_error and "saved to " in content:
                        result_file = (
                            content.split("saved to ", 1)[1].split()[0].rstrip(".")
                        )
                    if result_file and os.path.isfile(result_file):
                        try:
                            parsed = json.loads(
                                Path(result_file).read_text(encoding="utf-8")
                            )
                            wall_time = parsed.get("wall_time")
                            wall_time_capped = bool(parsed.get("wall_time_capped"))
                            restart_file = parsed.get("restart_file")
                        except Exception:
                            pass
                if not is_error and wall_time_capped:
                    # Wall-clock cap: record the step as "capped" (NOT done) and
                    # queue the same step as the pending next step so a resumed
                    # agent continues it as unfinished work.
                    self.run_manifest.record_step_end(
                        pending_step_idx,
                        result_file=result_file,
                        wall_time=wall_time,
                        status="capped",
                    )
                    # For a capped opt, the durable partial geometry is a
                    # standalone structure file; point the pending step's input at
                    # it so a resume continues from the moved atoms and skips
                    # recomputing from the original input.
                    pending_args = pending_step_args
                    if resume_input_file:
                        pending_args = dict(pending_args)
                        pending_args["input_structure_file"] = resume_input_file
                        reason = (
                            "wall-clock cap; resume from partial geometry "
                            f"{resume_input_file}"
                        )
                    elif restart_file:
                        reason = (
                            f"wall-clock cap; resume with restart_file={restart_file}"
                        )
                    else:
                        reason = (
                            "wall-clock cap; no restart written, rerun with "
                            "more wall-clock budget"
                        )
                    self.run_manifest.set_pending(
                        pending_step_tool,
                        pending_args,
                        reason=reason,
                    )
                    self.run_manifest.set_status("capped")
                else:
                    self.run_manifest.record_step_end(
                        pending_step_idx,
                        result_file=result_file,
                        wall_time=wall_time,
                        status="failed" if is_error else "done",
                    )
                    if not is_error:
                        # A genuinely-completed step clears any stale PENDING
                        # marker and resets a leftover 'capped' status, so a
                        # successful resume no longer renders the old pending
                        # block. (A failed step leaves both untouched.)
                        self.run_manifest.mark_running()
        except Exception as exc:  # manifest is best-effort, never break the run
            logger.debug("manifest observe skipped: %s", exc)

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
        from chemgraph.agent.turn import (
            _executed_tool_names,
            _state_messages,
            _terminal_tool_name,
        )

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
                        self._manifest_observe(new_message)
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
        thread_id = str(config["configurable"]["thread_id"])
        started = time.time()
        event = self.on_event or (lambda _event, _payload: None)
        if self.on_event:
            callbacks = list(config.get("callbacks") or [])
            callbacks.append(_AstreamEventCallback(self.on_event, thread_id))
            config["callbacks"] = callbacks
        logger.debug("validated config=%s", config)

        # When resuming, adopt the prior session's log dir BEFORE anything reads
        # it. The durable partials a resume must find ({stem}_opt.partial.xyz,
        # {stem}_vibcache, run_manifest.json) live under that session's log_dir;
        # a fresh auto-generated dir would hide them and force a full recompute.
        # Also re-point self.run_manifest at the prior manifest so new steps
        # append to it, keeping all bookkeeping in one
        # file. This is the single choke point for every resume path.
        if resume_from and self.session_store:
            try:
                prior = self.session_store.get_session(resume_from)
            except Exception:
                prior = None
            prior_log_dir = getattr(prior, "log_dir", None) if prior else None
            if prior_log_dir and os.path.isdir(prior_log_dir):
                self.log_dir = prior_log_dir
                os.environ["CHEMGRAPH_LOG_DIR"] = prior_log_dir
                from chemgraph.memory.manifest import RunManifest

                self.run_manifest = RunManifest(
                    os.path.join(prior_log_dir, "run_manifest.json")
                )

        # Initialize logging directory before determining inputs or running workflow
        # Check if CHEMGRAPH_LOG_DIR is already set
        if not os.environ.get("CHEMGRAPH_LOG_DIR"):
            os.environ["CHEMGRAPH_LOG_DIR"] = self.log_dir

        # Ensure session exists in memory store
        self._ensure_session(query)

        # If resuming from a previous session, prepend context. Augment the
        # text summary with the run manifest (completed steps + result-file
        # paths + pending next step) so the agent continues precisely and avoids
        # recomputing work whose args the summary layer dropped.
        if resume_from and self.session_store:
            context = self.session_store.build_context_summary(resume_from)
            from chemgraph.memory.manifest import RunManifest

            manifest = RunManifest.for_session(self.session_store, resume_from)
            if manifest:
                context = f"{context}\n\n{manifest.render_for_context()}"
            if context:
                query = (
                    f"{context}\n\n"
                    f"Now, continuing from the previous session above, "
                    f"please help with the following:\n\n{query}"
                )
                logger.info(f"Injected context from session {resume_from}")

        inputs = {"messages": query}
        event(
            "workflow_started",
            {
                "workflow_type": self.workflow_type,
                "thread_id": thread_id,
                "tool_names": [
                    getattr(tool, "name", str(tool)) for tool in self.tools or []
                ],
            },
        )

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

            messages = _state_messages(last_state)
            executed_tools = _executed_tool_names(messages)
            terminal_tool = _terminal_tool_name(
                executed_tools,
                self.terminal_tool_names,
            )
            event(
                "workflow_finished",
                {
                    "workflow_type": self.workflow_type,
                    "thread_id": thread_id,
                    "status": "completed",
                    "executed_tool_names": list(executed_tools),
                    "terminal_tool": terminal_tool,
                    "duration_s": round(time.time() - started, 3),
                },
            )

            return _save_state_and_select_return(last_state, config)

        except HumanInputRequired:
            # No human_input_handler configured — propagate so the
            # caller (CLI / UI) can prompt the user and resume.
            raise
        except Exception as e:
            event(
                "workflow_finished",
                {
                    "workflow_type": self.workflow_type,
                    "thread_id": thread_id,
                    "status": "failed",
                    "error": repr(e),
                    "duration_s": round(time.time() - started, 3),
                },
            )
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
