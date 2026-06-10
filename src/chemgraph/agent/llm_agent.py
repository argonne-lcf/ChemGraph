import asyncio
import datetime
import dataclasses
import os
import time
from typing import Any, Callable, Collection, List, Optional
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
)
from langgraph.errors import GraphInterrupt
from langchain_core.messages import AIMessage
from langchain_core.callbacks import BaseCallbackHandler

from chemgraph.graphs.single_agent import construct_single_agent_graph
from chemgraph.graphs.multi_agent import construct_multi_agent_graph
from chemgraph.graphs.graspa_mcp import construct_graspa_mcp_graph
from chemgraph.prompt.rag_prompt import rag_agent_prompt
from chemgraph.prompt.xanes_prompt import (
    xanes_single_agent_prompt as default_xanes_single_agent_prompt,
    xanes_formatter_prompt as default_xanes_formatter_prompt,
)
from chemgraph.tools.ase_tools import (
    file_to_atomsdata,
    run_ase,
    save_atomsdata_to_file,
)
from chemgraph.tools.cheminformatics_tools import (
    molecule_name_to_smiles,
    smiles_to_atomsdata,
    smiles_to_coordinate_file,
)
from chemgraph.tools.generic_tools import calculator, repl_tool
from chemgraph.tools.graspa_tools import run_graspa
from chemgraph.tools.rag_tools import load_document, query_knowledge_base
from chemgraph.tools.xanes_tools import (
    fetch_xanes_data,
    plot_xanes_data,
    run_xanes,
)

import logging

logger = logging.getLogger(__name__)


SINGLE_AGENT_TURN_WORKFLOWS = {
    "single_agent",
    "python_relp",
    "graspa",
    "mock_agent",
    "single_agent_mcp",
    "rag_agent",
    "single_agent_xanes",
}

LEGACY_GRAPH_WORKFLOWS = {"multi_agent", "graspa_mcp"}


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", getattr(tool, "__name__", repr(tool))))


def _merge_tools(*groups: Collection[Any] | None) -> list[Any]:
    """Merge tool groups by visible tool name while preserving order."""
    merged: list[Any] = []
    seen: set[str] = set()
    for group in groups:
        for tool in group or ():
            name = _tool_name(tool)
            if name not in seen:
                merged.append(tool)
                seen.add(name)
    return merged


def _xanes_tools() -> list[Any]:
    return [
        molecule_name_to_smiles,
        smiles_to_coordinate_file,
        run_ase,
        run_xanes,
        fetch_xanes_data,
        plot_xanes_data,
    ]


def _rag_tools() -> list[Any]:
    return [
        load_document,
        query_knowledge_base,
        file_to_atomsdata,
        smiles_to_coordinate_file,
        run_ase,
        molecule_name_to_smiles,
        save_atomsdata_to_file,
        calculator,
    ]


def _mock_tools() -> list[Any]:
    return [
        file_to_atomsdata,
        smiles_to_atomsdata,
        run_ase,
        molecule_name_to_smiles,
        save_atomsdata_to_file,
        calculator,
    ]


def _last_ai_message(state: dict[str, Any], fallback_text: str) -> AIMessage:
    """Return the last AI message from a turn state, preserving objects when present."""
    messages = state.get("messages", []) if isinstance(state, dict) else []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
        if isinstance(message, dict):
            message_type = message.get("type") or message.get("role")
            if message_type in {"ai", "assistant"}:
                return AIMessage(content=_message_text(message))
    return AIMessage(content=fallback_text)


def _is_mock_object(value) -> bool:
    """Return True for unittest.mock objects without importing test-only APIs.

    Parameters
    ----------
    value : Any
        Object to inspect.

    Returns
    -------
    bool
        ``True`` when the object comes from ``unittest.mock``.
    """
    return value.__class__.__module__.startswith("unittest.mock")


def serialize_state(state, *, max_depth: int = 50, _seen: set[int] | None = None):
    """Convert non-serializable objects in state to a JSON-friendly format.

    Parameters
    ----------
    state : Any
        The state object to be serialized. Can be a list, dict, or object with __dict__
    max_depth : int, optional
        Maximum object nesting depth to serialize before falling back to a
        placeholder. This prevents runaway recursion for complex graph objects.

    Returns
    -------
    Any
        A JSON-serializable version of the input state
    """
    if _seen is None:
        _seen = set()

    if max_depth < 0:
        return f"<max depth exceeded: {type(state).__name__}>"

    if isinstance(state, (str, int, float, bool)) or state is None:
        return state

    if isinstance(state, (datetime.datetime, datetime.date)):
        return state.isoformat()

    if _is_mock_object(state):
        return str(state)

    state_id = id(state)
    if state_id in _seen:
        return f"<circular reference: {type(state).__name__}>"

    if isinstance(state, dict):
        _seen.add(state_id)
        try:
            return {
                str(key): serialize_state(
                    value, max_depth=max_depth - 1, _seen=_seen
                )
                for key, value in state.items()
            }
        finally:
            _seen.remove(state_id)

    if isinstance(state, (list, tuple, set, frozenset)):
        _seen.add(state_id)
        try:
            return [
                serialize_state(item, max_depth=max_depth - 1, _seen=_seen)
                for item in state
            ]
        finally:
            _seen.remove(state_id)

    model_dump = getattr(state, "model_dump", None)
    if callable(model_dump):
        _seen.add(state_id)
        try:
            try:
                dumped = model_dump(mode="json")
            except TypeError:
                dumped = model_dump()
            return serialize_state(dumped, max_depth=max_depth - 1, _seen=_seen)
        except Exception:
            return str(state)
        finally:
            _seen.remove(state_id)

    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        _seen.add(state_id)
        try:
            return {
                field.name: serialize_state(
                    getattr(state, field.name),
                    max_depth=max_depth - 1,
                    _seen=_seen,
                )
                for field in dataclasses.fields(state)
            }
        finally:
            _seen.remove(state_id)

    if hasattr(state, "__dict__"):
        _seen.add(state_id)
        try:
            return {
                str(key): serialize_state(
                    value, max_depth=max_depth - 1, _seen=_seen
                )
                for key, value in vars(state).items()
            }
        finally:
            _seen.remove(state_id)

    return str(state)


def _custom_openai_compatible_kwargs(
    *,
    model_name: str,
    temperature: float,
    base_url: str,
    api_key: str,
    max_tokens: int,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    argo_user: str | None,
) -> dict:
    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "base_url": base_url,
        "api_key": api_key,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    user = argo_user or os.getenv("ARGO_USER")
    if base_url and "argoapi" in base_url and user:
        kwargs["model_kwargs"] = {"user": user}
    return kwargs


EventCallback = Callable[[str, dict], None]


@dataclasses.dataclass(frozen=True)
class TurnResult:
    """Result of one bounded ChemGraph single-agent turn."""

    final_text: str
    state: dict[str, Any]
    executed_tool_names: tuple[str, ...]
    terminal_tool: str | None
    thread_id: str
    duration_s: float


class _TurnEventCallback(BaseCallbackHandler):
    """Forward LangChain callback events to a small stable callback surface."""

    def __init__(self, on_event: EventCallback, thread_id: str) -> None:
        self._on_event = on_event
        self._thread_id = thread_id

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        try:
            self._on_event(event, {"thread_id": self._thread_id, **payload})
        except Exception:  # noqa: BLE001 - callbacks must not break the run.
            logger.debug("turn event callback failed", exc_info=True)

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self._emit(
            "llm_call_started",
            {
                "model": _serialized_name(serialized),
                "message_count": len(messages[0]) if messages else 0,
            },
        )

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self._emit(
            "llm_call_started",
            {
                "model": _serialized_name(serialized),
                "message_count": len(prompts or []),
            },
        )

    def on_llm_end(self, response, **kwargs) -> None:
        payload: dict[str, Any] = {}
        usage = getattr(response, "llm_output", None)
        if isinstance(usage, dict):
            payload["llm_output"] = usage
        self._emit("llm_call_finished", payload)

    def on_llm_error(self, error, **kwargs) -> None:
        self._emit("llm_call_failed", {"error": repr(error)})

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self._emit(
            "tool_call_started",
            {
                "tool_name": _serialized_name(serialized),
                "arguments": serialize_state(input_str),
            },
        )

    def on_tool_end(self, output, **kwargs) -> None:
        payload: dict[str, Any] = {"result": serialize_state(output)}
        name = kwargs.get("name")
        if name:
            payload["tool_name"] = name
        self._emit("tool_call_finished", payload)

    def on_tool_error(self, error, **kwargs) -> None:
        payload = {"error": repr(error)}
        name = kwargs.get("name")
        if name:
            payload["tool_name"] = name
        self._emit("tool_call_failed", payload)


def _serialized_name(serialized: Any) -> str | None:
    if isinstance(serialized, dict):
        return serialized.get("name") or serialized.get("id")
    return None


def _message_tool_calls(message: Any) -> list[Any]:
    if isinstance(message, dict):
        calls = message.get("tool_calls")
    else:
        calls = getattr(message, "tool_calls", None)
    return calls if isinstance(calls, list) else []


def _tool_message_name(message: Any) -> str | None:
    if isinstance(message, dict):
        name = message.get("name")
        role = message.get("role") or message.get("type")
        if name and role in {"tool", "tool_message", "ToolMessage"}:
            return str(name)
        return str(name) if name and not _message_tool_calls(message) else None
    name = getattr(message, "name", None)
    message_type = getattr(message, "type", None)
    if name and message_type == "tool":
        return str(name)
    return str(name) if name and not _message_tool_calls(message) else None


def _call_name(call: Any) -> str | None:
    if isinstance(call, dict):
        if call.get("name"):
            return str(call["name"])
        function = call.get("function")
        if isinstance(function, dict) and function.get("name"):
            return str(function["name"])
    name = getattr(call, "name", None)
    return str(name) if name else None


def _state_messages(state: Any) -> list[Any]:
    if isinstance(state, dict):
        messages = state.get("messages", [])
    else:
        messages = getattr(state, "messages", [])
    return list(messages or [])


def _executed_tool_names(messages: list[Any]) -> tuple[str, ...]:
    names: list[str] = []
    for message in messages:
        name = _tool_message_name(message)
        if name:
            names.append(name)
    if names:
        return tuple(names)
    for message in messages:
        for call in _message_tool_calls(message):
            if name := _call_name(call):
                names.append(name)
    return tuple(names)


def _terminal_tool_name(
    executed_tool_names: tuple[str, ...],
    terminal_tool_names: Collection[str],
) -> str | None:
    terminal = set(terminal_tool_names)
    for name in reversed(executed_tool_names):
        if name in terminal:
            return name
    return None


def _message_text(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _final_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        message_type = (
            message.get("role") or message.get("type")
            if isinstance(message, dict)
            else getattr(message, "type", None)
        )
        if message_type in {"ai", "assistant"}:
            return _message_text(message)
    return _message_text(messages[-1]) if messages else ""


def _load_turn_llm(
    *,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    argo_user: str | None,
) -> Any:
    temperature = 0.0
    if model_name in supported_openai_models or model_name in supported_argo_models:
        kwargs = {
            "model_name": model_name,
            "temperature": temperature,
            "base_url": base_url,
        }
        if argo_user is not None:
            kwargs["argo_user"] = argo_user
        return load_openai_model(**kwargs)
    if model_name in supported_ollama_models:
        return load_ollama_model(model_name=model_name, temperature=temperature)
    if model_name in supported_alcf_models:
        return load_alcf_model(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
        )
    if model_name in supported_anthropic_models:
        return load_anthropic_model(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
        )
    if model_name in supported_gemini_models:
        return load_gemini_model(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
        )
    if model_name.startswith("groq:"):
        return load_groq_model(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
        )

    endpoint = os.getenv("VLLM_BASE_URL", base_url or "")
    key = os.getenv("OPENAI_API_KEY", api_key or "dummy_vllm_key")
    if not endpoint:
        raise ValueError(f"Unsupported model or missing base URL for: {model_name}")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        **_custom_openai_compatible_kwargs(
            model_name=model_name,
            temperature=temperature,
            base_url=endpoint,
            api_key=key,
            max_tokens=4000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            argo_user=argo_user,
        ),
    )


async def run_turn(
    *,
    query: str,
    tools: list[Any] | None = None,
    model_name: str = "gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    argo_user: str | None = None,
    system_prompt: str = single_agent_prompt,
    formatter_prompt: str = default_formatter_prompt,
    structured_output: bool = False,
    generate_report: bool = False,
    report_prompt: str = default_report_prompt,
    recursion_limit: int = 50,
    thread_id: str | None = None,
    terminal_tool_names: Collection[str] = (),
    human_supervised: bool = False,
    on_event: EventCallback | None = None,
) -> TurnResult:
    """Run one bounded single-agent ChemGraph LangGraph turn."""

    started = time.time()
    thread_id = thread_id or str(uuid.uuid4())
    callbacks = [_TurnEventCallback(on_event, thread_id)] if on_event else []
    event = on_event or (lambda _event, _payload: None)
    event(
        "workflow_started",
        {
            "workflow_type": "single_agent",
            "thread_id": thread_id,
            "tool_names": [getattr(tool, "name", str(tool)) for tool in tools or []],
        },
    )
    llm = _load_turn_llm(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        argo_user=argo_user,
    )
    workflow = construct_single_agent_graph(
        llm,
        system_prompt,
        structured_output,
        formatter_prompt,
        generate_report,
        report_prompt,
        tools,
        human_supervised=human_supervised,
        terminal_tool_names=terminal_tool_names,
    )
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
    if callbacks:
        config["callbacks"] = callbacks

    last_state: Any = None
    try:
        async for state in workflow.astream(
            {"messages": query},
            stream_mode="values",
            config=config,
        ):
            last_state = state
    except Exception as exc:
        event(
            "workflow_finished",
            {
                "workflow_type": "single_agent",
                "thread_id": thread_id,
                "status": "failed",
                "error": repr(exc),
                "duration_s": round(time.time() - started, 3),
            },
        )
        raise

    if last_state is None:
        raise RuntimeError("ChemGraph turn produced no states.")

    messages = _state_messages(last_state)
    executed_tools = _executed_tool_names(messages)
    terminal_tool = _terminal_tool_name(executed_tools, terminal_tool_names)
    result = TurnResult(
        final_text=_final_text(messages),
        state=serialize_state(last_state),
        executed_tool_names=executed_tools,
        terminal_tool=terminal_tool,
        thread_id=thread_id,
        duration_s=round(time.time() - started, 3),
    )
    event(
        "workflow_finished",
        {
            "workflow_type": "single_agent",
            "thread_id": thread_id,
            "status": "completed",
            "executed_tool_names": list(result.executed_tool_names),
            "terminal_tool": terminal_tool,
            "duration_s": result.duration_s,
        },
    )
    return result


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
    ):
        """Initialize a ChemGraph workflow instance.

        Parameters
        ----------
        model_name : str, optional
            LLM model identifier.
        workflow_type : str, optional
            Workflow constructor key.
        base_url : str, optional
            Custom provider endpoint URL.
        api_key : str, optional
            API key passed to compatible model loaders.
        argo_user : str, optional
            Argo username for Argo-hosted models.
        system_prompt : str, optional
            System prompt for single-agent-style workflows.
        formatter_prompt : str, optional
            Prompt used to format single-agent final output.
        structured_output : bool, optional
            Whether structured final output is requested.
        return_option : str, optional
            Return mode, such as ``"last_message"`` or ``"state"``.
        recursion_limit : int, optional
            LangGraph recursion limit.
        planner_prompt : str, optional
            Planner prompt for multi-agent workflows.
        executor_prompt : str, optional
            Executor prompt for multi-agent workflows.
        aggregator_prompt : str, optional
            Aggregator prompt retained for compatibility.
        formatter_multi_prompt : str, optional
            Formatter prompt for multi-agent workflows.
        generate_report : bool, optional
            Whether report generation is enabled.
        report_prompt : str, optional
            Prompt used by the report-generation workflow.
        support_structured_output : bool, optional
            Whether the selected model supports structured output.
        tools : list, optional
            Custom tool list for applicable workflows.
        data_tools : list, optional
            Additional data-analysis tools for MCP workflows.
        session_store : SessionStore, optional
            Existing session store instance.
        enable_memory : bool, optional
            Whether persistent session memory is enabled.
        memory_db_path : str, optional
            SQLite path for the session store.
        log_dir : str, optional
            Directory for run logs and artifacts.
        max_retries : int, optional
            LLM parse-retry limit for formatter/planner nodes.
        human_input_handler : Callable[[str], str], optional
            Callback used to answer graph human-interrupt prompts.
        human_supervised : bool, optional
            Whether to expose human-supervision tools to the agent.
        """
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

                    llm_kwargs = _custom_openai_compatible_kwargs(
                        model_name=model_name,
                        temperature=temperature,
                        base_url=vllm_base_url,
                        api_key=vllm_api_key,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        argo_user=argo_user,
                    )
                    llm = ChatOpenAI(**llm_kwargs)
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

        supported_workflows = SINGLE_AGENT_TURN_WORKFLOWS | LEGACY_GRAPH_WORKFLOWS
        if workflow_type not in supported_workflows:
            raise ValueError(
                f"Unsupported workflow type: {workflow_type}. "
                f"Available types: {sorted(supported_workflows)}"
            )

        self._using_default_system_prompt = system_prompt == single_agent_prompt
        self._using_default_formatter_prompt = formatter_prompt == default_formatter_prompt

        self.workflow_type = workflow_type
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.argo_user = argo_user
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
        self._last_run_state: dict[str, Any] | None = None

        # When human supervision is disabled and the caller is using the
        # default system prompt, strip the ask_human instructions so the
        # LLM is not told to call a tool that is unavailable.
        if not self.human_supervised and self.system_prompt == single_agent_prompt:
            self.system_prompt = get_single_agent_prompt(human_supervised=False)

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
            "multi_agent": {"constructor": construct_multi_agent_graph},
            "graspa_mcp": {"constructor": construct_graspa_mcp_graph},
        }

        self.tools = self._resolve_turn_tools(tools, data_tools)
        self._resolve_turn_prompts()

        if self.workflow_type == "multi_agent":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm,
                planner_prompt=self.planner_prompt,
                executor_prompt=self.executor_prompt,
                executor_tools=self.tools,
                structured_output=self.structured_output,
                formatter_prompt=self.formatter_multi_prompt,
                max_retries=self.max_retries,
            )
        elif self.workflow_type == "graspa_mcp":
            self.workflow = self.workflow_map[workflow_type]["constructor"](
                llm=llm,
                executor_tools=self.tools,
                analysis_tools=self.data_tools,
            )
        else:
            self.workflow = None

    def _resolve_turn_tools(
        self,
        tools: Collection[Any] | None,
        data_tools: Collection[Any] | None,
    ) -> list[Any] | None:
        """Resolve the LangGraph tools for run_turn-backed workflows."""
        if self.workflow_type == "single_agent":
            return list(tools) if tools is not None else None
        if self.workflow_type == "python_relp":
            return _merge_tools(tools, [repl_tool, calculator])
        if self.workflow_type == "graspa":
            return _merge_tools(tools, [run_graspa])
        if self.workflow_type == "mock_agent":
            return _merge_tools(tools, _mock_tools())
        if self.workflow_type == "single_agent_mcp":
            resolved = _merge_tools(tools, data_tools)
            if not resolved:
                raise ValueError(
                    "No MCP tools loaded. Ensure MCP servers are configured and reachable."
                )
            return resolved
        if self.workflow_type == "rag_agent":
            return _merge_tools(tools, _rag_tools())
        if self.workflow_type == "single_agent_xanes":
            return _merge_tools(tools, _xanes_tools())
        return list(tools) if tools is not None else None

    def _resolve_turn_prompts(self) -> None:
        """Apply workflow-specific prompt defaults before run_turn."""
        if self.workflow_type == "rag_agent" and self._using_default_system_prompt:
            self.system_prompt = rag_agent_prompt
        elif self.workflow_type == "single_agent_xanes":
            if self._using_default_system_prompt:
                self.system_prompt = default_xanes_single_agent_prompt
            if self._using_default_formatter_prompt:
                self.formatter_prompt = default_xanes_formatter_prompt

    def visualize(self, method: str = "ascii"):
        """Visualize the LangGraph graph structure.

        This method creates and displays a visual representation of the workflow graph
        using Mermaid diagrams. The visualization is shown in Jupyter notebooks.

        Parameters
        ----------
        method : str, optional
            Visualization backend. ``"ascii"`` returns an ASCII graph;
            any other value renders a Mermaid PNG in the active notebook.

        Returns
        -------
        str or None
            ASCII graph text when ``method`` is ``"ascii"``; otherwise
            displays an image and returns ``None``.

        Notes
        -----
        Requires IPython and nest_asyncio to be installed.
        The visualization uses Mermaid diagrams with custom styling.
        """
        if self.workflow is None:
            raise RuntimeError(
                f"Workflow {self.workflow_type!r} is run-turn-backed and is built "
                "inside ChemGraph.run(); it is not available for pre-run visualization."
            )
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
        if self.workflow is None:
            if self._last_run_state is None:
                raise RuntimeError(
                    f"Workflow {self.workflow_type!r} has not produced state yet."
                )
            return self._last_run_state
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
        """Create a session record on first run if memory is enabled.

        Parameters
        ----------
        query : str
            User query used to generate the session title.
        """
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
        """Extract messages from workflow state and persist to session store.

        Parameters
        ----------
        last_state : dict
            Latest LangGraph state containing a ``messages`` sequence.
        query : str
            Original user query associated with the saved messages.
        """
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

        Parameters
        ----------
        question : str
            Prompt emitted by the graph for a human response.

        Returns
        -------
        str
            Human response returned by the configured handler.
        """
        handler = self.human_input_handler
        if handler is None:
            raise HumanInputRequired(question)
        if asyncio.iscoroutinefunction(handler):
            return await handler(question)
        return handler(question)

    async def run(
        self,
        query: str,
        config=None,
        resume_from: Optional[str] = None,
    ):
        """
        Async runner for run-turn-backed and legacy graph-backed workflows.

        Run-turn-backed workflows delegate to :func:`run_turn`, while legacy
        multi-node graph workflows stream through ``self.workflow.astream``.
        The return value follows ``self.return_option`` ("last_message" or
        "state").

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
        if config is None:
            config = {}
        if not isinstance(config, dict):
            raise TypeError(f"`config` must be a dictionary, got {type(config).__name__}")
        if "thread_id" in config:
            config.setdefault("configurable", {})["thread_id"] = str(config["thread_id"])
        config.setdefault("configurable", {}).setdefault("thread_id", "1")
        config["recursion_limit"] = self.recursion_limit

        if not os.environ.get("CHEMGRAPH_LOG_DIR"):
            os.environ["CHEMGRAPH_LOG_DIR"] = self.log_dir

        self._ensure_session(query)
        if resume_from and self.session_store:
            context = self.session_store.build_context_summary(resume_from)
            if context:
                query = (
                    f"{context}\n\n"
                    f"Now, continuing from the previous session above, "
                    f"please help with the following:\n\n{query}"
                )
                logger.info(f"Injected context from session {resume_from}")

        thread_id = str(config["configurable"]["thread_id"])
        if self.workflow_type in SINGLE_AGENT_TURN_WORKFLOWS:
            result = await run_turn(
                query=query,
                tools=self.tools,
                model_name=self.model_name,
                base_url=self.base_url,
                api_key=self.api_key,
                argo_user=self.argo_user,
                system_prompt=self.system_prompt,
                formatter_prompt=self.formatter_prompt,
                structured_output=self.structured_output,
                generate_report=self.generate_report,
                report_prompt=self.report_prompt,
                recursion_limit=self.recursion_limit,
                thread_id=thread_id,
                terminal_tool_names=self.terminal_tool_names,
                human_supervised=self.human_supervised,
            )
            self._last_run_state = result.state
            self._save_messages_to_store(result.state, query)
            self.write_state(config=config, file_path=None)
            if self.return_option == "state":
                return result.state
            if self.return_option == "last_message":
                return _last_ai_message(result.state, result.final_text)
            raise ValueError(
                f"Unsupported return_option: {self.return_option}. "
                "Use 'last_message' or 'state'."
            )

        try:
            last_state = None
            async for state in self.workflow.astream(
                {"messages": query},
                stream_mode="values",
                config=config,
            ):
                if "messages" in state:
                    for message in state["messages"][-1:]:
                        try:
                            message.pretty_print()
                        except Exception:
                            pass
                        logger.info(message)
                last_state = state
            if last_state is None:
                raise RuntimeError("Workflow produced no states")
            self._last_run_state = serialize_state(last_state)
            self._save_messages_to_store(last_state, query)
            self.write_state(config=config, file_path=None)
            if self.return_option == "state":
                return serialize_state(self.get_state(config=config))
            if self.return_option == "last_message":
                return last_state["messages"][-1]
            raise ValueError(
                f"Unsupported return_option: {self.return_option}. "
                "Use 'last_message' or 'state'."
            )
        except GraphInterrupt:
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
        """Initialize the exception with the pending human question.

        Parameters
        ----------
        question : str
            Question that should be presented to the user.
        """
        self.question = question
        super().__init__(question)
