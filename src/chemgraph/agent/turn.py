from __future__ import annotations

import dataclasses
import datetime
import logging
import os
import time
import uuid
from typing import Any, Collection

from chemgraph.graphs.single_agent import construct_single_agent_graph
from chemgraph.models.loader import load_chat_model
from chemgraph.models.settings import LLMSettings
from chemgraph.prompt.single_agent_prompt import (
    formatter_prompt as default_formatter_prompt,
)
from chemgraph.prompt.single_agent_prompt import report_prompt as default_report_prompt
from chemgraph.prompt.single_agent_prompt import single_agent_prompt

logger = logging.getLogger(__name__)


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
    # GPT-5* / o-series reject any non-default temperature + sampling
    # knobs. Drop them so the request payload matches what the model
    # accepts. Import is local to avoid an import cycle with
    # chemgraph.models.openai which itself imports langchain_openai.
    from chemgraph.models.openai import is_reasoning_model
    if is_reasoning_model(model_name):
        for k in ("temperature", "top_p", "frequency_penalty", "presence_penalty"):
            kwargs.pop(k, None)
    return kwargs


@dataclasses.dataclass(frozen=True)
class TurnResult:
    """Result of one bounded ChemGraph single-agent turn."""

    final_text: str
    state: dict[str, Any]
    executed_tool_names: tuple[str, ...]
    terminal_tool: str | None
    thread_id: str
    duration_s: float


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


def _response_tool_calls(response: Any) -> list[dict[str, str | None]]:
    try:
        generations = getattr(response, "generations", None) or []
        tool_calls: list[dict[str, str | None]] = []
        for generation_group in generations:
            for generation in generation_group or []:
                message = getattr(generation, "message", None)
                for call in _message_tool_calls(message):
                    name = _call_name(call)
                    if not name:
                        continue
                    tool_calls.append(
                        {
                            "name": name,
                            "id": _call_id(call),
                        },
                    )
        return tool_calls
    except Exception:  # noqa: BLE001 - event extraction must not break runs.
        logger.debug("failed to extract llm_decision tool calls", exc_info=True)
        return []


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


def _call_id(call: Any) -> str | None:
    if isinstance(call, dict):
        value = call.get("id") or call.get("tool_call_id")
    else:
        value = getattr(call, "id", None) or getattr(call, "tool_call_id", None)
    return str(value) if value else None


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
    try:
        return load_chat_model(
            settings=LLMSettings(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                argo_user=argo_user,
                temperature=temperature,
            ),
        )
    except ValueError:
        pass

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


from chemgraph.agent.events import EventCallback, _TurnEventCallback


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

