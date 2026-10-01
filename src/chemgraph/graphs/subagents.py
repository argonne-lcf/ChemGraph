"""Worker result adaptation and recording shared by agent entry points."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from deepagents.middleware.subagents import CompiledSubAgent
from langchain_core.messages import AIMessage, ToolMessage, convert_to_messages
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langgraph.errors import GraphInterrupt

from chemgraph.agent.events import SUBAGENT_METADATA_KEY
from chemgraph.memory.subagent_recorder import SubagentRunRecorder


def latest_assistant_text(messages: list[Any]) -> str:
    """Return the latest assistant message text from a message history."""
    for message in reversed(convert_to_messages(messages)):
        if isinstance(message, AIMessage):
            return str(message.text)
    return ""


def _preserve_terminal_tool_output(result: Any) -> Any:
    """Expose trailing worker tool output to Deep Agents as a final AI message."""
    if not isinstance(result, dict) or result.get("structured_response") is not None:
        return result

    messages = convert_to_messages(result.get("messages", []))
    if not messages or not isinstance(messages[-1], ToolMessage):
        return result

    trailing_text: list[str] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        if text := str(message.text).strip():
            trailing_text.append(text)
    if not trailing_text:
        return result

    return {
        **result,
        "messages": [
            *messages,
            AIMessage(content="\n".join(reversed(trailing_text))),
        ],
    }


def _adapt_subagent(
    spec: CompiledSubAgent,
    recorder: SubagentRunRecorder | None = None,
) -> CompiledSubAgent:
    runnable = spec["runnable"]

    def child_config(config: RunnableConfig) -> RunnableConfig:
        adapted_config = dict(config)
        metadata = dict(adapted_config.get("metadata") or {})
        metadata[SUBAGENT_METADATA_KEY] = spec["name"]
        adapted_config["metadata"] = metadata
        return adapted_config

    def invoke(state: Any, config: RunnableConfig) -> Any:
        run_id = recorder.start(spec["name"], state, config) if recorder else None
        try:
            result = runnable.invoke(state, config=child_config(config))
        except GraphInterrupt:
            if recorder and run_id:
                recorder.interrupted(run_id)
            raise
        except Exception as exc:
            if recorder and run_id:
                recorder.failed(run_id, exc)
            raise
        result = _preserve_terminal_tool_output(result)
        if recorder and run_id:
            recorder.completed(run_id, result)
        return result

    async def ainvoke(state: Any, config: RunnableConfig) -> Any:
        run_id = recorder.start(spec["name"], state, config) if recorder else None
        try:
            result = await runnable.ainvoke(state, config=child_config(config))
        except GraphInterrupt:
            if recorder and run_id:
                recorder.interrupted(run_id)
            raise
        except Exception as exc:
            if recorder and run_id:
                recorder.failed(run_id, exc)
            raise
        result = _preserve_terminal_tool_output(result)
        if recorder and run_id:
            recorder.completed(run_id, result)
        return result

    return {
        "name": spec["name"],
        "description": spec["description"],
        "runnable": RunnableLambda(invoke, afunc=ainvoke),
    }


def _validate_subagents(
    subagents: Sequence[CompiledSubAgent],
    recorder: SubagentRunRecorder | None = None,
) -> list[CompiledSubAgent]:
    validated: list[CompiledSubAgent] = []
    names: set[str] = set()
    for spec in subagents:
        name = spec.get("name")
        description = spec.get("description")
        runnable = spec.get("runnable")
        if not isinstance(name, str):
            raise TypeError("Subagent names must be strings.")
        if not name:
            raise ValueError("Subagent names must not be empty.")
        if name != name.strip():
            raise ValueError("Subagent names must not have surrounding whitespace.")
        if name in names:
            raise ValueError(f"Duplicate subagent name: {name!r}.")
        if not isinstance(description, str):
            raise TypeError(f"Subagent {name!r} must have a string description.")
        if not description.strip():
            raise ValueError(f"Subagent {name!r} must have a description.")
        if not callable(getattr(runnable, "invoke", None)) or not callable(
            getattr(runnable, "ainvoke", None)
        ):
            raise TypeError(
                f"Subagent {name!r} must provide invoke and ainvoke methods."
            )
        names.add(name)
        validated.append(_adapt_subagent(spec, recorder))
    return validated
