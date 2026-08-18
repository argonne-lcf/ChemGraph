from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, dict], None]
SUBAGENT_METADATA_KEY = "chemgraph_subagent"


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


def _serialize_state(value: Any) -> Any:
    from chemgraph.agent.turn import serialize_state

    return serialize_state(value)


class _BaseDashboardEventCallback(BaseCallbackHandler):
    """Forward LangChain callback events to the dashboard event surface."""

    _failure_log_message = "dashboard event callback failed"

    def __init__(self, on_event: EventCallback, thread_id: str) -> None:
        self._on_event = on_event
        self._thread_id = thread_id

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        try:
            self._on_event(event, {"thread_id": self._thread_id, **payload})
        except Exception:  # noqa: BLE001 - callbacks must not break the run.
            logger.debug(self._failure_log_message, exc_info=True)

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
        # Only surface an llm_decision when the model actually requested tool
        # calls; a plain text answer has no decision to report.
        tool_calls = _response_tool_calls(response)
        if tool_calls:
            self._emit("llm_decision", {"tool_calls": tool_calls})

    def on_llm_error(self, error, **kwargs) -> None:
        self._emit("llm_call_failed", {"error": repr(error)})

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        payload = {
            "tool_name": _serialized_name(serialized),
            "arguments": _serialize_state(input_str),
        }
        metadata = kwargs.get("metadata")
        if isinstance(metadata, dict) and metadata.get(SUBAGENT_METADATA_KEY):
            payload["subagent_name"] = str(metadata[SUBAGENT_METADATA_KEY])
        self._emit(
            "tool_call_started",
            payload,
        )

    def on_tool_end(self, output, **kwargs) -> None:
        payload: dict[str, Any] = {"result": _serialize_state(output)}
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


class _TurnEventCallback(_BaseDashboardEventCallback):
    """Forward run_turn callback events to the dashboard event surface."""

    _failure_log_message = "turn event callback failed"


class _AstreamEventCallback(_BaseDashboardEventCallback):
    """Forward graph stream callback events to the dashboard event surface."""

    _failure_log_message = "astream event callback failed"
