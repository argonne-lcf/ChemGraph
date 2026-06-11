from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

EventCallback = Callable[[str, dict], None]


def _serialized_name(serialized: Any) -> str | None:
    from chemgraph.agent.turn import _serialized_name as turn_serialized_name

    return turn_serialized_name(serialized)


def _response_tool_calls(response: Any) -> list[dict[str, str | None]]:
    from chemgraph.agent.turn import _response_tool_calls as turn_response_tool_calls

    return turn_response_tool_calls(response)


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
        if tool_calls := _response_tool_calls(response):
            self._emit("llm_decision", {"tool_calls": tool_calls})

    def on_llm_error(self, error, **kwargs) -> None:
        self._emit("llm_call_failed", {"error": repr(error)})

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self._emit(
            "tool_call_started",
            {
                "tool_name": _serialized_name(serialized),
                "arguments": _serialize_state(input_str),
            },
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
