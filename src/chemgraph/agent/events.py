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


def _as_usage_dict(value: Any) -> dict[str, Any]:
    """Return a provider usage value as a plain dictionary."""
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return {}
    try:
        try:
            dumped = model_dump(mode="json")
        except TypeError:
            dumped = model_dump()
    except Exception:  # noqa: BLE001 - usage extraction must not break runs.
        logger.debug("failed to serialize provider token usage", exc_info=True)
        return {}
    return dumped if isinstance(dumped, dict) else {}


def _usage_int(usage: dict[str, Any], *keys: str) -> int | None:
    """Return the first integer token count under ``keys``."""
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _message_usage(message: Any) -> dict[str, Any]:
    """Read canonical or provider-native usage from one model message."""
    if isinstance(message, dict):
        usage = message.get("usage_metadata")
        response_metadata = message.get("response_metadata")
    else:
        usage = getattr(message, "usage_metadata", None)
        response_metadata = getattr(message, "response_metadata", None)

    if usage_dict := _as_usage_dict(usage):
        return usage_dict

    metadata = _as_usage_dict(response_metadata)
    for key in ("token_usage", "usage"):
        if usage_dict := _as_usage_dict(metadata.get(key)):
            return usage_dict
    return {}


def _response_usage(response: Any) -> dict[str, Any]:
    """Return the highest-fidelity usage mapping from an LLM response."""
    try:
        for generation_group in getattr(response, "generations", None) or []:
            for generation in generation_group or []:
                if usage := _message_usage(getattr(generation, "message", None)):
                    return usage
    except Exception:  # noqa: BLE001 - usage extraction must not break runs.
        logger.debug("failed to extract message token usage", exc_info=True)

    llm_output = _as_usage_dict(getattr(response, "llm_output", None))
    for key in ("token_usage", "usage"):
        if usage := _as_usage_dict(llm_output.get(key)):
            return usage
    if any(
        key in llm_output
        for key in (
            "input_tokens",
            "prompt_tokens",
            "output_tokens",
            "completion_tokens",
            "total_tokens",
        )
    ):
        return llm_output
    return {}


def _response_token_counts(response: Any) -> dict[str, Any] | None:
    """Normalize provider token usage for LLM-finished event payloads."""
    usage = _response_usage(response)
    input_tokens = _usage_int(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_int(usage, "output_tokens", "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens")
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    counts: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "source": "provider",
        "raw_usage": _serialize_state(usage),
    }

    input_details = _as_usage_dict(usage.get("input_token_details"))
    cache_read = _usage_int(usage, "cache_read_input_tokens")
    if cache_read is None:
        cache_read = _usage_int(input_details, "cache_read")

    cache_creation = _usage_int(usage, "cache_creation_input_tokens")
    if cache_creation is None:
        ephemeral_5m = _usage_int(input_details, "ephemeral_5m_input_tokens")
        ephemeral_1h = _usage_int(input_details, "ephemeral_1h_input_tokens")
        if ephemeral_5m is not None or ephemeral_1h is not None:
            cache_creation = (ephemeral_5m or 0) + (ephemeral_1h or 0)
        else:
            cache_creation = _usage_int(input_details, "cache_creation")

    if cache_creation is not None:
        counts["cache_creation_input_tokens"] = cache_creation
    if cache_read is not None:
        counts["cache_read_input_tokens"] = cache_read
    return counts


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
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            payload["llm_output"] = llm_output
        if token_counts := _response_token_counts(response):
            payload["token_counts"] = token_counts
        self._emit("llm_call_finished", payload)
        if tool_calls := _response_tool_calls(response):
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
