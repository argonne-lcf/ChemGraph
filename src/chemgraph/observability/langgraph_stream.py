"""Live LangGraph/LangChain event emission for ChemGraph workflows."""

from __future__ import annotations

import json
import math
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from chemgraph.observability.events import emit_workflow_event
from chemgraph.observability.events import new_span_id


def _compact(value: Any, *, max_chars: int = 1000) -> Any:
    try:
        text = json.dumps(value, default=str, sort_keys=True)
    except TypeError:
        text = str(value)
    if len(text) <= max_chars:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return {
        "truncated": True,
        "preview": text[:max_chars],
    }


def _message_type(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("type") or message.get("role") or "")
    return str(getattr(message, "type", "") or getattr(message, "role", ""))


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _message_tool_calls(message: Any) -> list[dict[str, Any]]:
    calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if not isinstance(calls, list):
        return []
    normalized = []
    for call in calls:
        if isinstance(call, dict):
            normalized.append(
                {
                    "name": call.get("name"),
                    "id": call.get("id"),
                    "args": _compact(call.get("args") or {}, max_chars=2000),
                },
            )
        else:
            normalized.append({"name": str(call), "id": None, "args": {}})
    return normalized


def _message_usage_metadata(message: Any) -> dict[str, Any]:
    usage = (
        message.get("usage_metadata")
        if isinstance(message, dict)
        else getattr(message, "usage_metadata", None)
    )
    if isinstance(usage, dict) and usage:
        return usage
    response_metadata = (
        message.get("response_metadata")
        if isinstance(message, dict)
        else getattr(message, "response_metadata", None)
    )
    if not isinstance(response_metadata, dict):
        return {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    return token_usage if isinstance(token_usage, dict) else {}


def _usage_int(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _text_for_token_estimate(value: Any) -> str:
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except TypeError:
        return str(value)


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except TypeError:
        return str(value)


def _serialize_message(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return _json_safe(message)
    if hasattr(message, "model_dump"):
        return _json_safe(message.model_dump(mode="json"))
    return {
        "type": _message_type(message),
        "content": _json_safe(_message_content(message)),
        "tool_calls": _message_tool_calls(message),
    }


def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    return [_serialize_message(message) for message in messages]


def _estimate_tokens(text: str) -> int:
    try:
        import tiktoken  # type: ignore[import-not-found]

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 4))


def _llm_token_counts(
    *,
    previous_messages: list[Any],
    message: Any,
    tool_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    usage = _message_usage_metadata(message)
    provider_input = _usage_int(usage, "input_tokens", "prompt_tokens")
    provider_output = _usage_int(usage, "output_tokens", "completion_tokens")
    provider_total = _usage_int(usage, "total_tokens")
    if provider_input is not None or provider_output is not None or provider_total is not None:
        input_tokens = provider_input
        output_tokens = provider_output
        if provider_total is None:
            provider_total = (input_tokens or 0) + (output_tokens or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": provider_total,
            "source": "provider",
            "raw_usage": _compact(usage, max_chars=1000),
        }

    input_text = _text_for_token_estimate(previous_messages)
    output_text = _text_for_token_estimate(
        {
            "content": _message_content(message),
            "tool_calls": tool_calls,
        },
    )
    input_tokens = _estimate_tokens(input_text)
    output_tokens = _estimate_tokens(output_text)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "source": "local_estimate",
        "estimate_scope": "langgraph_state_messages",
    }


def emit_live_message_events(
    *,
    previous_messages: list[Any],
    current_messages: list[Any],
    workflow_span_id: str,
) -> int:
    """Emit live workflow events for newly streamed LangGraph messages."""
    if len(current_messages) <= len(previous_messages):
        return 0
    count = 0
    for index, message in enumerate(
        current_messages[len(previous_messages) :],
        start=len(previous_messages),
    ):
        message_type = _message_type(message)
        if message_type != "ai":
            continue
        tool_calls = _message_tool_calls(message)
        token_counts = _llm_token_counts(
            previous_messages=current_messages[:index],
            message=message,
            tool_calls=tool_calls,
        )
        prompt_messages = _serialize_messages(current_messages[:index])
        if tool_calls:
            emit_workflow_event(
                "llm_decision",
                {
                    "workflow_node": "ChemGraphAgent",
                    "message_index": index,
                    "tool_calls": tool_calls,
                    "token_counts": token_counts,
                    "prompt_messages": prompt_messages,
                },
                span_id=new_span_id("chemgraph-llm-decision"),
                parent_span_id=workflow_span_id,
            )
            count += 1
            continue
        content = _message_content(message)
        if content:
            emit_workflow_event(
                "workflow_output",
                {
                    "workflow_node": "ChemGraphAgent",
                    "message_index": index,
                    "content_preview": str(content)[:2000],
                    "token_counts": token_counts,
                    "prompt_messages": prompt_messages,
                },
                span_id=new_span_id("chemgraph-output"),
                parent_span_id=workflow_span_id,
            )
            count += 1
    return count


def _tool_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
    serialized = serialized or {}
    value = (
        serialized.get("name")
        or serialized.get("id")
        or kwargs.get("name")
        or kwargs.get("tool_name")
    )
    if isinstance(value, list) and value:
        value = value[-1]
    return str(value or "tool")


def _run_id_text(run_id: UUID | str | None) -> str:
    return str(run_id) if run_id is not None else new_span_id("tool-run")


class ChemGraphWorkflowCallback(BaseCallbackHandler):
    """Emit live tool lifecycle events for a ChemGraph LangGraph run."""

    def __init__(self, *, workflow_span_id: str) -> None:
        self.workflow_span_id = workflow_span_id
        self._tool_runs: dict[str, dict[str, Any]] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_run_id = _run_id_text(run_id)
        tool_name = _tool_name(serialized, kwargs)
        span_id = f"chemgraph-tool-call-{tool_run_id}"
        self._tool_runs[tool_run_id] = {
            "tool_name": tool_name,
            "span_id": span_id,
        }
        emit_workflow_event(
            "tool_call_started",
            {
                "workflow_node": "tools",
                "tool_name": tool_name,
                "tool_call_id": tool_run_id,
                "parent_tool_run_id": _run_id_text(parent_run_id)
                if parent_run_id
                else None,
                "input": _compact(inputs if inputs is not None else input_str),
            },
            span_id=span_id,
            parent_span_id=self.workflow_span_id,
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_run_id = _run_id_text(run_id)
        tool_run = self._tool_runs.get(tool_run_id, {})
        tool_name = str(tool_run.get("tool_name") or _tool_name(None, kwargs))
        span_id = str(
            tool_run.get("span_id") or f"chemgraph-tool-call-{tool_run_id}",
        )
        emit_workflow_event(
            "tool_call_finished",
            {
                "workflow_node": "tools",
                "tool_name": tool_name,
                "tool_call_id": tool_run_id,
                "parent_tool_run_id": _run_id_text(parent_run_id)
                if parent_run_id
                else None,
                "content_preview": str(_compact(output))[:2000],
            },
            span_id=span_id,
            parent_span_id=self.workflow_span_id,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_run_id = _run_id_text(run_id)
        tool_run = self._tool_runs.get(tool_run_id, {})
        tool_name = str(tool_run.get("tool_name") or _tool_name(None, kwargs))
        span_id = str(
            tool_run.get("span_id") or f"chemgraph-tool-call-{tool_run_id}",
        )
        emit_workflow_event(
            "tool_call_failed",
            {
                "workflow_node": "tools",
                "tool_name": tool_name,
                "tool_call_id": tool_run_id,
                "parent_tool_run_id": _run_id_text(parent_run_id)
                if parent_run_id
                else None,
                "error": repr(error),
            },
            span_id=span_id,
            parent_span_id=self.workflow_span_id,
        )
