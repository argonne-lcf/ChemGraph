"""Checkpointed reviews for tool nodes in legacy specialist graphs."""

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import _handle_tool_error, _infer_handled_types
from langgraph.runtime import Runtime
from typing import TypedDict

from chemgraph.execution.middleware import ScopedToolsMiddleware
from chemgraph.execution.scoped import ScopedExecutionError


class _ToolOutput(TypedDict):
    messages: list


def _answered_calls(state):
    answered = {}
    for message in reversed(state.get("messages", [])):
        if isinstance(message, AIMessage):
            return message, answered
        if isinstance(message, ToolMessage):
            answered.setdefault(message.tool_call_id, message)
    return None, answered


def reviewed_tool_node(tools, interrupt_on=None, *, state_schema, **kwargs):
    """Keep standalone ToolNode behavior; add reviews when a parent supplies policy.

    Rejection creates a ToolMessage while retaining the original tool call for
    message protocol correctness. ToolNode itself does not skip answered calls.
    """
    scoped = ScopedToolsMiddleware(tools=tools)

    def skip_answered(request, handler):
        _, answered = _answered_calls(request.state)
        previous = answered.get(request.tool_call["id"]) if interrupt_on else None
        return previous if previous is not None else scoped.wrap_tool_call(request, handler)

    async def askip_answered(request, handler):
        _, answered = _answered_calls(request.state)
        previous = answered.get(request.tool_call["id"]) if interrupt_on else None
        return previous if previous is not None else await scoped.awrap_tool_call(request, handler)

    node = ToolNode(
        tools, wrap_tool_call=skip_answered, awrap_tool_call=askip_answered, **kwargs,
    )
    original_errors = node._handle_tool_errors

    def handle_errors(exc: Exception) -> str:
        if isinstance(exc, (ScopedExecutionError, GraphBubbleUp)) or not original_errors:
            raise exc
        if isinstance(original_errors, type):
            handled = (original_errors,)
        elif isinstance(original_errors, tuple):
            handled = original_errors
        elif callable(original_errors):
            handled = _infer_handled_types(original_errors)
        else:
            handled = (Exception,)
        if not isinstance(exc, handled):
            raise exc
        return _handle_tool_error(exc, flag=original_errors)

    node._handle_tool_errors = handle_errors
    if not interrupt_on:
        return node
    review = HumanInTheLoopMiddleware(interrupt_on=interrupt_on)

    def review_batch(state, runtime: Runtime):
        return review.after_model(state, runtime) or {}

    def route(state):
        message, answered = _answered_calls(state)
        return "execute" if message and any(
            call["id"] not in answered for call in message.tool_calls
        ) else END

    # ToolNode returns messages, not the incoming planner's accumulated results.
    builder = StateGraph(state_schema, output_schema=_ToolOutput)
    builder.add_node("review", review_batch)
    builder.add_node("execute", node)
    builder.add_edge(START, "review")
    builder.add_conditional_edges("review", route)
    builder.add_edge("execute", END)
    return builder.compile(checkpointer=None)
