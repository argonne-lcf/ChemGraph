"""Checkpointed reviews for tool nodes in legacy specialist graphs."""

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from typing import TypedDict


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
    if not interrupt_on:
        return ToolNode(tools, **kwargs)
    review = HumanInTheLoopMiddleware(interrupt_on=interrupt_on)

    def review_batch(state, runtime: Runtime):
        return review.after_model(state, runtime) or {}

    def skip_answered(request, handler):
        _, answered = _answered_calls(request.state)
        previous = answered.get(request.tool_call["id"])
        return previous if previous is not None else handler(request)

    async def askip_answered(request, handler):
        _, answered = _answered_calls(request.state)
        previous = answered.get(request.tool_call["id"])
        return previous if previous is not None else await handler(request)

    def route(state):
        message, answered = _answered_calls(state)
        return "execute" if message and any(
            call["id"] not in answered for call in message.tool_calls
        ) else END

    # ToolNode returns messages, not the incoming planner's accumulated results.
    builder = StateGraph(state_schema, output_schema=_ToolOutput)
    builder.add_node("review", review_batch)
    builder.add_node("execute", ToolNode(
        tools, wrap_tool_call=skip_answered, awrap_tool_call=askip_answered, **kwargs,
    ))
    builder.add_edge(START, "review")
    builder.add_conditional_edges("review", route)
    builder.add_edge("execute", END)
    return builder.compile(checkpointer=None)
