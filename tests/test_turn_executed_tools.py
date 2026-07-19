"""Regression tests for executed-tool extraction in agent.turn.

Guards against counting a *named non-tool message* (e.g. a named AIMessage or
HumanMessage in a multi-agent flow) as an executed tool.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from chemgraph.agent.turn import _executed_tool_names, _tool_message_name


def test_tool_message_name_counts_tool_messages():
    msg = ToolMessage(content="42", name="run_ase", tool_call_id="c1")
    assert _tool_message_name(msg) == "run_ase"


def test_tool_message_name_ignores_named_ai_message():
    # A named AIMessage (common for named nodes / multi-agent peers) is NOT a tool.
    assert _tool_message_name(AIMessage(content="hi", name="agent-a")) is None


def test_tool_message_name_ignores_named_human_message():
    assert _tool_message_name(HumanMessage(content="hi", name="peer-b")) is None


def test_tool_message_name_ignores_named_dict_message():
    assert _tool_message_name({"role": "assistant", "name": "agent-a"}) is None
    assert _tool_message_name({"role": "tool", "name": "run_ase"}) == "run_ase"


def test_executed_tool_names_from_tool_messages():
    messages = [
        HumanMessage(content="do it", name="peer-b"),
        AIMessage(
            content="",
            tool_calls=[{"name": "run_ase", "id": "c1", "args": {}}],
        ),
        ToolMessage(content="ok", name="run_ase", tool_call_id="c1"),
        AIMessage(content="done", name="agent-a"),
    ]
    # Only the real tool result is counted; the named AI/Human messages are not.
    assert _executed_tool_names(messages) == ("run_ase",)


def test_executed_tool_names_falls_back_to_tool_calls():
    # No tool *result* messages yet -> fall back to the AI message's tool_calls,
    # and the named messages still must not leak in.
    messages = [
        HumanMessage(content="do it", name="peer-b"),
        AIMessage(
            content="",
            name="agent-a",
            tool_calls=[{"name": "run_ase", "id": "c1", "args": {}}],
        ),
    ]
    assert _executed_tool_names(messages) == ("run_ase",)


def test_executed_tool_names_empty_when_no_tools():
    messages = [
        HumanMessage(content="hi", name="peer-b"),
        AIMessage(content="just chatting", name="agent-a"),
    ]
    assert _executed_tool_names(messages) == ()
