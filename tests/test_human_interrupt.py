"""Tests for the human-in-the-loop interrupt/resume functionality.

Tests cover:
- The ``ask_human`` tool (interrupt mechanism)
- The ``PlannerResponse`` schema with ``ask_human`` next_step
- The ``human_review_node`` in the multi-agent graph
- The ``planner_agent`` returning ``ask_human`` routing
- Single-agent and multi-agent graph construction with interrupt support
"""

import json

import pytest
from langchain_core.messages import AIMessage

from chemgraph.schemas.multi_agent_response import PlannerResponse
from chemgraph.graphs.multi_agent import (
    planner_agent,
    human_review_node,
    unified_planner_router,
)


# ---------------------------------------------------------------------------
# PlannerResponse schema tests for ask_human
# ---------------------------------------------------------------------------


def test_planner_response_ask_human():
    """PlannerResponse should accept next_step='ask_human' with clarification."""
    payload = {
        "thought_process": "User did not specify calculator or temperature.",
        "next_step": "ask_human",
        "clarification": "Which calculator should I use (MACE, xTB, or EMT)?",
    }
    parsed = PlannerResponse.model_validate(payload)
    assert parsed.next_step == "ask_human"
    assert parsed.clarification is not None
    assert "calculator" in parsed.clarification
    assert parsed.tasks is None


def test_planner_response_ask_human_with_question_key():
    """PlannerResponse should accept legacy 'question' key as clarification."""
    payload = {
        "thought_process": "Molecule name is ambiguous.",
        "next_step": "ask_human",
        "question": "Did you mean ethanol or ethane?",
    }
    parsed = PlannerResponse.model_validate(payload)
    assert parsed.next_step == "ask_human"
    assert parsed.clarification == "Did you mean ethanol or ethane?"


def test_planner_response_ask_human_no_clarification():
    """PlannerResponse should allow ask_human with clarification=None."""
    payload = {
        "thought_process": "Need more info.",
        "next_step": "ask_human",
    }
    parsed = PlannerResponse.model_validate(payload)
    assert parsed.next_step == "ask_human"
    assert parsed.clarification is None


# ---------------------------------------------------------------------------
# planner_agent tests for ask_human routing
# ---------------------------------------------------------------------------

_ASK_HUMAN_JSON = json.dumps(
    {
        "thought_process": "The user did not specify which calculator to use.",
        "next_step": "ask_human",
        "clarification": "Which calculator should I use? Options: MACE, xTB, EMT.",
    }
)


class _DummyResponse:
    """Mimics the object returned by ``llm.invoke(messages)``."""

    def __init__(self, content: str):
        self.content = content


class _DummyLLM:
    """Returns a fixed raw-text response."""

    def __init__(self, content: str):
        self._content = content

    def invoke(self, _messages):
        return _DummyResponse(self._content)


def test_planner_agent_ask_human():
    """planner_agent should return next_step='ask_human' and clarification."""
    llm = _DummyLLM(_ASK_HUMAN_JSON)
    state = {"messages": [{"role": "user", "content": "Calculate energy of water"}]}
    out = planner_agent(state=state, llm=llm, system_prompt="planner")

    assert out["next_step"] == "ask_human"
    assert out["clarification"] is not None
    assert "calculator" in out["clarification"].lower()
    assert out["tasks"] == []


# ---------------------------------------------------------------------------
# unified_planner_router tests for ask_human
# ---------------------------------------------------------------------------


def test_unified_planner_router_ask_human():
    """Router should return 'human_review' when next_step is 'ask_human'."""
    state = {
        "messages": [],
        "next_step": "ask_human",
        "tasks": [],
        "executor_results": [],
        "executor_logs": {},
        "planner_iterations": 0,
        "clarification": "Which calculator?",
    }
    result = unified_planner_router(state, structured_output=False)
    assert result == "human_review"


def test_unified_planner_router_ask_human_with_structured_output():
    """Router should return 'human_review' regardless of structured_output."""
    state = {
        "messages": [],
        "next_step": "ask_human",
        "tasks": [],
        "executor_results": [],
        "executor_logs": {},
        "planner_iterations": 0,
        "clarification": "Which method?",
    }
    result = unified_planner_router(state, structured_output=True)
    assert result == "human_review"


# ---------------------------------------------------------------------------
# human_review_node tests
# ---------------------------------------------------------------------------


def test_human_review_node_calls_interrupt(monkeypatch):
    """human_review_node should call interrupt() with the clarification question."""
    captured_values = []

    def fake_interrupt(value):
        captured_values.append(value)
        # Simulate a human response
        return "Use MACE calculator"

    monkeypatch.setattr("chemgraph.graphs.multi_agent.interrupt", fake_interrupt)

    state = {
        "messages": [],
        "clarification": "Which calculator should I use?",
    }
    result = human_review_node(state)

    assert len(captured_values) == 1
    assert captured_values[0] == {"question": "Which calculator should I use?"}
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert "Use MACE calculator" in result["messages"][0].content


def test_human_review_node_default_question(monkeypatch):
    """human_review_node should use a default question when clarification is missing."""
    captured_values = []

    def fake_interrupt(value):
        captured_values.append(value)
        return "Some answer"

    monkeypatch.setattr("chemgraph.graphs.multi_agent.interrupt", fake_interrupt)

    state = {"messages": []}
    result = human_review_node(state)

    assert "provide more details" in captured_values[0]["question"].lower()


def test_human_review_node_dict_response(monkeypatch):
    """human_review_node should handle dict responses from the human."""

    def fake_interrupt(value):
        return {"answer": "Use xTB"}

    monkeypatch.setattr("chemgraph.graphs.multi_agent.interrupt", fake_interrupt)

    state = {"messages": [], "clarification": "Which calculator?"}
    result = human_review_node(state)

    assert "Use xTB" in result["messages"][0].content


# ---------------------------------------------------------------------------
# ask_human tool tests
# ---------------------------------------------------------------------------


def test_ask_human_tool_exists():
    """ask_human tool should be importable from generic_tools."""
    from chemgraph.tools.generic_tools import ask_human

    assert ask_human is not None
    assert ask_human.name == "ask_human"


def test_ask_human_tool_calls_interrupt(monkeypatch):
    """ask_human tool should call interrupt() with the question."""
    from chemgraph.tools import generic_tools

    captured = []

    def fake_interrupt(value):
        captured.append(value)
        return "42 degrees"

    monkeypatch.setattr(generic_tools, "interrupt", fake_interrupt)

    result = generic_tools.ask_human.invoke("What temperature?")
    assert len(captured) == 1
    assert captured[0] == {"question": "What temperature?"}
    assert result == "42 degrees"


def test_ask_human_tool_handles_dict_response(monkeypatch):
    """ask_human tool should extract 'answer' from dict responses."""
    from chemgraph.tools import generic_tools

    def fake_interrupt(value):
        return {"answer": "298 K"}

    monkeypatch.setattr(generic_tools, "interrupt", fake_interrupt)

    result = generic_tools.ask_human.invoke("What temperature?")
    assert result == "298 K"


# ---------------------------------------------------------------------------
# Graph construction tests (ask_human tool included in defaults)
# ---------------------------------------------------------------------------


def test_single_agent_graph_includes_ask_human(monkeypatch):
    """construct_single_agent_graph should include ask_human in default tools."""
    from chemgraph.graphs.single_agent import construct_single_agent_graph
    from chemgraph.tools.generic_tools import ask_human

    # Use a dummy LLM
    class FakeLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return AIMessage(content="done")

    graph = construct_single_agent_graph(llm=FakeLLM())
    # The graph should compile without errors; verify it has nodes.
    node_names = list(graph.get_graph().nodes.keys())
    assert "ChemGraphAgent" in node_names
    assert "tools" in node_names


def test_single_agent_graph_excludes_ask_human_when_unsupervised():
    """construct_single_agent_graph should exclude ask_human when human_supervised=False."""
    from chemgraph.graphs.single_agent import construct_single_agent_graph
    from chemgraph.tools.generic_tools import ask_human

    captured_tools = []

    class FakeLLM:
        def bind_tools(self, tools):
            captured_tools.extend(tools)
            return self

        def invoke(self, messages):
            return AIMessage(content="done")

    graph = construct_single_agent_graph(llm=FakeLLM(), human_supervised=False)
    node_names = list(graph.get_graph().nodes.keys())
    assert "ChemGraphAgent" in node_names
    assert "tools" in node_names
    # ask_human must NOT be among the tools registered with the ToolNode.
    tool_names = [
        getattr(t, "name", None)
        for t in graph.get_graph().nodes["tools"].data.tools_by_name.values()
    ]
    assert "ask_human" not in tool_names


def test_get_single_agent_prompt_strips_ask_human():
    """get_single_agent_prompt(human_supervised=False) should remove the ask_human block."""
    from chemgraph.prompt.single_agent_prompt import (
        get_single_agent_prompt,
        single_agent_prompt,
    )

    prompt_with = get_single_agent_prompt(human_supervised=True)
    prompt_without = get_single_agent_prompt(human_supervised=False)

    assert "ask_human" in prompt_with
    assert "ask_human" not in prompt_without
    # The prompt without should be shorter
    assert len(prompt_without) < len(prompt_with)
    # The default prompt should match the supervised version
    assert prompt_with == single_agent_prompt


def test_multi_agent_graph_includes_human_review(monkeypatch):
    """construct_multi_agent_graph should include human_review node."""
    from chemgraph.graphs.multi_agent import construct_multi_agent_graph
    from chemgraph.tools.generic_tools import calculator

    # Use a dummy LLM
    class FakeLLM:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return AIMessage(content="done")

        async def ainvoke(self, messages):
            return AIMessage(content="done")

    graph = construct_multi_agent_graph(
        llm=FakeLLM(), executor_tools=[calculator]
    )
    node_names = list(graph.get_graph().nodes.keys())
    assert "Planner" in node_names
    assert "human_review" in node_names
