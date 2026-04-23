"""Tests for the planner_agent function in the Send()-based multi-agent graph."""

import json
import pytest

from chemgraph.graphs.multi_agent import planner_agent


# -- Helpers / mocks ----------------------------------------------------------

_VALID_PLANNER_JSON = json.dumps(
    {
        "thought_process": "Delegating parsed tasks to executors.",
        "next_step": "executor_subgraph",
        "tasks": [
            {"task_index": 1, "prompt": "Calculate water enthalpy using xtb calculator."}
        ],
    }
)

_FINISH_JSON = json.dumps(
    {
        "thought_process": "All tasks complete. Reaction enthalpy = -42.5 eV.",
        "next_step": "FINISH",
    }
)


class _DummyResponse:
    """Mimics the object returned by ``llm.invoke(messages)``."""

    def __init__(self, content: str):
        self.content = content


class _DummyLLM:
    """Returns a fixed raw-text response (prompt-injection style)."""

    def __init__(self, content: str = _VALID_PLANNER_JSON):
        self._content = content

    def invoke(self, _messages):
        return _DummyResponse(self._content)


class _FailThenSucceedLLM:
    """Returns invalid text on the first call, valid JSON on the second."""

    def __init__(self):
        self._calls = 0

    def invoke(self, _messages):
        self._calls += 1
        if self._calls == 1:
            return _DummyResponse("This is not valid JSON at all.")
        return _DummyResponse(_VALID_PLANNER_JSON)


# -- Tests --------------------------------------------------------------------


def test_planner_agent_returns_tasks_and_next_step():
    """planner_agent should return messages, next_step, and tasks."""
    llm = _DummyLLM()
    state = {"messages": [{"role": "user", "content": "test"}]}
    out = planner_agent(state=state, llm=llm, system_prompt="planner")

    assert "messages" in out
    assert out["next_step"] == "executor_subgraph"
    assert len(out["tasks"]) == 1
    assert out["tasks"][0].task_index == 1


def test_planner_agent_finish():
    """planner_agent should handle FINISH with no tasks."""
    llm = _DummyLLM(_FINISH_JSON)
    state = {"messages": [{"role": "user", "content": "test"}]}
    out = planner_agent(state=state, llm=llm, system_prompt="planner")

    assert out["next_step"] == "FINISH"
    assert out["tasks"] == []


def test_planner_agent_uses_executor_results():
    """When executor_results are present, they should be included in context."""
    llm = _DummyLLM()
    state = {
        "messages": [{"role": "user", "content": "test"}],
        "executor_results": ["[worker_1] Result: energy = -76.4 eV"],
    }
    out = planner_agent(state=state, llm=llm, system_prompt="planner")
    assert out["next_step"] == "executor_subgraph"


def test_planner_agent_retries_on_parse_failure():
    """planner_agent should retry when the first response is unparseable."""
    llm = _FailThenSucceedLLM()
    state = {"messages": [{"role": "user", "content": "test"}]}
    out = planner_agent(state=state, llm=llm, system_prompt="planner", max_retries=1)

    assert out["next_step"] == "executor_subgraph"
    assert len(out["tasks"]) == 1


def test_planner_agent_raises_after_retries_exhausted():
    """planner_agent should raise ValueError when all retries fail."""
    llm = _DummyLLM("not json")
    state = {"messages": [{"role": "user", "content": "test"}]}
    with pytest.raises(ValueError, match="Planner failed to produce valid JSON"):
        planner_agent(state=state, llm=llm, system_prompt="planner", max_retries=1)


def test_planner_agent_parses_markdown_fenced_json():
    """planner_agent should extract JSON from markdown fences."""
    fenced = f"Sure, here is the plan:\n```json\n{_VALID_PLANNER_JSON}\n```\n"
    llm = _DummyLLM(fenced)
    state = {"messages": [{"role": "user", "content": "test"}]}
    out = planner_agent(state=state, llm=llm, system_prompt="planner")

    assert out["next_step"] == "executor_subgraph"
    assert len(out["tasks"]) == 1
