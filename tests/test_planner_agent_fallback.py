import json

from chemgraph.graphs.multi_agent import PlannerAgent as planner_agent
from chemgraph.graphs.multi_agent_mcp import PlannerAgent as planner_agent_mcp


class _StructuredLLMFailure:
    def invoke(self, messages):
        raise ValueError("1 validation error for PlannerResponse")


class _DummyResponse:
    def __init__(self, content):
        self.content = content


class _DummyLLM:
    def __init__(self, raw_content):
        self.raw_content = raw_content

    def with_structured_output(self, _schema):
        return _StructuredLLMFailure()

    def invoke(self, _messages):
        return _DummyResponse(self.raw_content)


def _assert_fallback_works(planner_fn):
    llm = _DummyLLM(
        '[{"task_index": 1, "prompt": "Calculate water enthalpy using xtb calculator."}]'
    )
    state = {"messages": [{"role": "user", "content": "test"}]}
    out = planner_fn(
        state=state,
        llm=llm,
        system_prompt="planner",
        support_structured_output=True,
    )
    payload = json.loads(out["messages"][0])
    assert "worker_tasks" in payload
    assert payload["worker_tasks"][0]["task_index"] == 1


def test_planner_agent_falls_back_when_structured_parse_fails():
    _assert_fallback_works(planner_agent)


def test_planner_agent_mcp_falls_back_when_structured_parse_fails():
    _assert_fallback_works(planner_agent_mcp)
