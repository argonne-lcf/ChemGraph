from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from chemgraph.agent import llm_agent
from chemgraph.agent.llm_agent import ChemGraph


class _DummyTool:
    def __init__(self, name):
        self.name = name


class _FakeWorkflow:
    def __init__(self):
        self.astream_calls = []
        self.last_state = {"messages": [AIMessage(content="done")]}

    async def astream(self, inputs, *, stream_mode, config):
        self.astream_calls.append(
            {"inputs": inputs, "stream_mode": stream_mode, "config": config},
        )
        for callback in config.get("callbacks", []):
            callback.on_chat_model_start({"name": "FakeChatModel"}, [["hello"]])
            callback.on_llm_end(SimpleNamespace(generations=[]))
        yield self.last_state

    def get_state(self, config):
        return SimpleNamespace(values=self.last_state)


@pytest.mark.parametrize(
    ("workflow_type", "constructor_attr", "kwargs"),
    [
        ("single_agent", "construct_single_agent_graph", {}),
        ("multi_agent", "construct_multi_agent_graph", {}),
        ("python_relp", "construct_relp_graph", {}),
        ("graspa", "construct_graspa_graph", {}),
        ("mock_agent", "construct_mock_agent_graph", {}),
        (
            "single_agent_mcp",
            "construct_single_agent_mcp_graph",
            {"tools": [_DummyTool("mcp_tool")]},
        ),
        (
            "graspa_mcp",
            "construct_graspa_mcp_graph",
            {"tools": [_DummyTool("executor")], "data_tools": [_DummyTool("analysis")]},
        ),
        ("rag_agent", "construct_rag_agent_graph", {}),
        ("single_agent_xanes", "construct_single_agent_xanes_graph", {}),
    ],
)
def test_graph_constructor_is_called(
    monkeypatch,
    tmp_path,
    workflow_type,
    constructor_attr,
    kwargs,
):
    called = {}
    workflow = _FakeWorkflow()

    def fake_constructor(*args, **constructor_kwargs):
        called["args"] = args
        called["kwargs"] = constructor_kwargs
        return workflow

    monkeypatch.setattr(f"chemgraph.agent.llm_agent.{constructor_attr}", fake_constructor)
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_openai_model",
        lambda **_kwargs: "FAKE_LLM",
    )

    cg = ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type=workflow_type,
        enable_memory=False,
        log_dir=str(tmp_path / "logs"),
        **kwargs,
    )

    assert cg.workflow is workflow
    args = called.get("args", ())
    constructor_kwargs = called.get("kwargs", {})
    assert (args and args[0] == "FAKE_LLM") or constructor_kwargs.get("llm") == "FAKE_LLM"


@pytest.mark.asyncio
async def test_graph_backed_run_uses_astream_and_emits_events(monkeypatch, tmp_path):
    workflow = _FakeWorkflow()
    events = []

    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.construct_single_agent_graph",
        lambda *_args, **_kwargs: workflow,
    )
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_openai_model",
        lambda **_kwargs: "FAKE_LLM",
    )

    cg = ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        enable_memory=False,
        log_dir=str(tmp_path / "logs"),
        return_option="last_message",
        on_event=lambda event, payload: events.append((event, payload)),
    )
    response = await cg.run("hello", config={"thread_id": "test-thread"})

    assert response.content == "done"
    assert workflow.astream_calls[0]["inputs"] == {"messages": "hello"}
    assert workflow.astream_calls[0]["stream_mode"] == "values"
    assert workflow.astream_calls[0]["config"]["configurable"]["thread_id"] == "test-thread"
    assert [event for event, _payload in events] == [
        "workflow_started",
        "llm_call_started",
        "llm_call_finished",
        "workflow_finished",
    ]


def test_single_agent_initialization_injects_calculator_availability(monkeypatch, tmp_path):
    called = {}

    def fake_constructor(*args, **kwargs):
        called["args"] = (args, kwargs)
        return _FakeWorkflow()

    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.construct_single_agent_graph",
        fake_constructor,
    )
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_openai_model",
        lambda **_kwargs: "FAKE_LLM",
    )

    cg = ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        enable_memory=False,
        log_dir=str(tmp_path / "logs"),
    )

    args_tuple, _ = called["args"]
    system_prompt = args_tuple[1]
    assert "Calculator availability detected during ChemGraph initialization" in system_prompt
    assert cg.default_calculator in system_prompt
    assert cg.default_calculator in cg.available_calculators


def test_rag_and_xanes_default_prompts_are_preserved(monkeypatch, tmp_path):
    captured = {}

    def fake_constructor(*args, **kwargs):
        captured[kwargs.get("system_prompt")] = True
        return _FakeWorkflow()

    monkeypatch.setattr("chemgraph.agent.llm_agent.construct_rag_agent_graph", fake_constructor)
    monkeypatch.setattr("chemgraph.agent.llm_agent.construct_single_agent_xanes_graph", fake_constructor)
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_openai_model",
        lambda **_kwargs: "FAKE_LLM",
    )

    ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type="rag_agent",
        enable_memory=False,
        log_dir=str(tmp_path / "rag-logs"),
    )
    ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type="single_agent_xanes",
        enable_memory=False,
        log_dir=str(tmp_path / "xanes-logs"),
    )

    assert llm_agent.rag_agent_prompt in captured
    assert llm_agent.default_xanes_single_agent_prompt in captured
