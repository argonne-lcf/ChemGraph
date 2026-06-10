import pytest
from langchain_core.messages import AIMessage

from chemgraph.agent import llm_agent
from chemgraph.agent.llm_agent import ChemGraph, TurnResult


class _DummyTool:
    def __init__(self, name):
        self.name = name


def _tool_names(tools):
    return [getattr(tool, "name", str(tool)) for tool in tools or []]


@pytest.mark.parametrize(
    ("workflow_type", "constructor_attr", "kwargs"),
    [
        ("multi_agent", "construct_multi_agent_graph", {}),
        (
            "graspa_mcp",
            "construct_graspa_mcp_graph",
            {"tools": [_DummyTool("executor")], "data_tools": [_DummyTool("analysis")]},
        ),
    ],
)
def test_legacy_graph_constructor_is_called(
    monkeypatch,
    tmp_path,
    workflow_type,
    constructor_attr,
    kwargs,
):
    called = {}

    def fake_constructor(*args, **constructor_kwargs):
        called["args"] = args
        called["kwargs"] = constructor_kwargs
        return f"WORKFLOW-SENTINEL-{workflow_type}"

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

    assert cg.workflow == f"WORKFLOW-SENTINEL-{workflow_type}"
    args = called.get("args", ())
    constructor_kwargs = called.get("kwargs", {})
    assert (args and args[0] == "FAKE_LLM") or constructor_kwargs.get("llm") == "FAKE_LLM"


@pytest.mark.parametrize(
    ("workflow_type", "kwargs", "expected_extra_tools", "expected_prompt"),
    [
        ("single_agent", {"tools": [_DummyTool("custom")]}, [], None),
        ("python_relp", {"tools": [_DummyTool("custom")]}, ["python_repl", "calculator"], None),
        ("graspa", {"tools": [_DummyTool("custom")]}, ["run_graspa"], None),
        (
            "mock_agent",
            {"tools": [_DummyTool("custom")]},
            [
                "file_to_atomsdata",
                "smiles_to_atomsdata",
                "run_ase",
                "molecule_name_to_smiles",
                "save_atomsdata_to_file",
                "calculator",
            ],
            None,
        ),
        (
            "single_agent_mcp",
            {"tools": [_DummyTool("mcp_tool")], "data_tools": [_DummyTool("data_tool")]},
            ["data_tool"],
            None,
        ),
        (
            "rag_agent",
            {"tools": [_DummyTool("custom")]},
            [
                "load_document",
                "query_knowledge_base",
                "file_to_atomsdata",
                "smiles_to_coordinate_file",
                "run_ase",
                "molecule_name_to_smiles",
                "save_atomsdata_to_file",
                "calculator",
            ],
            llm_agent.rag_agent_prompt,
        ),
        (
            "single_agent_xanes",
            {"tools": [_DummyTool("custom")]},
            [
                "molecule_name_to_smiles",
                "smiles_to_coordinate_file",
                "run_ase",
                "run_xanes",
                "fetch_xanes_data",
                "plot_xanes_data",
            ],
            llm_agent.default_xanes_single_agent_prompt,
        ),
    ],
)
@pytest.mark.asyncio
async def test_run_turn_workflow_tool_and_prompt_wiring(
    monkeypatch,
    tmp_path,
    workflow_type,
    kwargs,
    expected_extra_tools,
    expected_prompt,
):
    captured = {}

    async def fake_run_turn(**run_kwargs):
        captured.update(run_kwargs)
        return TurnResult(
            final_text="done",
            state={"messages": [AIMessage(content="done")]},
            executed_tool_names=(),
            terminal_tool=None,
            thread_id=run_kwargs["thread_id"],
            duration_s=0.0,
        )

    monkeypatch.setattr("chemgraph.agent.llm_agent.run_turn", fake_run_turn)
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
    response = await cg.run("hello", config={"thread_id": "test-thread"})

    assert response.content == "done"
    tool_names = _tool_names(captured["tools"])
    assert tool_names[0] == list(kwargs["tools"])[0].name
    for name in expected_extra_tools:
        assert name in tool_names
    if expected_prompt is not None:
        assert captured["system_prompt"] == expected_prompt


def test_single_agent_initialization_injects_calculator_availability(monkeypatch, tmp_path):
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

    assert "Calculator availability detected during ChemGraph initialization" in cg.system_prompt
    assert cg.default_calculator in cg.system_prompt
    assert cg.default_calculator in cg.available_calculators
