import pytest
from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.models.endpoints import PreparedModel


def _fake_prepared(**_kwargs):
    """Mock ``load_chat_model_prepared``: return ``(client, PreparedModel)``."""
    return (
        "FAKE_LLM",
        PreparedModel(
            endpoint_name="test",
            protocol="openai_compatible",
            client_kwargs={},
        ),
    )


WORKFLOWS = [
    "single_agent",
    "main_agent",
    "multi_agent",
    "python_relp",
    "graspa",
    "mock_agent",
    "graspa_mcp",
    "single_agent_xanes",
    "molecular_docking",
]


@pytest.mark.parametrize("workflow_type", WORKFLOWS)
def test_constructor_is_called(monkeypatch, workflow_type):
    called = {}

    def fake_constructor(*args, **kwargs):
        called["args"] = (args, kwargs)
        return f"WORKFLOW-SENTINEL-{workflow_type}"

    # Patch the constructor name used by chemgraph.agent.llm_agent
    constructor_attr = {
        "single_agent": "construct_single_agent_graph",
        "main_agent": "construct_main_agent_graph",
        "multi_agent": "construct_multi_agent_graph",
        "python_relp": "construct_relp_graph",
        "graspa": "construct_graspa_graph",
        "mock_agent": "construct_mock_agent_graph",
        "graspa_mcp": "construct_graspa_mcp_graph",
        "single_agent_xanes": "construct_single_agent_xanes_graph",
        "molecular_docking": "construct_molecular_docking_graph",
    }[workflow_type]

    monkeypatch.setattr(
        f"chemgraph.agent.llm_agent.{constructor_attr}",
        fake_constructor,
    )

    # Ensure model loading is deterministic and doesn't call external APIs
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_chat_model_prepared",
        _fake_prepared,
    )

    # For MCP workflows some constructors expect tools; pass a non-empty list
    kwargs = {}
    if workflow_type == "graspa_mcp":
        kwargs["tools"] = ["DUMMY_TOOL"]
        kwargs["data_tools"] = ["DUMMY_TOOL"]

    cg = ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type=workflow_type,
        enable_memory=False,
        **kwargs,
    )
    assert cg.workflow == f"WORKFLOW-SENTINEL-{workflow_type}"
    args_tuple, kwargs_called = called["args"]
    if args_tuple:
        assert args_tuple[0] == "FAKE_LLM"
    else:
        assert kwargs_called.get("llm") == "FAKE_LLM"


def test_single_agent_initialization_injects_calculator_availability(monkeypatch):
    called = {}

    def fake_constructor(*args, **kwargs):
        called["args"] = (args, kwargs)
        return "WORKFLOW-SENTINEL-single_agent"

    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.construct_single_agent_graph",
        fake_constructor,
    )
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_chat_model_prepared",
        _fake_prepared,
    )

    cg = ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        enable_memory=False,
    )

    args_tuple, _ = called["args"]
    system_prompt = args_tuple[1]
    assert "Calculator availability detected during ChemGraph initialization" in system_prompt
    assert cg.default_calculator in system_prompt
    assert cg.default_calculator in cg.available_calculators
