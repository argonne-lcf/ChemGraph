import pytest
from chemgraph.agent.llm_agent import ChemGraph


WORKFLOWS = [
    "single_agent",
    "multi_agent",
    "python_relp",
    "graspa",
    "mock_agent",
    "single_agent_mcp",
    "multi_agent_mcp",
    "graspa_mcp",
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
        "multi_agent": "construct_multi_agent_graph",
        "python_relp": "construct_relp_graph",
        "graspa": "construct_graspa_graph",
        "mock_agent": "construct_mock_agent_graph",
        "single_agent_mcp": "construct_single_agent_mcp_graph",
        "multi_agent_mcp": "construct_multi_agent_mcp_graph",
        "graspa_mcp": "construct_graspa_mcp_graph",
    }[workflow_type]

    monkeypatch.setattr(
        f"chemgraph.agent.llm_agent.{constructor_attr}",
        fake_constructor,
    )

    # Ensure model loading is deterministic and doesn't call external APIs
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_openai_model",
        lambda model_name, temperature, base_url=None: "FAKE_LLM",
    )

    # For MCP workflows some constructors expect tools; pass a non-empty list
    kwargs = {}
    if workflow_type in {"single_agent_mcp", "multi_agent_mcp", "graspa_mcp"}:
        kwargs["tools"] = ["DUMMY_TOOL"]
        kwargs["data_tools"] = ["DUMMY_TOOL"]

    cg = ChemGraph(model_name="gpt-4o-mini", workflow_type=workflow_type, **kwargs)
    assert cg.workflow == f"WORKFLOW-SENTINEL-{workflow_type}"
    args_tuple, kwargs_called = called["args"]
    if args_tuple:
        assert args_tuple[0] == "FAKE_LLM"
    else:
        assert kwargs_called.get("llm") == "FAKE_LLM"
