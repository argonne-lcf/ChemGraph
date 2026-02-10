import pytest
from chemgraph.agent.llm_agent import ChemGraph

WORKFLOWS = [
    "single_agent", "multi_agent", "python_relp", "graspa",
    "mock_agent", "single_agent_mcp", "multi_agent_mcp", "graspa_mcp",
]

@pytest.mark.parametrize("workflow_type", WORKFLOWS)
def test_constructor_is_called(monkeypatch, workflow_type):
    called_data = {}

    def fake_constructor(*args, **kwargs):
        called_data["args"] = args
        called_data["kwargs"] = kwargs
        return f"WORKFLOW-SENTINEL-{workflow_type}"

    mapping = {
        "single_agent": "construct_single_agent_graph",
        "multi_agent": "construct_multi_agent_graph", 
        "python_relp": "construct_relp_graph",
        "graspa": "construct_graspa_graph",
        "mock_agent": "construct_mock_agent_graph",
        "single_agent_mcp": "construct_single_agent_mcp_graph",
        "multi_agent_mcp": "construct_multi_agent_mcp_graph",
        "graspa_mcp": "construct_graspa_mcp_graph",
    }
    
    constructor_attr = mapping[workflow_type]

    # Patch the graph constructor
    monkeypatch.setattr(f"chemgraph.agent.llm_agent.{constructor_attr}", fake_constructor)
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_openai_model",
        lambda **kwargs: "FAKE_LLM",
    )

    # Set up inputs
    test_tools = ["DUMMY_TOOL"]
    kwargs = {"tools": test_tools, "data_tools": test_tools} if "_mcp" in workflow_type else {}

    # Initialize
    cg = ChemGraph(model_name="gpt-4o-mini", workflow_type=workflow_type, **kwargs)

    # Assertions
    assert cg.workflow == f"WORKFLOW-SENTINEL-{workflow_type}"
    
    # Check if LLM was passed as the first positional arg or a keyword arg
    args = called_data.get("args", [])
    kwargs_called = called_data.get("kwargs", {})
    
    llm_passed = (len(args) > 0 and args[0] == "FAKE_LLM") or (kwargs_called.get("llm") == "FAKE_LLM")
    assert llm_passed, f"LLM not passed to {workflow_type} constructor"

    # Specific check for MCP tool passing
    if workflow_type == "graspa_mcp":
        assert kwargs_called.get("executor_tools") == test_tools