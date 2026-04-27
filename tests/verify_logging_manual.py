import os
import shutil
import asyncio
from unittest.mock import MagicMock, patch

# Mock dependencies to avoid needing API keys or real models
import sys

sys.modules["chemgraph.models.openai"] = MagicMock()
sys.modules["chemgraph.models.alcf_endpoints"] = MagicMock()
sys.modules["chemgraph.models.local_model"] = MagicMock()
sys.modules["chemgraph.models.anthropic"] = MagicMock()
sys.modules["chemgraph.models.gemini"] = MagicMock()
sys.modules["chemgraph.models.groq"] = MagicMock()

# Mock supported models to pass validation
with patch("chemgraph.models.supported_models.supported_openai_models", ["mock-model"]):
    from chemgraph.agent.llm_agent import ChemGraph
    from chemgraph.tools.ase_core import _resolve_path


def test_resolve_path():
    print("Testing _resolve_path...")
    # 1. No env var
    if "CHEMGRAPH_LOG_DIR" in os.environ:
        del os.environ["CHEMGRAPH_LOG_DIR"]
    assert _resolve_path("foo.txt") == "foo.txt"

    # 2. Env var set
    log_dir = os.path.abspath(".log_test_temp")
    os.environ["CHEMGRAPH_LOG_DIR"] = log_dir
    expected = os.path.join(log_dir, "foo.txt")
    assert _resolve_path("foo.txt") == expected
    assert os.environ["CHEMGRAPH_LOG_DIR"] == log_dir

    # Clean up
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    print("PASS: _resolve_path")


async def test_agent_logging():
    print("Testing Agent logging...")
    # Clear env var
    if "CHEMGRAPH_LOG_DIR" in os.environ:
        del os.environ["CHEMGRAPH_LOG_DIR"]

    # Mock workflow to avoid real execution
    cg = ChemGraph(model_name="mock-model", workflow_type="mock_agent")

    # Mock the workflow.astream to yield a message
    mock_workflow = MagicMock()

    async def mock_astream(*args, **kwargs):
        yield {"messages": [MagicMock(content="done")]}

    cg.workflow = MagicMock()
    cg.workflow.astream = mock_astream

    # Mock get_state so write_state works without recursion
    mock_state_snapshot = MagicMock()
    mock_state_snapshot.values = {"messages": ["mock_message"]}
    cg.workflow.get_state.return_value = mock_state_snapshot

    # Run
    await cg.run("test query")

    # Check if .log/session_* was created and CHEMGRAPH_LOG_DIR was set
    log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")
    assert log_dir is not None, "CHEMGRAPH_LOG_DIR was not set by run()"
    assert ".log/session_" in log_dir, f"Unexpected log dir: {log_dir}"
    assert os.path.exists(log_dir), "Log dir does not exist"
    assert os.path.exists(os.path.join(log_dir, "state.json")), "state.json not created"

    print(f"PASS: Agent created log dir: {log_dir}")

    # Cleanup
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        # remove parent .log if empty? No, keep it.


if __name__ == "__main__":
    test_resolve_path()
    asyncio.run(test_agent_logging())
