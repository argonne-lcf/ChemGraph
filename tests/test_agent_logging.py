import os
import shutil
import pytest
from unittest.mock import patch, Mock
from chemgraph.agent.llm_agent import ChemGraph


@pytest.fixture
def clean_env():
    # Cache and clear relevant env vars
    old_log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")

    if "CHEMGRAPH_LOG_DIR" in os.environ:
        del os.environ["CHEMGRAPH_LOG_DIR"]

    yield

    # Restore
    if old_log_dir:
        os.environ["CHEMGRAPH_LOG_DIR"] = old_log_dir
    elif "CHEMGRAPH_LOG_DIR" in os.environ:
        del os.environ["CHEMGRAPH_LOG_DIR"]


def test_init_generates_log_dir(clean_env):
    with (
        patch("chemgraph.agent.llm_agent.load_openai_model") as mock_load,
        patch("chemgraph.agent.llm_agent.construct_single_agent_graph") as mock_graph,
    ):
        mock_load.return_value = Mock()
        mock_graph.return_value = Mock()

        agent = ChemGraph()

        assert agent.log_dir is not None
        assert agent.uuid is not None
        assert os.path.join("cg_logs", "session_") in agent.log_dir
        assert os.environ.get("CHEMGRAPH_LOG_DIR") == agent.log_dir

        # Cleanup created dir
        if os.path.exists(agent.log_dir):
            shutil.rmtree(agent.log_dir, ignore_errors=True)


def test_init_respects_env_var(clean_env):
    with (
        patch("chemgraph.agent.llm_agent.load_openai_model") as mock_load,
        patch("chemgraph.agent.llm_agent.construct_single_agent_graph") as mock_graph,
    ):
        mock_load.return_value = Mock()
        mock_graph.return_value = Mock()

        test_dir = "/tmp/test_chemgraph_logs_custom"
        os.environ["CHEMGRAPH_LOG_DIR"] = test_dir

        agent = ChemGraph()
        assert agent.log_dir == test_dir
        # uuid should always be set now, even when CHEMGRAPH_LOG_DIR is pre-set
        assert agent.uuid is not None
        assert len(agent.uuid) == 8
