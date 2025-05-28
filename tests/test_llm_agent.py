import pytest
from chemgraph.agent.llm_agent import ChemGraph
from unittest.mock import Mock, patch


@pytest.fixture
def mock_llm():
    return Mock()


def test_chemgraph_initialization():
    with patch("chemgraph.agent.llm_agent.load_openai_model") as mock_load:
        mock_load.return_value = Mock()
        agent = ChemGraph(model_name="gpt-4o-mini")
        assert hasattr(agent, "graph")


def test_agent_query(mock_llm):
    with patch("chemgraph.agent.llm_agent.load_openai_model") as mock_load:
        mock_load.return_value = mock_llm
        agent = ChemGraph(model_name="gpt-4o-mini")

        # Mock response
        mock_llm.invoke.return_value = "Test response"

        response = agent.run("What is the SMILES string for water?")
        assert response == "Test response"
        mock_llm.invoke.assert_called_once()
