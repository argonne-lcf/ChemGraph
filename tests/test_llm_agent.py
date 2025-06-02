import pytest
from chemgraph.agent.llm_agent import ChemGraph
from unittest.mock import Mock, patch
from langchain_core.messages import HumanMessage, AIMessage


@pytest.fixture
def mock_llm():
    return Mock()


def test_chemgraph_initialization():
    with patch("chemgraph.agent.llm_agent.load_openai_model") as mock_load:
        mock_load.return_value = Mock()
        agent = ChemGraph(model_name="gpt-4o-mini")
        assert hasattr(agent, "workflow")


def test_agent_query(mock_llm):
    with patch("chemgraph.agent.llm_agent.load_openai_model") as mock_load:
        # Set up the mock chain
        mock_chain = Mock()
        mock_chain.invoke.return_value = AIMessage(content="Test response")
        mock_llm.bind_tools.return_value = mock_chain
        mock_load.return_value = mock_llm

        agent = ChemGraph(model_name="gpt-4o-mini")
        response = agent.run("What is the SMILES string for water?")
        assert response == "Test response"
        mock_llm.bind_tools.assert_called_once()
        mock_chain.invoke.assert_called_once()
