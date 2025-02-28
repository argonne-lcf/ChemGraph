import pytest
from comp_chem_agent.agent.llm_agent import CompChemAgent
from unittest.mock import Mock, patch


@pytest.fixture
def mock_llm():
    return Mock()


def test_comp_chem_agent_initialization():
    with patch("comp_chem_agent.agent.llm_agent.load_openai_model") as mock_load:
        mock_load.return_value = Mock()
        agent = CompChemAgent(model_name="gpt-3.5-turbo")
        assert agent.llm is not None
        assert hasattr(agent, "graph")


def test_agent_query(mock_llm):
    with patch("comp_chem_agent.agent.llm_agent.load_openai_model") as mock_load:
        mock_load.return_value = mock_llm
        agent = CompChemAgent(model_name="gpt-3.5-turbo")

        # Mock response
        mock_llm.invoke.return_value = "Test response"

        response = agent.runq("What is the SMILES string for water?")
        assert response == "Test response"
        mock_llm.invoke.assert_called_once()
