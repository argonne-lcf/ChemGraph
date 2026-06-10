import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage
from chemgraph.agent.llm_agent import ChemGraph, _TurnEventCallback


@pytest.fixture
def mock_llm():
    return Mock()


def test_chemgraph_initialization(tmp_path):
    with patch("chemgraph.agent.llm_agent.load_openai_model") as mock_load:
        mock_load.return_value = Mock()
        agent = ChemGraph(
            model_name="gpt-4o-mini",
            enable_memory=False,
            log_dir=str(tmp_path / "logs"),
        )
        assert hasattr(agent, "workflow")

def test_agent_query(mock_llm, tmp_path):
    with patch("chemgraph.agent.llm_agent.load_openai_model") as mock_init_load, patch(
        "chemgraph.models.loader.load_openai_model"
    ) as mock_turn_load:
        # Set up the mock chain
        mock_chain = Mock()
        mock_chain.invoke.return_value = AIMessage(content="Test response")
        mock_llm.bind_tools.return_value = mock_chain
        mock_init_load.return_value = mock_llm
        mock_turn_load.return_value = mock_llm

        agent = ChemGraph(
            model_name="gpt-4o-mini",
            enable_memory=False,
            log_dir=str(tmp_path / "logs"),
        )
        response = asyncio.run(agent.run("What is the SMILES string for water?"))
        assert isinstance(response, AIMessage)
        assert response.content == "Test response"
        mock_llm.bind_tools.assert_called_once()
        mock_chain.invoke.assert_called_once()


def test_turn_event_callback_emits_llm_decision_for_tool_calls():
    events = []
    callback = _TurnEventCallback(
        lambda event, payload: events.append((event, payload)),
        "thread-1",
    )
    response = SimpleNamespace(
        llm_output={"token_usage": {"total_tokens": 12}},
        generations=[
            [
                SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=[
                            {"name": "molecule_name_to_smiles", "id": "call-1"},
                            {
                                "function": {"name": "smiles_to_coordinate_file"},
                                "tool_call_id": "call-2",
                            },
                        ],
                    ),
                ),
            ],
        ],
    )

    callback.on_llm_end(response)

    assert events == [
        (
            "llm_call_finished",
            {
                "thread_id": "thread-1",
                "llm_output": {"token_usage": {"total_tokens": 12}},
            },
        ),
        (
            "llm_decision",
            {
                "thread_id": "thread-1",
                "tool_calls": [
                    {"name": "molecule_name_to_smiles", "id": "call-1"},
                    {"name": "smiles_to_coordinate_file", "id": "call-2"},
                ],
            },
        ),
    ]


def test_turn_event_callback_skips_llm_decision_without_tool_calls():
    events = []
    callback = _TurnEventCallback(
        lambda event, payload: events.append((event, payload)),
        "thread-1",
    )

    callback.on_llm_end(
        SimpleNamespace(generations=[[SimpleNamespace(message=AIMessage(content="done"))]]),
    )

    assert [event for event, _payload in events] == ["llm_call_finished"]


def test_turn_event_callback_ignores_llm_decision_extraction_errors():
    class BrokenGenerationGroup:
        def __iter__(self):
            raise RuntimeError("broken response")

    events = []
    callback = _TurnEventCallback(
        lambda event, payload: events.append((event, payload)),
        "thread-1",
    )

    callback.on_llm_end(SimpleNamespace(generations=[BrokenGenerationGroup()]))

    assert [event for event, _payload in events] == ["llm_call_finished"]
