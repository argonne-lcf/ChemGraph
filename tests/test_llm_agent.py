import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage
from chemgraph.agent.events import SUBAGENT_METADATA_KEY
from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.agent.turn import _TurnEventCallback
from chemgraph.models.endpoints import PreparedModel


def _prepared(llm=None, *, supports_structured_output=True):
    """Return a ``(client, PreparedModel)`` tuple matching the loader seam.

    ``ChemGraph.__init__`` now calls ``load_chat_model_prepared``; tests mock
    that seam and must return the same 2-tuple shape.
    """
    return (
        llm if llm is not None else Mock(),
        PreparedModel(
            endpoint_name="test",
            protocol="openai_compatible",
            client_kwargs={},
            supports_structured_output=supports_structured_output,
        ),
    )


@pytest.fixture
def mock_llm():
    return Mock()


def test_chemgraph_initialization(tmp_path):
    with patch("chemgraph.agent.llm_agent.load_chat_model_prepared") as mock_load:
        mock_load.return_value = _prepared()
        agent = ChemGraph(
            model_name="gpt-4o-mini",
            enable_memory=False,
            log_dir=str(tmp_path / "logs"),
        )
        assert hasattr(agent, "workflow")


@pytest.mark.parametrize(
    ("model_name", "reasoning_effort", "expected_effort"),
    [
        ("argo:gpt-5.6-luna", None, "none"),
        ("argo:gpt-5.6-sol", "high", "high"),
        ("argo:gpt-5.6-terra", None, "none"),
    ],
)
def test_gpt56_reasoning_effort_is_passed_to_loader(
    tmp_path, model_name, reasoning_effort, expected_effort
):
    with patch("chemgraph.agent.llm_agent.load_chat_model_prepared") as mock_load:
        mock_load.return_value = _prepared()
        agent = ChemGraph(
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            enable_memory=False,
            log_dir=str(tmp_path / "logs"),
        )

    assert mock_load.call_args.kwargs["reasoning_effort"] == expected_effort
    assert agent.reasoning_effort == expected_effort


def test_reasoning_effort_is_not_passed_to_sonnet5(tmp_path):
    with patch("chemgraph.agent.llm_agent.load_chat_model_prepared") as mock_load:
        mock_load.return_value = _prepared()
        agent = ChemGraph(
            model_name="argo:claude-sonnet-5",
            enable_memory=False,
            log_dir=str(tmp_path / "logs"),
        )

    assert mock_load.call_args.kwargs["reasoning_effort"] is None
    assert agent.reasoning_effort is None


def test_reasoning_effort_rejects_unverified_model():
    with pytest.raises(ValueError, match="does not have verified"):
        ChemGraph(model_name="argo:gpt-5.4", reasoning_effort="none")


@pytest.mark.parametrize("reasoning_effort", ["fast", ""])
def test_reasoning_effort_rejects_invalid_value(reasoning_effort):
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        ChemGraph(
            model_name="argo:gpt-5.6-sol", reasoning_effort=reasoning_effort
        )


def test_agent_query(mock_llm, tmp_path):
    with patch(
        "chemgraph.agent.llm_agent.load_chat_model_prepared"
    ) as mock_init_load, patch(
        "chemgraph.agent.turn.load_chat_model"
    ) as mock_turn_load:
        # Set up the mock chain
        mock_chain = Mock()
        mock_chain.invoke.return_value = AIMessage(content="Test response")
        mock_llm.bind_tools.return_value = mock_chain
        mock_init_load.return_value = _prepared(mock_llm)
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


def test_visualize_returns_ascii_graph():
    graph = Mock()
    graph.draw_ascii.return_value = "ASCII GRAPH"
    workflow = Mock()
    workflow.get_graph.return_value = graph
    agent = object.__new__(ChemGraph)
    agent.workflow = workflow

    assert agent.visualize() == "ASCII GRAPH"
    graph.draw_ascii.assert_called_once_with()


def test_state_access_delegates_capability_and_uses_fresh_defaults():
    class AsyncLookingWorkflow:
        checkpointer = type("AsyncCustomSaver", (), {})()

        def __init__(self):
            self.sync_configs = []
            self.async_configs = []

        def get_state(self, config):
            self.sync_configs.append(config)
            return SimpleNamespace(values={"mode": "sync"})

        async def aget_state(self, config):
            self.async_configs.append(config)
            return SimpleNamespace(values={"mode": "async"})

    workflow = AsyncLookingWorkflow()
    agent = object.__new__(ChemGraph)
    agent.workflow = workflow

    assert agent.get_state() == {"mode": "sync"}
    first_config = workflow.sync_configs[0]
    first_config["mutated"] = True
    assert agent.get_state() == {"mode": "sync"}
    assert "mutated" not in workflow.sync_configs[1]
    assert asyncio.run(agent.aget_state()) == {"mode": "async"}


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
                "token_counts": {
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": 12,
                    "source": "provider",
                    "raw_usage": {"total_tokens": 12},
                },
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


def test_turn_event_callback_emits_anthropic_cache_usage():
    events = []
    callback = _TurnEventCallback(
        lambda event, payload: events.append((event, payload)),
        "thread-1",
    )
    usage = {
        "input_tokens": 1500,
        "output_tokens": 120,
        "total_tokens": 1620,
        "input_token_details": {
            "cache_creation": 0,
            "cache_read": 400,
            "ephemeral_5m_input_tokens": 1000,
        },
    }

    callback.on_llm_end(
        SimpleNamespace(
            generations=[
                [
                    SimpleNamespace(
                        message=AIMessage(content="done", usage_metadata=usage),
                    ),
                ],
            ],
        ),
    )

    assert events == [
        (
            "llm_call_finished",
            {
                "thread_id": "thread-1",
                "token_counts": {
                    "input_tokens": 1500,
                    "output_tokens": 120,
                    "total_tokens": 1620,
                    "cache_creation_input_tokens": 1000,
                    "cache_read_input_tokens": 400,
                    "source": "provider",
                    "raw_usage": usage,
                },
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

    assert events == [("llm_call_finished", {"thread_id": "thread-1"})]


def test_turn_event_callback_ignores_malformed_usage_metadata():
    class BrokenUsage:
        def model_dump(self, **kwargs):
            if kwargs:
                raise TypeError("mode is unsupported")
            raise RuntimeError("broken usage metadata")

    events = []
    callback = _TurnEventCallback(
        lambda event, payload: events.append((event, payload)),
        "thread-1",
    )

    callback.on_llm_end(
        SimpleNamespace(
            generations=[
                [SimpleNamespace(message=SimpleNamespace(usage_metadata=BrokenUsage()))],
            ],
        ),
    )

    assert events == [("llm_call_finished", {"thread_id": "thread-1"})]


def test_tool_event_includes_optional_subagent_name():
    events = []
    callback = _TurnEventCallback(
        lambda event, payload: events.append((event, payload)),
        "thread-1",
    )

    callback.on_tool_start(
        {"name": "run_ase"},
        "{'calculator': 'EMT'}",
        metadata={SUBAGENT_METADATA_KEY: "chemgraph"},
    )
    callback.on_tool_start(
        {"name": "read_file"},
        "{'file_path': '/result.txt'}",
    )

    assert events[0] == (
        "tool_call_started",
        {
            "thread_id": "thread-1",
            "tool_name": "run_ase",
            "arguments": "{'calculator': 'EMT'}",
            "subagent_name": "chemgraph",
        },
    )
    assert "subagent_name" not in events[1][1]


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


@pytest.mark.asyncio
async def test_cli_trace_events_are_emitted_from_astream_path(monkeypatch, tmp_path):
    from chemgraph.cli.trace import CLIRunTrace

    class FakeWorkflow:
        def __init__(self):
            self.state = {"messages": [AIMessage(content="done")]}

        async def astream(self, inputs, *, stream_mode, config):
            for callback in config.get("callbacks", []):
                callback.on_chat_model_start({"name": "FakeChatModel"}, [["hello"]])
                callback.on_llm_end(SimpleNamespace(generations=[]))
            yield self.state

        def get_state(self, config):
            return SimpleNamespace(values=self.state)

    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.construct_single_agent_graph",
        lambda *_args, **_kwargs: FakeWorkflow(),
    )
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_chat_model_prepared",
        lambda **_kwargs: _prepared(),
    )

    trace = CLIRunTrace(
        tmp_path / "trace",
        run_id="trace-test",
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        query="x",
    )
    trace.start()
    agent = ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        enable_memory=False,
        log_dir=str(tmp_path / "logs"),
        on_event=trace.on_event,
    )
    await agent.run("x")
    trace.finish(status="completed")

    events = [
        json.loads(line)["event"]
        for line in (tmp_path / "trace" / "events.jsonl").read_text().splitlines()
    ]
    assert events == [
        "run_started",
        "workflow_started",
        "llm_call_started",
        "llm_call_finished",
        "workflow_finished",
        "run_finished",
    ]
