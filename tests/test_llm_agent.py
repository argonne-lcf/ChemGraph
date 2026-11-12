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
        ("argo:gpt-5.6-luna", None, "medium"),
        ("argo:gpt-5.6-sol", None, "medium"),
        ("argo:gpt-5.6-terra", None, "medium"),
        ("argo:claude-opus-4.8", None, "medium"),
        ("argo:claude-opus-5", None, "medium"),
        ("argo:gpt-5.6-sol", "high", "high"),
        ("argo:claude-opus-5", "xhigh", "xhigh"),
    ],
)
def test_reasoning_effort_is_passed_to_loader(
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


def test_claude_reasoning_effort_rejects_none():
    with pytest.raises(ValueError, match="supports: low, medium, high, xhigh, max"):
        ChemGraph(model_name="argo:claude-opus-4.8", reasoning_effort="none")


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
                    "input_tokens": None, "output_tokens": None, "total_tokens": 12,
                    "cached_input_tokens": None, "reasoning_output_tokens": None,
                    "raw_usage": {"total_tokens": 12}, "source": "provider",
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
                callback.on_chat_model_start({"name": "FakeChatModel"}, [["hello"]], run_id="test-call")
                callback.on_llm_end(SimpleNamespace(generations=[]), run_id="test-call")
            yield self.state

        def get_state(self, config):
            return SimpleNamespace(values=self.state)

        async def aget_state(self, config):
            return self.get_state(config)

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


def test_cli_run_query_prints_compact_activity_once_per_thread(monkeypatch, tmp_path):
    """Skill reads are summarized, parallel results labelled, history not replayed."""
    from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

    from chemgraph.cli import commands
    from chemgraph.cli.formatting import console

    skill = "\n".join(f"{n:>6}  skill line {n}" for n in range(1, 501))
    batch = [{"name": "run_ase", "args": {"calculator": c}, "id": f"c{c}"} for c in "ABC"]
    turn1 = [
        HumanMessage("q1"),
        AIMessage("", id="a1", tool_calls=[
            {"name": "read_file", "args": {"file_path": "/skills/x/SKILL.md"}, "id": "r"},
        ]),
        ToolMessage(skill, tool_call_id="r", name="read_file", id="t1"),
        AIMessage("", id="a2", tool_calls=batch),
        *[ToolMessage(f"energy {c}", tool_call_id=f"c{c}", name="run_ase", id=f"t{c}") for c in "BAC"],
        AIMessage("final answer one", id="a3"),
    ]
    states = {"messages": turn1}

    class FakeWorkflow:
        async def astream(self, inputs, *, stream_mode, config):
            for end in range(1, len(states["messages"]) + 1):
                # Copies, as checkpoint deserialization produces.
                yield {"messages": [m.model_copy() for m in states["messages"][:end]]}

        async def aget_state(self, config):
            return SimpleNamespace(values=states)

    monkeypatch.setattr("chemgraph.agent.llm_agent.construct_single_agent_graph", lambda *_a, **_k: FakeWorkflow())
    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", lambda **_k: _prepared())
    printed = []
    monkeypatch.setattr(BaseMessage, "pretty_print", lambda message: printed.append(message.id))
    agent = ChemGraph(workflow_type="single_agent", enable_memory=False, log_dir=str(tmp_path / "logs"))

    with console.capture() as capture:
        commands.run_query(agent, "q1", thread_id=7)
    first = capture.get()
    assert "⏺ read_file(/skills/x/SKILL.md)" in first and "⎿ Read 500 lines" in first
    assert "skill line" not in first and "final answer one" not in first  # Answer is printed once later.
    assert [line for line in first.splitlines() if "⎿ run_ase" in line] == [
        "  ⎿ run_ase #2/3 · energy B", "  ⎿ run_ase #1/3 · energy A", "  ⎿ run_ase #3/3 · energy C",
    ]

    states["messages"] = [*turn1, HumanMessage("q2"), AIMessage("answer two", id="a4")]
    with console.capture() as capture:
        commands.run_query(agent, "q2", thread_id=7)
    assert capture.get().strip() == "" and agent.message_renderer is None and printed == []

    # Structured workflows return the answer as a graph-generated human message.
    states["messages"] = [
        HumanMessage("q3", id="h3"),
        AIMessage("", id="a5", tool_calls=batch[:2]),
        ToolMessage("result A", tool_call_id="cA", name="run_ase", id="x1"),
        ToolMessage("result B", tool_call_id="cB", name="run_ase", id="x2"),
        AIMessage("Water.", id="a6"),
        HumanMessage('{"smiles": "O"}', id="h4"),
    ]
    with console.capture() as capture:
        commands.run_query(agent, "q3", thread_id=8)
    assert '{"smiles": "O"}' in capture.get() and "q3" not in capture.get()
    assert "Water." not in capture.get()  # Left to format_response, so shown once.

    commands.run_query(agent, "q3", thread_id=9, verbose=True)
    assert printed == ["h3", "a5", "x1", "x2", "a6", "h4"]  # Every parallel result, in full.
