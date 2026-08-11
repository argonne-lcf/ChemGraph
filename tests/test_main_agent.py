"""Tests for the middleware-based main-agent supervisor."""

from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import interrupt
from pydantic import Field

from chemgraph.agent.main_session import MainAgentSession
from chemgraph.graphs.main_agent import (
    construct_main_agent_graph,
    latest_assistant_text,
)
from chemgraph.graphs.single_agent import construct_single_agent_graph


class _MessageState(TypedDict):
    messages: Annotated[list, add_messages]


class _ScriptedChatModel(BaseChatModel):
    responses: list[Any]
    response_index: int = 0
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-main-agent-test"

    def bind_tools(self, tools, **_kwargs):
        self.bound_tools = list(tools)
        return self

    def _generate(
        self,
        _messages,
        stop=None,
        run_manager=None,
        **_kwargs,
    ) -> ChatResult:
        response = self.responses[self.response_index]
        self.response_index += 1
        if isinstance(response, BaseException):
            raise response
        return ChatResult(generations=[ChatGeneration(message=response)])


def _task_call(call_id: str, description: str = "do the work") -> dict:
    return {
        "name": "task",
        "args": {"subagent_type": "worker", "description": description},
        "id": call_id,
        "type": "tool_call",
    }


def _answering_subgraph(answer: str, calls: list[str] | None = None):
    def answer_node(state):
        if calls is not None:
            calls.append(state["messages"][-1].content)
        return {"messages": [AIMessage(content=answer)]}

    builder = StateGraph(_MessageState)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=None)


def _interrupting_subgraph():
    def clarification_node(_state):
        answer = interrupt({"question": "Which calculator should I use?"})
        return {"messages": [AIMessage(content=f"worker used {answer}")]}

    builder = StateGraph(_MessageState)
    builder.add_node("clarification", clarification_node)
    builder.add_edge(START, "clarification")
    builder.add_edge("clarification", END)
    return builder.compile(checkpointer=None)


def _parallel_interrupting_subgraph():
    def first_node(_state):
        answer = interrupt({"question": "first"})
        return {"messages": [AIMessage(content=f"first={answer}")]}

    def second_node(_state):
        answer = interrupt({"question": "second"})
        return {"messages": [AIMessage(content=f"second={answer}")]}

    builder = StateGraph(_MessageState)
    builder.add_node("first", first_node)
    builder.add_node("second", second_node)
    builder.add_edge(START, "first")
    builder.add_edge(START, "second")
    builder.add_edge("first", END)
    builder.add_edge("second", END)
    return builder.compile(checkpointer=None)


def _terminal_tool_subgraph():
    def terminal_node(state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_report",
                            "args": {},
                            "id": "report-1",
                            "type": "tool_call",
                        },
                        {
                            "name": "save_data",
                            "args": {},
                            "id": "data-1",
                            "type": "tool_call",
                        },
                    ],
                ),
                ToolMessage(
                    content="Report written to report.html",
                    tool_call_id="report-1",
                    name="write_report",
                ),
                ToolMessage(
                    content="Data written to results.csv",
                    tool_call_id="data-1",
                    name="save_data",
                ),
            ]
        }

    builder = StateGraph(_MessageState)
    builder.add_node("terminal", terminal_node)
    builder.add_edge(START, "terminal")
    builder.add_edge("terminal", END)
    return builder.compile(checkpointer=None)


def _subagent(runnable, *, name: str = "worker", description: str = "Test worker"):
    return {"name": name, "description": description, "runnable": runnable}


@pytest.mark.asyncio
async def test_main_agent_delegates_and_keeps_normal_turns_on_one_thread():
    worker_calls = []
    llm = _ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_task_call("call-1")]),
            AIMessage(content="The worker found 42."),
            AIMessage(content="Here is the follow-up answer."),
        ]
    )
    graph = construct_main_agent_graph(
        llm,
        subagents=[_subagent(_answering_subgraph("42", worker_calls))],
    )
    session = MainAgentSession(graph, thread_id="main-agent-turns")

    first = await session.run("Calculate something")
    second = await session.run("Explain that result")

    assert first.status == "completed"
    assert first.assistant_response == "The worker found 42."
    assert first.interrupts == ()
    assert second.status == "completed"
    assert second.assistant_response == "Here is the follow-up answer."
    assert worker_calls == ["do the work"]
    messages = graph.get_state(session.config).values["messages"]
    assert [message.content for message in messages if isinstance(message, HumanMessage)] == [
        "Calculate something",
        "Explain that result",
    ]
    assert {tool.name for tool in llm.bound_tools} == {"task"}


@pytest.mark.asyncio
async def test_nested_subagent_interrupt_resumes_then_completes_turn():
    llm = _ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_task_call("call-1")]),
            AIMessage(content="The calculation used EMT."),
        ]
    )
    graph = construct_main_agent_graph(
        llm,
        subagents=[_subagent(_interrupting_subgraph())],
    )
    session = MainAgentSession(graph, thread_id="nested-interrupt")

    clarification = await session.run("Run a calculation")

    assert clarification.status == "waiting_for_user"
    assert len(clarification.interrupts) == 1
    assert clarification.interrupts[0].payload == {
        "question": "Which calculator should I use?"
    }
    with pytest.raises(RuntimeError, match="waiting for interrupt"):
        await session.run("Do not skip the pending answer")

    completed = await session.resume("EMT")

    assert completed.status == "completed"
    assert completed.assistant_response == "The calculation used EMT."
    assert completed.interrupts == ()


@pytest.mark.asyncio
async def test_parallel_subagent_interrupts_require_id_mapped_responses():
    llm = _ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_task_call("parallel-call")]),
            AIMessage(content="Both clarifications were applied."),
        ]
    )
    graph = construct_main_agent_graph(
        llm,
        subagents=[_subagent(_parallel_interrupting_subgraph())],
    )
    session = MainAgentSession(graph, thread_id="parallel-interrupts")
    clarification = await session.run("Run both tasks")

    assert len(clarification.interrupts) == 2
    assert {item.payload["question"] for item in clarification.interrupts} == {
        "first",
        "second",
    }
    assert all(item.id for item in clarification.interrupts)
    with pytest.raises(ValueError, match="require a mapping"):
        await session.resume("one answer")

    responses = {
        item.id: f"answer-{item.payload['question']}"
        for item in clarification.interrupts
    }
    completed = await session.resume(responses)

    assert completed.status == "completed"
    assert completed.assistant_response == "Both clarifications were applied."


@pytest.mark.asyncio
async def test_terminal_tool_outputs_are_returned_to_supervisor():
    llm = _ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[_task_call("report-call")]),
            AIMessage(content="Artifacts are ready."),
        ]
    )
    graph = construct_main_agent_graph(
        llm,
        subagents=[_subagent(_terminal_tool_subgraph())],
    )
    session = MainAgentSession(graph, thread_id="terminal-output")

    result = await session.run("Create report artifacts")

    assert result.assistant_response == "Artifacts are ready."
    task_results = [
        message.content
        for message in graph.get_state(session.config).values["messages"]
        if isinstance(message, ToolMessage) and message.name == "task"
    ]
    assert task_results == [
        "Report written to report.html\nData written to results.csv"
    ]


def test_latest_assistant_text_uses_standard_message_normalization():
    messages = [
        {"role": "assistant", "content": "older"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "new"},
                {"type": "reasoning", "reasoning": "hidden"},
                {"type": "text", "text": " answer"},
            ],
        },
    ]

    assert latest_assistant_text(messages) == "new answer"


@pytest.mark.asyncio
async def test_unknown_subagent_is_reported_to_main_model():
    call = _task_call("unknown-id")
    call["args"]["subagent_type"] = "missing"
    llm = _ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[call]),
            AIMessage(content="That specialist is unavailable."),
        ]
    )
    graph = construct_main_agent_graph(
        llm,
        subagents=[_subagent(_answering_subgraph("done"))],
    )
    session = MainAgentSession(graph, thread_id="unknown-worker")

    result = await session.run("Use a missing worker")

    assert result.assistant_response == "That specialist is unavailable."
    tool_messages = [
        message.text
        for message in graph.get_state(session.config).values["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert any("does not exist" in text for text in tool_messages)


def test_default_worker_forwards_options_and_inherits_parent_checkpoint(monkeypatch):
    captured = {}

    def fake_single_agent(*_args, **kwargs):
        captured.update(kwargs)
        return _answering_subgraph("done")

    monkeypatch.setattr(
        "chemgraph.graphs.main_agent.construct_single_agent_graph",
        fake_single_agent,
    )
    graph = construct_main_agent_graph(
        _ScriptedChatModel(responses=[]),
        subagent_system_prompt="worker prompt",
        subagent_formatter_prompt="formatter prompt",
        subagent_report_prompt="report prompt",
        subagent_structured_output=True,
        subagent_generate_report=True,
        subagent_max_retries=3,
        subagent_human_supervised=True,
        subagent_terminal_tool_names=("save_result",),
    )

    assert captured == {
        "tools": None,
        "structured_output": True,
        "generate_report": True,
        "max_retries": 3,
        "human_supervised": True,
        "terminal_tool_names": ("save_result",),
        "checkpointer": None,
        "system_prompt": "worker prompt",
        "formatter_prompt": "formatter prompt",
        "report_prompt": "report prompt",
    }
    assert isinstance(graph.checkpointer, InMemorySaver)


@pytest.mark.parametrize(
    ("specs", "error", "match"),
    [
        ([], ValueError, "At least one"),
        (
            [_subagent(_answering_subgraph("done"), name=" worker ")],
            ValueError,
            "surrounding whitespace",
        ),
        (
            [
                _subagent(_answering_subgraph("one")),
                _subagent(_answering_subgraph("two")),
            ],
            ValueError,
            "Duplicate",
        ),
        (
            [_subagent(_answering_subgraph("done"), description=" ")],
            ValueError,
            "description",
        ),
    ],
)
def test_subagent_validation(specs, error, match):
    with pytest.raises(error, match=match):
        construct_main_agent_graph(
            _ScriptedChatModel(responses=[]),
            subagents=specs,
        )


def test_main_tools_are_extensible_and_task_is_reserved():
    @tool
    def read_file(path: str) -> str:
        """Return test file contents."""
        return path

    llm = _ScriptedChatModel(responses=[AIMessage(content="done")])
    graph = construct_main_agent_graph(
        llm,
        subagents=[_subagent(_answering_subgraph("done"))],
        main_tools=[read_file],
    )
    assert graph is not None

    @tool("task")
    def duplicate_task(description: str) -> str:
        """Conflict with the middleware task tool."""
        return description

    with pytest.raises(ValueError, match="reserved"):
        construct_main_agent_graph(
            llm,
            subagents=[_subagent(_answering_subgraph("done"))],
            main_tools=[duplicate_task],
        )


def test_single_agent_preserves_default_and_allows_inherited_checkpointer():
    llm = _ScriptedChatModel(responses=[AIMessage(content="done")])

    standalone = construct_single_agent_graph(llm)
    embedded = construct_single_agent_graph(llm, checkpointer=None)

    assert isinstance(standalone.checkpointer, InMemorySaver)
    assert embedded.checkpointer is None


@pytest.mark.asyncio
async def test_session_validation():
    graph = construct_main_agent_graph(
        _ScriptedChatModel(responses=[AIMessage(content="ready")]),
        subagents=[_subagent(_answering_subgraph("done"))],
    )
    session = MainAgentSession(graph, thread_id="lifecycle")

    assert session.failed is False
    assert session.pending_interrupts == ()
    with pytest.raises(RuntimeError, match="not waiting"):
        await session.resume("too early")
    with pytest.raises(RuntimeError, match="no failed operation"):
        await session.retry()
    with pytest.raises(ValueError, match="non-empty"):
        await session.run(" ")

    result = await session.run("hello")

    assert result.status == "completed"
    assert session.failed is False


@pytest.mark.asyncio
async def test_failed_initial_turn_can_retry_without_duplicate_user_message():
    graph = construct_main_agent_graph(
        _ScriptedChatModel(
            responses=[RuntimeError("transient failure"), AIMessage(content="ready")]
        ),
        subagents=[_subagent(_answering_subgraph("done"))],
    )
    session = MainAgentSession(graph, thread_id="retry-initial")

    with pytest.raises(RuntimeError, match="transient failure"):
        await session.run("hello")
    assert session.failed is True
    with pytest.raises(RuntimeError, match="retry it"):
        await session.run("duplicate")

    result = await session.retry()

    assert result.assistant_response == "ready"
    assert session.failed is False
    human_messages = [
        message
        for message in graph.get_state(session.config).values["messages"]
        if isinstance(message, HumanMessage)
    ]
    assert [message.content for message in human_messages] == ["hello"]


@pytest.mark.asyncio
async def test_failed_follow_up_can_retry_without_duplicate_user_message():
    graph = construct_main_agent_graph(
        _ScriptedChatModel(
            responses=[
                AIMessage(content="first response"),
                RuntimeError("follow-up failure"),
                AIMessage(content="second response"),
            ]
        ),
        subagents=[_subagent(_answering_subgraph("done"))],
    )
    session = MainAgentSession(graph, thread_id="retry-follow-up")

    await session.run("first question")
    with pytest.raises(RuntimeError, match="follow-up failure"):
        await session.run("second question")
    result = await session.retry()

    assert result.assistant_response == "second response"
    human_messages = [
        message
        for message in graph.get_state(session.config).values["messages"]
        if isinstance(message, HumanMessage)
    ]
    assert [message.content for message in human_messages] == [
        "first question",
        "second question",
    ]
