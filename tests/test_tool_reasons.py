"""Reviewed tools require a reason that is displayed but never executed."""

from uuid import uuid4

import pytest
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool, Tool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict
from pydantic.v1 import BaseModel as BaseModelV1

from chemgraph.agent.events import _AstreamEventCallback
from chemgraph.graphs.deep_agent import construct_deep_agent_graph
from chemgraph.graphs.tool_reasons import ToolReasonMiddleware
from chemgraph.graphs.workspace import prepare_workspace_runtime
from tests.test_main_agent import _ScriptedChatModel


class _RecordingModel(_ScriptedChatModel):
    bindings: list = []

    def bind_tools(self, tools, **kwargs):
        self.bindings = [*self.bindings, list(tools)]
        return super().bind_tools(tools, **kwargs)


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"{name}-{uuid4().hex}"}])


def parameters(entry):
    return convert_to_openai_tool(entry)["function"]["parameters"]


class _StrictArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str


RECEIVED = []


def _record(**kwargs):
    RECEIVED.append(kwargs)
    return "ok"


strict_tool = StructuredTool.from_function(_record, name="write_file", description="Strict.",
                                           args_schema=_StrictArgs)
dict_tool = StructuredTool.from_function(
    _record, name="run_ase", description="MCP-shaped.",
    args_schema={"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]},
)


@tool
def read_file(path: str) -> str:
    """Unreviewed read."""
    return path


@tool
def delete(path: str, reason: str) -> str:
    """A tool whose own schema already asks for a reason."""
    return reason


class _InferredTool(BaseTool):
    """A BaseTool subclass whose schema LangChain infers from ``_run``."""

    name: str = "edit_file"
    description: str = "Inferred schema."

    def _run(self, path: str) -> str:
        RECEIVED.append({"path": path})
        return "ok"


class _V1Args(BaseModelV1):
    path: str
    reason: str


v1_tool = StructuredTool.from_function(_record, name="execute", description="Pydantic v1.", args_schema=_V1Args)
single_input = Tool(name="custom_write", func=lambda text: RECEIVED.append(text), description="Single input.")


def test_reviewed_schemas_require_reason_and_execution_never_sees_it():
    inferred = _InferredTool()
    middleware = ToolReasonMiddleware({"write_file", "run_ase", "delete", "edit_file", "execute", "custom_write"})
    seen = {}

    def handler(request):
        seen["tools"] = request.tools
        return "response"

    class _Request:
        tools = [strict_tool, dict_tool, read_file, delete, inferred, v1_tool, single_input, {"type": "web_search"}]

        def override(self, **kwargs):
            request = _Request()
            request.tools = kwargs["tools"]
            return request

    assert middleware.wrap_model_call(_Request(), handler) == "response"
    bound = {getattr(entry, "name", "builtin"): entry for entry in seen["tools"]}
    for name in ("write_file", "run_ase"):
        assert "reason" in parameters(bound[name])["required"]
    assert "reason" not in parameters(bound["read_file"])["properties"]
    # Native reasons and schemas the middleware cannot extend safely pass through.
    for entry in (delete, inferred, v1_tool, single_input):
        assert bound[entry.name] is entry
    assert bound["builtin"] == {"type": "web_search"}

    RECEIVED.clear()
    for entry, args in ((strict_tool, {"path": "/a"}), (dict_tool, {"value": 1})):
        tool_call = {"name": entry.name, "args": {**args, "reason": "Because."}, "id": "1", "type": "tool_call"}
        request = ToolCallRequest(tool_call=tool_call, tool=entry, state={}, runtime=None)
        middleware.wrap_tool_call(request, lambda selected: selected.tool.invoke(selected.tool_call))
        assert tool_call["args"]["reason"] == "Because."  # The checkpointed call is not mutated.
    # A native reason parameter is preserved for execution.
    tool_call = {"name": "execute", "args": {"path": "/c", "reason": "Mine."}, "id": "2", "type": "tool_call"}
    middleware.wrap_tool_call(ToolCallRequest(tool_call=tool_call, tool=v1_tool, state={}, runtime=None),
                              lambda selected: selected.tool.invoke(selected.tool_call))
    assert RECEIVED == [{"path": "/a"}, {"value": 1}, {"path": "/c", "reason": "Mine."}]

    # A same-named replacement with another schema gets a fresh extension.
    replacement = StructuredTool.from_function(
        _record, name="run_ase", description="Replaced.",
        args_schema={"type": "object", "properties": {"steps": {"type": "integer"}}, "required": ["steps"]},
    )
    _Request.tools = [replacement]
    middleware.wrap_model_call(_Request(), handler)
    assert set(parameters(seen["tools"][0])["required"]) == {"steps", "reason"}


def test_reason_reaches_review_and_one_tool_event_after_approval():
    events = []
    graph = construct_deep_agent_graph(
        _ScriptedChatModel(responses=[
            call("write_file", file_path="/notes.txt", content="hello", reason="Save the note."),
            AIMessage(content="done"),
        ]),
        discover_skills=False,
    )
    config = {"configurable": {"thread_id": "reason"},
              "callbacks": [_AstreamEventCallback(lambda *event: events.append(event), "reason")]}
    graph.invoke({"messages": [HumanMessage(content="Write a note")]}, config)
    [pending] = graph.get_state(config).interrupts
    assert pending.value["action_requests"][0]["args"]["reason"] == "Save the note."

    state = graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    assert "/notes.txt" in state["files"]
    started = [payload for name, payload in events if name == "tool_call_started"]
    assert [payload.get("reason") for payload in started] == ["Save the note."]
    assert "reason" not in started[0]["arguments"]


@pytest.mark.parametrize("review", [False, True])
def test_general_purpose_worker_requires_and_reports_reasons(monkeypatch, review):
    import deepagents.middleware.subagents as subagents

    stacks = []
    real_create_agent = subagents.create_agent

    def recording_create_agent(model, **kwargs):
        if kwargs.get("name") == "general-purpose":
            stacks.append([type(item).__name__ for item in kwargs["middleware"]])
        return real_create_agent(model, **kwargs)

    monkeypatch.setattr(subagents, "create_agent", recording_create_agent)
    events = []
    model = _RecordingModel(responses=[
        call("task", subagent_type="general-purpose", description="Record the result"),
        call("write_file", file_path="/w.txt", content="x", reason="Record the result."),
        AIMessage(content="worker done"),
        AIMessage(content="all done"),
    ])
    graph = construct_deep_agent_graph(model, discover_skills=False,
                                       **({} if review else {"interrupt_on": None}))
    config = {"configurable": {"thread_id": "gp"},
              "callbacks": [_AstreamEventCallback(lambda *event: events.append(event), "gp")]}
    graph.invoke({"messages": [HumanMessage(content="Delegate")]}, config)
    if review:
        # The resumed run reuses the carried callback, so the reason is reported once.
        assert graph.get_state(config).interrupts
        graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)

    worker_tools = next(tools for tools in model.bindings if "task" not in {t.name for t in tools})
    assert "reason" in parameters(next(t for t in worker_tools if t.name == "write_file"))["required"]
    started = [payload for name, payload in events if name == "tool_call_started"
               and payload["tool_name"] == "write_file"]
    assert [(p["reason"], p["subagent_name"]) for p in started] == [("Record the result.", "general-purpose")]

    if review:
        return
    # The explicit worker keeps the implicit stack, including ChemGraph's skills slot.
    backend, sources, middleware, _ = prepare_workspace_runtime(discover_skills=False, interrupt_on=None)
    create_deep_agent(model=model, skills=sources, middleware=middleware, backend=backend)
    explicit, implicit = stacks[0], stacks[-1]
    assert "ChemGraphSkillsMiddleware" in implicit
    assert [name for name in explicit if name != "ToolReasonMiddleware"] == implicit
    assert "ToolReasonMiddleware" in explicit


@pytest.fixture(autouse=True)
def _reset():
    RECEIVED.clear()
