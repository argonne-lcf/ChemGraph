import asyncio
import json
from types import SimpleNamespace

import pytest
from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from chemgraph.agent import llm_agent
from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.agent.main_session import MainAgentSession
from chemgraph.cli import commands
from chemgraph.cli.commands import check_api_keys
from chemgraph.cli.formatting import console
from chemgraph.graphs.deep_agent import construct_deep_agent_graph
from chemgraph.graphs.main_agent import construct_main_agent_graph
from chemgraph.graphs.single_agent import construct_single_agent_graph
from chemgraph.models import codex as codex_model
from chemgraph.models.codex import (
    CodexAuthenticationError,
    CodexChatModel,
    CodexResponseError,
    _strip_codex_prefix,
)
from chemgraph.models import loader
from chemgraph.models.protocols import codex_native


class _FakeCodexConfig:
    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.created.append(self)


class _FakeSandbox:
    read_only = "read-only"


class _FakeApprovalMode:
    deny_all = "deny-all"


class _FakeAccountResponse:
    def __init__(self, account):
        self._account = account

    def model_dump(self, **_kwargs):
        return {"account": self._account, "requiresOpenaiAuth": True}


class _FakeUsageBreakdown:
    def model_dump(self, **_kwargs):
        return {
            "inputTokens": 11,
            "outputTokens": 7,
            "totalTokens": 18,
        }


class _FakeThread:
    def __init__(self, state):
        self.state = state

    def run(self, prompt, **kwargs):
        self.state.run_calls.append((prompt, kwargs))
        response = self.state.responses.pop(0)
        return SimpleNamespace(
            final_response=response,
            usage=SimpleNamespace(last=_FakeUsageBreakdown()),
        )

    def turn(self, prompt, **kwargs):
        result = self.run(prompt, **kwargs)

        def stream():
            counts = result.usage.last.model_dump()
            yield SimpleNamespace(method="thread/tokenUsage/updated", payload={
                "turnId": "turn-1", "tokenUsage": {"total": counts, "last": counts},
            })
            yield SimpleNamespace(method="item/completed", payload={
                "turnId": "turn-1", "item": {
                    "type": "agentMessage", "text": result.final_response, "phase": "final_answer",
                },
            })
            yield SimpleNamespace(method="turn/completed", payload={
                "turn": {"id": "turn-1", "status": "completed"},
            })

        return SimpleNamespace(id="turn-1", stream=stream)


@pytest.fixture
def fake_codex_sdk(monkeypatch):
    state = SimpleNamespace(
        account={"type": "chatgpt", "email": "chemist@example.com"},
        responses=[],
        clients=[],
        thread_start_calls=[],
        run_calls=[],
    )
    _FakeCodexConfig.created.clear()

    class FakeCodex:
        def __init__(self, config=None):
            self.config = config
            state.clients.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def account(self):
            return _FakeAccountResponse(state.account)

        def thread_start(self, **kwargs):
            state.thread_start_calls.append(kwargs)
            return _FakeThread(state)

    monkeypatch.setattr(
        codex_model,
        "_load_codex_sdk",
        lambda: (
            FakeCodex,
            _FakeCodexConfig,
            _FakeSandbox,
            _FakeApprovalMode,
        ),
    )
    return state


def test_codex_prefix_validation():
    assert _strip_codex_prefix("codex:gpt-5.6-terra") == "gpt-5.6-terra"
    with pytest.raises(ValueError, match="cannot be empty"):
        _strip_codex_prefix("codex:")
    with pytest.raises(ValueError, match="must use"):
        _strip_codex_prefix("gpt-5.6-terra")


@pytest.mark.parametrize(
    ("account", "message"),
    [
        (None, "No Codex login"),
        ({"type": "apiKey"}, "uses an API key"),
        ({"type": "amazonBedrock"}, "not ChatGPT"),
    ],
)
def test_codex_rejects_non_chatgpt_authentication(
    fake_codex_sdk,
    account,
    message,
):
    fake_codex_sdk.account = account
    model = CodexChatModel(model_id="gpt-5.6-terra")

    with pytest.raises(CodexAuthenticationError, match=message):
        model.validate_authentication()


def test_codex_sdk_text_invocation_is_ephemeral_and_read_only(fake_codex_sdk):
    fake_codex_sdk.responses.append(
        json.dumps({"content": "Aspirin is acetylsalicylic acid.", "tool_calls": []})
    )
    model = CodexChatModel(model_id="gpt-5.6-terra")

    response = model.invoke([HumanMessage(content="What is aspirin?")])

    assert response.content == "Aspirin is acetylsalicylic acid."
    assert response.usage_metadata == {
        "input_tokens": 11,
        "output_tokens": 7,
        "total_tokens": 18,
    }
    config = _FakeCodexConfig.created[-1].kwargs
    assert config["env"] == {"OPENAI_API_KEY": "", "CODEX_API_KEY": ""}
    thread_call = fake_codex_sdk.thread_start_calls[-1]
    assert thread_call["model"] == "gpt-5.6-terra"
    assert thread_call["ephemeral"] is True
    assert thread_call["sandbox"] == _FakeSandbox.read_only
    assert thread_call["approval_mode"] == _FakeApprovalMode.deny_all
    assert thread_call["cwd"] == config["cwd"]
    run_prompt, run_kwargs = fake_codex_sdk.run_calls[-1]
    assert "What is aspirin?" in run_prompt
    assert run_kwargs["sandbox"] == _FakeSandbox.read_only
    assert run_kwargs["output_schema"]["additionalProperties"] is False


@tool
def lookup_smiles(name: str) -> str:
    """Return a deterministic test SMILES string for a molecule name."""
    assert name == "aspirin"
    return "CC(=O)OC1=CC=CC=C1C(=O)O"


def test_codex_tool_bridge_creates_langchain_tool_call(fake_codex_sdk):
    fake_codex_sdk.responses.append(
        json.dumps(
            {
                "content": "",
                "tool_calls": [
                    {
                        "name": "lookup_smiles",
                        "arguments": json.dumps({"name": "aspirin"}),
                    }
                ],
            }
        )
    )
    model = CodexChatModel(model_id="gpt-5.6-terra").bind_tools(
        [lookup_smiles],
        tool_choice="lookup_smiles",
        parallel_tool_calls=False,
    )

    response = model.invoke([HumanMessage(content="Look up aspirin")])

    assert response.tool_calls[0]["name"] == "lookup_smiles"
    assert response.tool_calls[0]["args"] == {"name": "aspirin"}
    assert response.tool_calls[0]["id"].startswith("call_")
    schema = fake_codex_sdk.run_calls[-1][1]["output_schema"]
    assert schema["properties"]["tool_calls"]["minItems"] == 1
    assert schema["properties"]["tool_calls"]["maxItems"] == 1


def test_codex_tool_bridge_rejects_unknown_tool(fake_codex_sdk):
    fake_codex_sdk.responses.append(
        json.dumps(
            {
                "content": "",
                "tool_calls": [
                    {"name": "not_bound", "arguments": json.dumps({})}
                ],
            }
        )
    )
    model = CodexChatModel(model_id="gpt-5.6-terra").bind_tools([lookup_smiles])

    with pytest.raises(CodexResponseError, match="unknown tool"):
        model.invoke([HumanMessage(content="Look up aspirin")])


def test_codex_adapter_runs_existing_single_agent_tool_loop(fake_codex_sdk):
    fake_codex_sdk.responses.extend(
        [
            json.dumps(
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "lookup_smiles",
                            "arguments": json.dumps({"name": "aspirin"}),
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "content": "The aspirin SMILES is CC(=O)OC1=CC=CC=C1C(=O)O.",
                    "tool_calls": [],
                }
            ),
        ]
    )
    graph = construct_single_agent_graph(
        CodexChatModel(model_id="gpt-5.6-terra"),
        system_prompt="Use the lookup tool before answering.",
        tools=[lookup_smiles],
    )

    state = graph.invoke(
        {"messages": "What is the SMILES for aspirin?"},
        config={"configurable": {"thread_id": "codex-test"}},
    )

    assert state["messages"][-2].name == "lookup_smiles"
    assert state["messages"][-1].content.startswith("The aspirin SMILES")
    assert len(fake_codex_sdk.run_calls) == 2


def _codex_payload(sdk, index=-1):
    prompt = sdk.run_calls[index][0]
    return json.loads(prompt.split("decision for this conversation:\n", 1)[1])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_codex_usage_is_retained_on_invalid_decision(fake_codex_sdk, asynchronous):
    from chemgraph.agent.usage import UsageCollector
    collector = UsageCollector("session", "thread")
    fake_codex_sdk.responses.append("{invalid")
    model = CodexChatModel(model_id="codex-test")
    with pytest.raises(CodexResponseError):
        if asynchronous:
            asyncio.run(model.ainvoke("test", config={"callbacks": [collector]}))
        else:
            model.invoke("test", config={"callbacks": [collector]})
    assert collector.summary["total_tokens"] == 18
    assert collector.summary["call_count"] == 1
    assert collector.summary["partial"] is False


def test_codex_stream_cumulative_usage_includes_internal_calls_and_failures(fake_codex_sdk, monkeypatch):
    from chemgraph.agent.usage import UsageCollector

    def turn(self, prompt, **kwargs):
        def stream():
            for total in (10, 30):
                yield SimpleNamespace(method="thread/tokenUsage/updated", payload={
                    "turnId": "turn", "tokenUsage": {
                        "total": {"inputTokens": total, "outputTokens": 4,
                                  "totalTokens": total + 4, "cachedInputTokens": 5,
                                  "reasoningOutputTokens": 2},
                        "last": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                    },
                })
            yield SimpleNamespace(method="turn/completed", payload={
                "turn": {"id": "turn", "status": "failed", "error": {"message": "SDK failed"}},
            })
        return SimpleNamespace(id="turn", stream=stream)

    monkeypatch.setattr(_FakeThread, "turn", turn)
    collector = UsageCollector("session", "thread")
    with pytest.raises(RuntimeError, match="SDK failed"):
        CodexChatModel(model_id="test").invoke("test", config={"callbacks": [collector]})
    counts = collector.summary
    assert counts["total_tokens"] == 34
    assert counts["cached_input_tokens"] == 5
    assert counts["reasoning_output_tokens"] == 2
    assert counts["partial"] is True
    assert counts["call_count"] == 1


def test_codex_correction_attempt_snapshots_replace_counts(fake_codex_sdk, monkeypatch):
    from chemgraph.agent.usage import UsageCollector
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from uuid import uuid4
    collector = UsageCollector("session", "thread")
    manager = CallbackManagerForLLMRun(run_id=uuid4(), handlers=[collector], inheritable_handlers=[])
    # Same ephemeral thread, two correction attempts with cumulative SDK totals.
    for total in (18, 36):
        codex_model._emit_usage(manager, {"total": {
            "inputTokens": total - 4, "outputTokens": 4, "totalTokens": total,
        }}, "test", complete=True)
    collector.on_llm_error(CodexResponseError("exhausted"), run_id=manager.run_id)
    assert collector.summary["call_count"] == 1
    assert collector.summary["total_tokens"] == 36


def test_codex_metadata_preserves_details_and_does_not_use_last():
    result = SimpleNamespace(usage={"total": {
        "inputTokens": 100, "outputTokens": 20, "totalTokens": 120,
        "cachedInputTokens": 80, "reasoningOutputTokens": 15,
    }, "last": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}})
    usage = codex_model._usage_metadata(result)
    assert usage["total_tokens"] == 120
    assert usage["input_token_details"] == {"cache_read": 80}
    assert usage["output_token_details"] == {"reasoning": 15}


def test_codex_preserves_parallel_tool_history_and_errors(fake_codex_sdk):
    calls = [
        {"name": "read_file", "args": {"file_path": path}, "id": call_id}
        for path, call_id in (("/first.txt", "first"), ("/second.txt", "second"))
    ]
    messages = [
        SystemMessage(content="Inspect the files."),
        HumanMessage(content="Compare them.", name="chemist"),
        AIMessage(content="", name="deepagent", tool_calls=calls),
        ToolMessage(
            content="Permission denied", name="read_file",
            tool_call_id="second", status="error",
        ),
        ToolMessage(
            content=[{"type": "text", "text": "First file contents"}],
            name="read_file", tool_call_id="first",
        ),
    ]
    fake_codex_sdk.responses.append(json.dumps({"content": "Done", "tool_calls": []}))
    CodexChatModel(model_id="test-model").invoke(messages)
    history = _codex_payload(fake_codex_sdk)["conversation"]
    assert history[0] == {"role": "system", "content": "Inspect the files."}
    assert history[1]["name"] == "chemist"
    assert history[2]["name"] == "deepagent"
    assert history[2]["tool_calls"] == messages[2].tool_calls
    assert history[3] == {
        "role": "tool", "content": "Permission denied", "name": "read_file",
        "tool_call_id": "second", "status": "error",
    }
    assert history[4]["tool_call_id"] == "first"
    assert history[4]["status"] == "success"
    assert json.loads(history[4]["content"]) == messages[4].content


@pytest.mark.parametrize("asynchronous", [False, True])
def test_codex_deep_agent_reads_skill_and_receives_result(
    fake_codex_sdk, tmp_path, asynchronous,
):
    skill_dir = tmp_path / "skills/test-analysis"
    skill_dir.mkdir(parents=True)
    body = "Required environment: analysis-env. Helper: scripts/analyze.py."
    (skill_dir / "SKILL.md").write_text(
        "---\nname: test-analysis\ndescription: Analyze test results.\n---\n" + body
    )
    path = "/workspace/skills/test-analysis/SKILL.md"
    fake_codex_sdk.responses.extend([
        json.dumps({"content": "", "tool_calls": [{
            "name": "read_file",
            "arguments": json.dumps({"file_path": path, "limit": 1000}),
        }]}),
        json.dumps({"content": body, "tool_calls": []}),
    ])
    graph = construct_deep_agent_graph(
        CodexChatModel(model_id="test-model"),
        backend=LocalShellBackend(root_dir=tmp_path, env={}),
        discover_skills=False,
        skills=["/workspace/skills/"],
    )
    data = {"messages": [HumanMessage(content="Read the test-analysis skill.")]}
    config = {"configurable": {"thread_id": "codex-skill"}}
    result = (
        asyncio.run(graph.ainvoke(data, config))
        if asynchronous else graph.invoke(data, config)
    )
    first = _codex_payload(fake_codex_sdk, 0)
    names = {tool["function"]["name"] for tool in first["available_tools"]}
    assert {"read_file", "write_file", "execute", "task"} <= names
    assert any(path in m["content"] for m in first["conversation"] if m["role"] == "system")
    instructions = fake_codex_sdk.thread_start_calls[0]["base_instructions"]
    assert "You may request any applicable tool listed" in instructions
    assert "Do not inspect files" not in instructions
    assert "Do not invoke Codex-native tools" in instructions
    history = _codex_payload(fake_codex_sdk)["conversation"]
    call = history[-2]["tool_calls"][0]
    assert call["name"] == "read_file" and call["args"]["file_path"] == path
    assert history[-1]["tool_call_id"] == call["id"]
    assert history[-1]["name"] == "read_file"
    assert history[-1]["status"] == "success"
    assert body in history[-1]["content"]
    assert result["messages"][-1].content == body


@pytest.mark.parametrize("write_decision,read_decision", [
    ("approve", "approve"), ("approve", "reject"), ("reject", None),
])
def test_codex_default_catalog_prepares_carbonic_acid(
    fake_codex_sdk, tmp_path, monkeypatch, write_decision, read_decision,
):
    from math import isfinite
    from chemgraph.registry import ToolRegistry
    from chemgraph.tools.ase_tools import file_to_atomsdata

    structure = tmp_path / "carbonic_acid.xyz"
    readbacks = []
    read_structure = file_to_atomsdata.func

    def record_readback(fname):
        atoms = read_structure(fname)
        readbacks.append(atoms)
        return atoms

    monkeypatch.setattr(file_to_atomsdata, "func", record_readback)

    def response(name, **arguments):
        return json.dumps({"content": "", "tool_calls": [{
            "name": name, "arguments": json.dumps(arguments),
        }]})

    fake_codex_sdk.responses.extend([
        response("search_tools", query="coordinate"),
        response("load_tools", names=["smiles_to_coordinate_file", "file_to_atomsdata"]),
        response("smiles_to_coordinate_file", smiles="O=C(O)O", output_file=str(structure), randomSeed=2025),
    ])
    if write_decision == "approve":
        fake_codex_sdk.responses.append(response("file_to_atomsdata", fname=str(structure)))
    fake_codex_sdk.responses.append(json.dumps({"content": "Done", "tool_calls": []}))
    agent = ChemGraph(
        model_name="codex:gpt-5.6-sol", workflow_type="deep_agent",
        deepagent_discover_skills=False,
        deepagent_backend=LocalShellBackend(root_dir=tmp_path, env={}),
        enable_memory=False, log_dir=str(tmp_path / "logs"),
    )
    assert agent.deepagent_tool_registry.names() == tuple(
        spec.name for spec in ToolRegistry().specs() if not spec.interactive
    )
    assert agent.deepagent_tool_registry._tools == {}
    config = {"configurable": {"thread_id": "carbonic-acid"}}
    state = agent.workflow.invoke(
        {"messages": [HumanMessage(content="Generate and validate carbonic acid locally.")]}, config,
    )
    assert state["__interrupt__"] and not structure.exists()
    assert state["__interrupt__"][0].value["action_requests"][0]["name"] == "smiles_to_coordinate_file"
    first = {t["function"]["name"] for t in _codex_payload(fake_codex_sdk, 0)["available_tools"]}
    assert {"search_tools", "load_tools"} <= first
    assert not set(ToolRegistry().names()) & first
    loaded = {t["function"]["name"] for t in _codex_payload(fake_codex_sdk, 2)["available_tools"]}
    assert {"smiles_to_coordinate_file", "file_to_atomsdata"} <= loaded
    assert "run_ase" not in loaded
    state = agent.workflow.invoke(Command(resume={"decisions": [{"type": write_decision}]}), config)
    if write_decision == "approve":
        assert state["__interrupt__"][0].value["action_requests"][0]["name"] == "file_to_atomsdata"
        assert structure.exists() and not readbacks
        state = agent.workflow.invoke(Command(resume={"decisions": [{"type": read_decision}]}), config)
    assert "__interrupt__" not in state
    assert structure.exists() == (write_decision == "approve")
    outputs = [m for m in state["messages"] if m.type == "tool"]
    assert "execute" not in {m.name for m in outputs}
    assert len(readbacks) == (1 if read_decision == "approve" else 0)
    if write_decision == "approve":
        artifact = json.loads(next(m.content for m in outputs if m.name == "smiles_to_coordinate_file"))
        assert artifact["ok"] and artifact["natoms"] == 6 and artifact["path"] == str(structure)
        expected_status = "success" if read_decision == "approve" else "error"
        assert next(m for m in outputs if m.name == "file_to_atomsdata").status == expected_status
    if read_decision == "approve":
        assert sorted(readbacks[0].numbers) == [1, 1, 6, 8, 8, 8]
        assert all(isfinite(value) for position in readbacks[0].positions for value in position)


@pytest.mark.parametrize("tool_name", ["write_file", "execute"])
@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_codex_deep_agent_actions_require_approval(
    fake_codex_sdk, tmp_path, tool_name, decision,
):
    effects = []

    class RecordingBackend(LocalShellBackend):
        def write(self, file_path, content):
            effects.append((file_path, content))
            return super().write(file_path, content)

        def execute(self, command, *, timeout=None):
            effects.append(command)
            return ExecuteResponse(output="Recorded execution", exit_code=0)

    arguments = (
        {"file_path": "/workspace/result.txt", "content": "Written once"}
        if tool_name == "write_file" else {"command": "echo recorded"}
    )
    fake_codex_sdk.responses.extend([
        json.dumps({"content": "", "tool_calls": [{
            "name": tool_name, "arguments": json.dumps(arguments),
        }]}),
        json.dumps({"content": "Done", "tool_calls": []}),
    ])
    graph = construct_deep_agent_graph(
        CodexChatModel(model_id="test-model"),
        backend=RecordingBackend(root_dir=tmp_path, env={}),
        discover_skills=False,
    )
    config = {"configurable": {"thread_id": "codex-approval"}}
    state = graph.invoke({"messages": [HumanMessage(content="Perform the task.")]}, config)
    assert state["__interrupt__"]
    assert effects == []
    assert not (tmp_path / "result.txt").exists()
    state = graph.invoke(Command(resume={"decisions": [{"type": decision}]}), config)
    assert "__interrupt__" not in state
    assert len(effects) == (1 if decision == "approve" else 0)
    if tool_name == "write_file" and decision == "approve":
        assert (tmp_path / "result.txt").read_text() == "Written once"
    else:
        assert not (tmp_path / "result.txt").exists()
    history = _codex_payload(fake_codex_sdk)["conversation"]
    assert history[-1]["tool_call_id"] == history[-2]["tool_calls"][0]["id"]
    assert history[-1]["name"] == tool_name
    for thread in fake_codex_sdk.thread_start_calls:
        assert thread["sandbox"] == _FakeSandbox.read_only
        assert thread["approval_mode"] == _FakeApprovalMode.deny_all


@pytest.mark.parametrize("choice", ["none", "required"])
@pytest.mark.parametrize("request_tool", [False, True])
def test_codex_tool_choice_constraints_remain_enforced(
    fake_codex_sdk, choice, request_tool,
):
    calls = [{"name": "lookup_smiles", "arguments": json.dumps({"name": "aspirin"})}]
    fake_codex_sdk.responses.append(json.dumps({
        "content": "", "tool_calls": calls if request_tool else [],
    }))
    model = CodexChatModel(model_id="test-model").bind_tools([lookup_smiles], tool_choice=choice)
    if request_tool == (choice == "required"):
        response = model.invoke([HumanMessage(content="Look up aspirin.")])
        assert bool(response.tool_calls) == request_tool
    else:
        with pytest.raises(CodexResponseError, match="required tool call|tools were disabled"):
            model.invoke([HumanMessage(content="Look up aspirin.")])
    prompt, kwargs = fake_codex_sdk.run_calls[-1]
    schema = kwargs["output_schema"]["properties"]["tool_calls"]
    if choice == "none":
        assert schema["maxItems"] == 0
        assert "Do not request a tool" in prompt
    else:
        assert schema["minItems"] == 1
        assert "JSON-encoded object string" in prompt


def test_shared_loader_routes_codex_prefix(monkeypatch):
    monkeypatch.setattr(
        codex_native,
        "load_codex_model",
        lambda model_name: ("codex-model", model_name),
    )

    assert loader.load_chat_model("codex:test-model") == (
        "codex-model",
        "codex:test-model",
    )


@pytest.mark.parametrize(
    ("workflow_type", "constructor_name"),
    [
        ("single_agent", "construct_single_agent_graph"),
        ("main_agent", "construct_main_agent_graph"),
        ("deep_agent", "construct_deep_agent_graph"),
    ],
)
def test_chemgraph_routes_codex_to_supported_workflow(
    monkeypatch,
    tmp_path,
    workflow_type,
    constructor_name,
):
    captured = {}
    monkeypatch.setattr(
        codex_native,
        "load_codex_model",
        lambda model_name: ("codex-model", model_name),
    )
    monkeypatch.setattr(
        llm_agent,
        constructor_name,
        lambda llm, *_args, **_kwargs: captured.setdefault("llm", llm),
    )

    ChemGraph(
        model_name="codex:test-model",
        workflow_type=workflow_type,
        enable_memory=False,
        log_dir=str(tmp_path),
    )

    assert captured["llm"] == ("codex-model", "codex:test-model")


def test_chemgraph_rejects_codex_for_unsupported_workflows():
    with pytest.raises(ValueError, match="single_agent, main_agent, and deep_agent"):
        ChemGraph(model_name="codex:test-model", workflow_type="multi_agent")


@pytest.mark.asyncio
async def test_codex_adapter_runs_main_agent_delegation(fake_codex_sdk):
    fake_codex_sdk.responses.extend(
        [
            json.dumps(
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "task",
                            "arguments": json.dumps(
                                {
                                    "subagent_type": "chemgraph",
                                    "description": "Look up the aspirin SMILES.",
                                }
                            ),
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "name": "lookup_smiles",
                            "arguments": json.dumps({"name": "aspirin"}),
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "content": "The aspirin SMILES is "
                    "CC(=O)OC1=CC=CC=C1C(=O)O.",
                    "tool_calls": [],
                }
            ),
            json.dumps(
                {
                    "content": "The chemistry worker found the aspirin SMILES.",
                    "tool_calls": [],
                }
            ),
        ]
    )
    model = CodexChatModel(model_id="gpt-5.6-terra")
    worker = construct_single_agent_graph(
        model,
        system_prompt="Use the lookup tool before answering.",
        tools=[lookup_smiles],
        checkpointer=None,
    )
    graph = construct_main_agent_graph(
        model,
        subagents=[
            {
                "name": "chemgraph",
                "description": "Executes chemistry lookup tasks.",
                "runnable": worker,
            }
        ],
    )

    result = await MainAgentSession(
        graph,
        thread_id="codex-main-agent-test",
    ).run("Find the aspirin SMILES")

    assert result.status == "completed"
    assert result.assistant_response == (
        "The chemistry worker found the aspirin SMILES."
    )
    assert len(fake_codex_sdk.run_calls) == 4


def test_codex_models_never_require_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert check_api_keys("codex:o3") == (True, "")


def test_codex_install_hint_preserves_extra_name(monkeypatch):
    def fail_to_initialize(*args, **kwargs):
        raise ImportError(
            "Install it with `pip install 'chemgraph[codex]'`."
        )

    monkeypatch.setattr("chemgraph.agent.llm_agent.ChemGraph", fail_to_initialize)

    with console.capture() as capture:
        agent = commands.initialize_agent(
            model_name="codex:test-model",
            workflow_type="single_agent",
            structured_output=False,
            return_option="last_message",
            generate_report=False,
            recursion_limit=10,
        )

    assert agent is None
    output = capture.get()
    assert "Codex CLI separately" in output
    assert "chemgraph[codex]" in output
    assert "codex login" in output
