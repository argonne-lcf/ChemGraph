import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langchain_core.tools import tool

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
        assert self.state.active
        assert Path(self.state.clients[-1].config.kwargs["cwd"]).is_dir()
        self.state.run_calls.append((prompt, kwargs))
        response = self.state.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, SimpleNamespace):
            return response
        return SimpleNamespace(
            final_response=response,
            usage=SimpleNamespace(last=_FakeUsageBreakdown()),
        )


@pytest.fixture
def fake_codex_sdk(monkeypatch):
    state = SimpleNamespace(
        account={"type": "chatgpt", "email": "chemist@example.com"},
        responses=[],
        active=False,
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
            state.active = True
            return self

        def __exit__(self, *_args):
            state.active = False
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
                        "arguments": {"name": "aspirin"},
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
                            "arguments": {"name": "aspirin"},
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
                            "arguments": {
                                "subagent_type": "chemgraph",
                                "description": "Look up the aspirin SMILES.",
                            },
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
                            "arguments": {"name": "aspirin"},
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


@tool
def execute(command: str, timeout: int | None = None) -> str:
    """Record a command without running a shell."""
    return command


@tool
def flexible(options: dict) -> dict:
    """Return free-form options using the compatibility encoding."""
    return options


def _decision(name=None, arguments=None):
    return json.dumps({
        "content": "" if name else "Done",
        "tool_calls": [{"name": name, "arguments": arguments}] if name else [],
    })


def _payload(state):
    return json.loads(state.run_calls[0][0].split("conversation:\n", 1)[1])


@pytest.mark.parametrize("extra", [{}, {"timeout": None}, {"timeout": 0}])
def test_codex_preserves_structured_commands_and_optional_arguments(fake_codex_sdk, extra):
    command = "python - <<'PY'\nprint(\"H₂O\", r'C:\\tmp', 'quoted \\\"text\\\"')\nPY\n"
    arguments = {"command": command, **extra}
    fake_codex_sdk.responses.append(_decision("execute", arguments))
    model = CodexChatModel(model_id="test").bind_tools([execute, flexible])
    result = model.invoke("Prepare the input")
    assert result.tool_calls[0]["args"] == arguments
    assert result.response_metadata["codex_decision_attempts"] == 1
    assert _payload(fake_codex_sdk)["argument_encodings"] == {
        "execute": "object", "flexible": "json_string",
    }


@pytest.mark.parametrize("bad_response", [
    None, "", "{", "[]", '{"content": null, "tool_calls": []}',
    '{"content": "", "tool_calls": [], "extra": "private"}',
    '{"content": "", "tool_calls": [null]}',
    _decision("execute", []), _decision("execute", '{"command": "pwd"}'),
    _decision("flexible", {"options": {}}), _decision("flexible", "[]"),
    _decision("flexible", '{"options": {"secret": "private"}'),
    _decision("flexible", '{"options": {"value": NaN}}'),
])
@pytest.mark.asyncio
async def test_codex_corrects_malformed_responses_in_same_thread(fake_codex_sdk, bad_response):
    fake_codex_sdk.responses.extend([
        bad_response, _decision("flexible", json.dumps({"options": {"x": 1}})),
    ])
    model = CodexChatModel(model_id="test").bind_tools([execute, flexible])
    result = await model.ainvoke("Use the tools")
    assert result.tool_calls[0]["args"] == {"options": {"x": 1}}
    assert result.response_metadata["codex_decision_attempts"] == 2
    assert result.usage_metadata == {
        "input_tokens": 22, "output_tokens": 14, "total_tokens": 36,
    }
    assert len(fake_codex_sdk.thread_start_calls) == 1
    first, correction = fake_codex_sdk.run_calls
    assert first[1] == correction[1]
    assert "complete replacement" in correction[0]
    assert "private" not in correction[0]
    assert not fake_codex_sdk.active


def test_codex_exhaustion_has_safe_parser_details_and_no_partial_calls(fake_codex_sdk, monkeypatch):
    response = json.loads(_decision("execute", {"command": "private command"}))
    response["tool_calls"].append({"name": "flexible", "arguments": '{"private":'})
    fake_codex_sdk.responses.extend([json.dumps(response)] * 3)
    ids = []
    monkeypatch.setattr(codex_model.uuid, "uuid4", lambda: ids.append("allocated"))
    model = CodexChatModel(model_id="test").bind_tools([execute, flexible])
    with pytest.raises(CodexResponseError) as caught:
        model.invoke("Use the tools")
    message = str(caught.value)
    assert "after 3 attempts" in message
    assert "flexible" in message and "line 1, column" in message
    assert "private" not in message
    assert ids == []
    assert len(fake_codex_sdk.run_calls) == 3
    assert not fake_codex_sdk.active


def test_codex_omits_incomplete_usage_totals(fake_codex_sdk):
    fake_codex_sdk.responses.extend([
        SimpleNamespace(final_response="{", usage=None), _decision(),
    ])
    result = CodexChatModel(model_id="test").invoke("Answer")
    assert result.usage_metadata is None
    assert result.response_metadata["codex_decision_attempts"] == 2


@pytest.mark.parametrize("response", [RuntimeError("SDK failure"), KeyboardInterrupt()])
def test_codex_does_not_retry_sdk_errors_or_interrupts(fake_codex_sdk, response):
    fake_codex_sdk.responses.append(response)
    with pytest.raises(type(response)):
        CodexChatModel(model_id="test").invoke("Answer")
    assert len(fake_codex_sdk.run_calls) == 1
    assert not fake_codex_sdk.active


@pytest.mark.parametrize(("choice", "parallel", "calls", "error"), [
    ("none", False, [{"name": "execute", "arguments": {}}], "tools were disabled"),
    ("required", True, [], "required tool call"),
    ("execute", True, [{"name": "flexible", "arguments": "{}"}], "instead of required"),
    (None, False, [{"name": "execute", "arguments": {}}] * 2, "parallel tool calls"),
    (None, True, [{"name": "unknown-private", "arguments": {}}], "unknown tool"),
])
def test_codex_constraints_fail_without_retry(fake_codex_sdk, choice, parallel, calls, error):
    fake_codex_sdk.responses.append(json.dumps({"content": "", "tool_calls": calls}))
    model = CodexChatModel(model_id="test").bind_tools(
        [execute, flexible], tool_choice=choice, parallel_tool_calls=parallel,
    )
    with pytest.raises(CodexResponseError, match=error) as caught:
        model.invoke("Use the tools")
    assert "private" not in str(caught.value)
    assert len(fake_codex_sdk.run_calls) == 1
    schema = fake_codex_sdk.run_calls[0][1]["output_schema"]["properties"]["tool_calls"]
    if choice == "none":
        assert schema["maxItems"] == 0


@pytest.mark.parametrize("tool_name", ["execute", "write_file"])
@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_codex_corrected_actions_still_require_approval(
    fake_codex_sdk, tmp_path, tool_name, decision,
):
    effects = []

    class RecordingBackend(LocalShellBackend):
        def write(self, file_path, content):
            effects.append((file_path, content))
            return super().write(file_path, content)

        def execute(self, command, *, timeout=None):
            effects.append(command)
            return ExecuteResponse(output="Recorded", exit_code=0)

    content = 'line one\nprint("H₂O", r"C:\\tmp")\n'
    args = ({"command": content} if tool_name == "execute" else {
        "file_path": "/workspace/result.txt", "content": content,
    })
    bad = json.loads(_decision(tool_name, args))
    bad["tool_calls"].append({"name": tool_name, "arguments": []})
    fake_codex_sdk.responses.extend([
        json.dumps(bad), _decision(tool_name, args), "{", _decision(),
    ])
    graph = construct_deep_agent_graph(
        CodexChatModel(model_id="test"),
        backend=RecordingBackend(root_dir=tmp_path, env={}),
    )
    config = {"configurable": {"thread_id": "codex-correction-approval"}}
    state = graph.invoke({"messages": [HumanMessage(content="Perform the task")]}, config)
    assert state["__interrupt__"]
    assert len(fake_codex_sdk.run_calls) == 2
    encodings = _payload(fake_codex_sdk)["argument_encodings"]
    for name in ("ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute", "task"):
        assert encodings[name] == "object"
    assert effects == []
    state = graph.invoke(Command(resume={"decisions": [{"type": decision}]}), config)
    assert "__interrupt__" not in state
    assert len(effects) == (1 if decision == "approve" else 0)
    if tool_name == "write_file" and decision == "approve":
        assert (tmp_path / "result.txt").read_text(encoding="utf-8") == content
    elif tool_name == "execute" and decision == "approve":
        assert effects == [content]
    assert len(fake_codex_sdk.run_calls) == 4
    assert len(fake_codex_sdk.thread_start_calls) == 2
    for thread in fake_codex_sdk.thread_start_calls:
        assert thread["approval_mode"] == _FakeApprovalMode.deny_all
        assert thread["sandbox"] == _FakeSandbox.read_only


def test_codex_parameter_validation_stays_in_tool_framework(fake_codex_sdk):
    fake_codex_sdk.responses.extend([
        _decision("execute", {"command": "echo safe", "timeout": "not an integer"}),
        _decision(),
    ])
    graph = construct_single_agent_graph(
        CodexChatModel(model_id="test"), tools=[execute],
    )
    state = graph.invoke(
        {"messages": "Run the task"},
        {"configurable": {"thread_id": "invalid-parameter"}},
    )
    assert state["messages"][-2].status == "error"
    assert "timeout" in state["messages"][-2].content
    assert len(fake_codex_sdk.thread_start_calls) == 2



def test_codex_accepts_third_attempt_and_counts_all_usage(fake_codex_sdk):
    fake_codex_sdk.responses.extend(["{", "{", _decision()])
    result = CodexChatModel(model_id="test").invoke("Answer")
    assert result.response_metadata["codex_decision_attempts"] == 3
    assert result.usage_metadata == {
        "input_tokens": 33, "output_tokens": 21, "total_tokens": 54,
    }
    assert len(fake_codex_sdk.thread_start_calls) == 1


def test_codex_generation_authentication_error_is_not_retried(fake_codex_sdk):
    fake_codex_sdk.account = {"type": "apiKey"}
    with pytest.raises(CodexAuthenticationError):
        CodexChatModel(model_id="test").invoke("Answer")
    assert fake_codex_sdk.run_calls == []
    assert fake_codex_sdk.thread_start_calls == []


def test_codex_duplicate_tool_names_fail_at_binding():
    with pytest.raises(ValueError, match="distinct names"):
        CodexChatModel(model_id="test").bind_tools([execute, execute])
