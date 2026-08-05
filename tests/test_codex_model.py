import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from chemgraph.agent import llm_agent
from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.cli import commands
from chemgraph.cli.commands import check_api_keys
from chemgraph.cli.formatting import console
from chemgraph.graphs.single_agent import construct_single_agent_graph
from chemgraph.models import codex as codex_model
from chemgraph.models.codex import (
    CodexAuthenticationError,
    CodexChatModel,
    CodexResponseError,
    _strip_codex_prefix,
)
from chemgraph.models import loader


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


def test_shared_loader_routes_codex_prefix(monkeypatch):
    monkeypatch.setattr(
        loader,
        "load_codex_model",
        lambda model_name: ("codex-model", model_name),
    )

    assert loader.load_chat_model("codex:test-model") == (
        "codex-model",
        "codex:test-model",
    )


def test_chemgraph_routes_codex_to_single_agent(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        llm_agent,
        "load_codex_model",
        lambda model_name: ("codex-model", model_name),
    )
    monkeypatch.setattr(
        llm_agent,
        "construct_single_agent_graph",
        lambda llm, *_args, **_kwargs: captured.setdefault("llm", llm),
    )

    ChemGraph(
        model_name="codex:test-model",
        workflow_type="single_agent",
        enable_memory=False,
        log_dir=str(tmp_path),
    )

    assert captured["llm"] == ("codex-model", "codex:test-model")


def test_chemgraph_rejects_codex_for_other_workflows():
    with pytest.raises(ValueError, match="only the single_agent workflow"):
        ChemGraph(model_name="codex:test-model", workflow_type="multi_agent")


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
    assert "chemgraph[codex]" in capture.get()
