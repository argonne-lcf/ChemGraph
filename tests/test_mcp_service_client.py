"""Bearer credentials stay in environment variables and out of CLI output."""

from types import SimpleNamespace

import pytest

from chemgraph.cli import mcp_utils
from chemgraph.cli.main import create_argument_parser, load_config


@pytest.mark.parametrize("fail", [False, True])
def test_client_sends_env_token_and_redacts_errors(monkeypatch, fail):
    captured = []
    secret = "test-only-not-a-real-service-token"
    monkeypatch.setenv("TEST_CHEMGRAPH_TOKEN", secret)
    monkeypatch.setattr(mcp_utils.time, "sleep", lambda _: None)

    class Client:
        def __init__(self, connections):
            captured.append(connections)

        async def get_tools(self):
            if fail:
                raise RuntimeError(f"transport failed with {secret}")
            return [SimpleNamespace(name="sample")]

    monkeypatch.setattr("langchain_mcp_adapters.client.MultiServerMCPClient", Client)
    with mcp_utils.console.capture() as output:
        result = mcp_utils.load_mcp_tools_from_config(
            url="https://chemgraph.example/mcp", token_env="TEST_CHEMGRAPH_TOKEN")
    connection = next(iter(captured[0].values()))
    assert connection["headers"] == {"Authorization": f"Bearer {secret}"}
    assert secret not in output.get()
    assert bool(result) is not fail


def test_missing_token_and_remote_plaintext_fail_before_connect(monkeypatch):
    monkeypatch.delenv("MISSING_TOKEN", raising=False)
    assert mcp_utils.load_mcp_tools_from_config(url="https://example.test/mcp", token_env="MISSING_TOKEN") is None
    monkeypatch.setenv("TEST_TOKEN", "test-only-token")
    assert mcp_utils.load_mcp_tools_from_config(url="http://example.test/mcp", token_env="TEST_TOKEN") is None


def test_cli_and_toml_accept_env_variable_name(tmp_path):
    args = create_argument_parser().parse_args(["run", "--mcp-token-env", "MY_TOKEN"])
    assert args.mcp_token_env == "MY_TOKEN"
    path = tmp_path / "client.toml"
    path.write_text('[mcp]\nurl="https://chemgraph.example/mcp"\ntoken_env="MY_TOKEN"\n')
    assert load_config(str(path))["mcp_token_env"] == "MY_TOKEN"
    args = create_argument_parser().parse_args(["mcp", "--config", "service.toml", "token", "issue", "alice"])
    assert args.service_config == "service.toml" and args.user == "alice"
