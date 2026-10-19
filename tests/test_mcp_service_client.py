"""Bearer credentials stay in environment variables and out of CLI output."""

from types import SimpleNamespace

import pytest

from chemgraph.cli import mcp_utils


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
