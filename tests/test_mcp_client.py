from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from chemgraph.mcp.client import load_mcp_tools
from chemgraph.mcp.config import MCPToolSourceConfig


@pytest.fixture
def client(monkeypatch):
    import langchain_mcp_adapters.client
    fake = SimpleNamespace(get_tools=AsyncMock())
    factory = lambda connections: fake  # noqa: E731
    monkeypatch.setattr(langchain_mcp_adapters.client, "MultiServerMCPClient", factory)
    return fake


@pytest.mark.asyncio
async def test_filters_per_server_without_metadata(client):
    client.get_tools.side_effect = [
        [SimpleNamespace(name=name, metadata=None) for name in ("a", "unexpected")],
        [SimpleNamespace(name=name, metadata=None) for name in ("b", "unexpected")],
    ]
    sources = [MCPToolSourceConfig(name=name, url="http://localhost/mcp", allowed_tools=(tool,))
               for name, tool in (("one", "a"), ("two", "b"))]
    result = await load_mcp_tools(sources)
    assert [tool.name for tool in result] == ["a", "b"]
    assert [tool.metadata["mcp_server"] for tool in result] == ["one", "two"]
    assert [call.kwargs for call in client.get_tools.call_args_list] == [
        {"server_name": "one"}, {"server_name": "two"},
    ]


@pytest.mark.asyncio
async def test_missing_required_tool(client):
    client.get_tools.return_value = []
    with pytest.raises(ValueError, match="missing tools"):
        await load_mcp_tools([MCPToolSourceConfig(
            name="one", url="http://localhost/mcp", allowed_tools=("absent",),
        )])


@pytest.mark.asyncio
async def test_duplicate_tool_names_fail(client):
    client.get_tools.side_effect = [[SimpleNamespace(name="same", metadata=None)] for _ in range(2)]
    with pytest.raises(ValueError, match="Duplicate MCP tool"):
        await load_mcp_tools([MCPToolSourceConfig(name=name, url="http://localhost/mcp")
                              for name in ("one", "two")])


@pytest.mark.asyncio
async def test_stdio_forwards_runtime_environment(monkeypatch):
    import langchain_mcp_adapters.client
    captured = []
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", "/run/artifacts")
    monkeypatch.setenv("GLOBUS_COMPUTE_ENDPOINT_ID", "endpoint")
    monkeypatch.setattr(langchain_mcp_adapters.client, "MultiServerMCPClient",
                        lambda connections: captured.append(connections) or SimpleNamespace(get_tools=AsyncMock(return_value=[])))
    await load_mcp_tools([MCPToolSourceConfig(name="one", transport="stdio", command="python", env={"CUSTOM": "value"})])
    env = captured[0]["one"]["env"]
    assert env["CHEMGRAPH_LOG_DIR"] == "/run/artifacts"
    assert env["GLOBUS_COMPUTE_ENDPOINT_ID"] == "endpoint"
    assert env["CUSTOM"] == "value"
