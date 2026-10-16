"""Shared MCP tool loading for ChemGraph runtimes and the CLI."""

from __future__ import annotations

import os
from collections.abc import Iterable

from langchain_core.tools import BaseTool

from chemgraph.mcp.config import MCPToolSourceConfig

# Env vars that the MCP stdio subprocess may need. The MCP SDK's stdio
# transport inherits only a hard-coded whitelist of standard system vars
# (PATH, HOME, etc.) by default -- ChemGraph- and Globus-specific keys
# must be passed through explicitly or the spawned MCP server has no way
# to see what the user exported in their shell.
_FORWARDED_ENV_VARS = (
    # Shell essentials (so python and the user's HOME resolve correctly)
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    # ChemGraph runtime selection
    "CHEMGRAPH_EXECUTION_BACKEND",
    "CHEMGRAPH_LOG_DIR",
    # Globus Compute
    "GLOBUS_COMPUTE_ENDPOINT_ID",
    # Globus Transfer
    "GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID",
    "GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID",
    "GLOBUS_TRANSFER_DESTINATION_BASE_PATH",
    # ALCF inference endpoints
    "ALCF_ACCESS_TOKEN",
)


async def load_mcp_tools(sources: Iterable[MCPToolSourceConfig]) -> list[BaseTool]:
    """Load and filter each server separately, preserving tool provenance."""
    sources = list(sources)
    if not sources:
        return []
    from langchain_mcp_adapters.client import MultiServerMCPClient

    connections = {}
    for source in sources:
        if source.name in connections:
            raise ValueError(f"Duplicate MCP server name: {source.name!r}")
        if source.transport == "stdio":
            env = {key: os.environ[key] for key in _FORWARDED_ENV_VARS if key in os.environ}
            connections[source.name] = {
                "transport": "stdio", "command": source.command,
                "args": list(source.args), "env": {**env, **source.env},
            }
        else:
            connections[source.name] = {"transport": "streamable_http", "url": source.url}
    client = MultiServerMCPClient(connections)
    selected: list[BaseTool] = []
    names: set[str] = set()
    for source in sources:
        tools = await client.get_tools(server_name=source.name)
        available = {tool.name for tool in tools}
        allowed = set(source.allowed_tools)
        if source.require_allowed_tools and (missing := allowed - available):
            raise ValueError(f"MCP server {source.name!r} is missing tools: {sorted(missing)}")
        for tool in tools:
            if allowed and tool.name not in allowed:
                continue
            if tool.name in names:
                raise ValueError(f"Duplicate MCP tool name: {tool.name!r}")
            names.add(tool.name)
            tool.metadata = {**(tool.metadata or {}), "mcp_server": source.name}
            selected.append(tool)
    return selected
