"""Generic MCP tool loading for ChemGraph graphs.

This module deliberately lives under ``chemgraph.mcp`` so Academy integrations
do not own MCP/science tool loading.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from langchain_core.tools import BaseTool


class MCPToolSource(Protocol):
    """Protocol shared with academy_sim config objects."""

    name: str
    transport: str
    url: str | None
    command: str | None
    args: tuple[str, ...]
    env: dict[str, str]
    allowed_tools: tuple[str, ...]
    require_allowed_tools: bool


async def load_mcp_tools(sources: Iterable[MCPToolSource]) -> list[BaseTool]:
    """Load LangChain tools from configured MCP sources."""

    from langchain_mcp_adapters.client import MultiServerMCPClient

    connections: dict[str, dict[str, Any]] = {}
    filters: dict[str, tuple[set[str], bool]] = {}
    for source in sources:
        if source.transport == 'streamable_http':
            connections[source.name] = {
                'transport': 'streamable_http',
                'url': source.url,
            }
        elif source.transport == 'stdio':
            connections[source.name] = {
                'transport': 'stdio',
                'command': source.command,
                'args': list(source.args),
                'env': dict(source.env),
            }
        else:
            raise ValueError(f'unsupported MCP transport {source.transport!r}')
        filters[source.name] = (
            set(source.allowed_tools),
            bool(source.require_allowed_tools),
        )

    if not connections:
        return []

    client = MultiServerMCPClient(connections)
    tools = await client.get_tools()
    return _filter_tools(tools, filters)


def _filter_tools(
    tools: list[BaseTool],
    filters: dict[str, tuple[set[str], bool]],
) -> list[BaseTool]:
    """Apply per-source filters when adapter metadata is available.

    ``langchain-mcp-adapters`` returns flattened tools. Some versions preserve
    server metadata and some do not. When metadata is absent and there is only
    one source, filtering is still unambiguous.
    """

    source_names = set(filters)
    single_source = next(iter(source_names)) if len(source_names) == 1 else None
    selected: list[BaseTool] = []
    seen: set[str] = set()
    matched: dict[str, set[str]] = {name: set() for name in source_names}

    for tool in tools:
        source_name = _tool_source_name(tool)
        if source_name not in filters and single_source is not None:
            source_name = single_source
        allowed, _required = filters.get(source_name, (set(), False))
        if allowed and tool.name not in allowed:
            continue
        if tool.name in seen:
            raise RuntimeError(f'duplicate MCP tool name {tool.name!r}')
        seen.add(tool.name)
        if source_name in matched:
            matched[source_name].add(tool.name)
        selected.append(tool)

    missing: dict[str, list[str]] = {}
    for source_name, (allowed, required) in filters.items():
        if required and allowed:
            absent = sorted(allowed.difference(matched[source_name]))
            if absent:
                missing[source_name] = absent
    if missing:
        raise RuntimeError(f'MCP sources did not provide required tools: {missing}')
    return selected


def _tool_source_name(tool: BaseTool) -> str | None:
    metadata = getattr(tool, 'metadata', None)
    if isinstance(metadata, dict):
        for key in ('mcp_server', 'server_name', 'server'):
            value = metadata.get(key)
            if isinstance(value, str):
                return value
    return None
