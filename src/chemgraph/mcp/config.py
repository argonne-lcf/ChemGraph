"""MCP source config shared by ChemGraph runtimes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MCPToolSourceConfig(BaseModel):
    """One MCP source whose tools can be bound into a ChemGraph graph."""

    model_config = ConfigDict(extra='forbid')

    source: Literal['mcp'] = 'mcp'
    name: str = Field(min_length=1)
    transport: Literal['streamable_http', 'stdio'] = 'streamable_http'
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    require_allowed_tools: bool = True

    @model_validator(mode='after')
    def _validate_connection(self) -> 'MCPToolSourceConfig':
        if self.transport == 'streamable_http':
            if not self.url:
                raise ValueError('streamable_http MCP source requires url')
            if self.command or self.args:
                raise ValueError('streamable_http MCP source must not set command/args')
        if self.transport == 'stdio':
            if not self.command:
                raise ValueError('stdio MCP source requires command')
            if self.url:
                raise ValueError('stdio MCP source must not set url')
        return self
