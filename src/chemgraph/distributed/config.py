"""Configuration for independently launched ChemGraph peers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chemgraph.mcp.config import MCPToolSourceConfig

Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


class PeerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_type: Literal["main_agent", "single_agent", "multi_agent"] = "main_agent"
    model_name: str = "gpt-4o-mini"
    base_url: str | None = None
    allowed_peers: tuple[Name, ...] = ()
    mcp_servers: tuple[MCPToolSourceConfig, ...] = ()
    prompts: dict[str, str] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    idle_timeout_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_options(self):
        reserved = {"workflow_type", "model_name", "base_url", "tools", "prompts", "log_dir",
                    "on_event", "checkpointer", "session_store", "human_input_handler", "memory_db_path",
                    "configuration_id", "return_option", "tool_registry"}
        if conflicts := reserved.intersection(self.options):
            raise ValueError(f"Peer options override runtime-owned fields: {sorted(conflicts)}")
        if len(self.allowed_peers) != len(set(self.allowed_peers)):
            raise ValueError("Duplicate allowed_peers")
        return self


class DistributedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: Name
    launch_id: Name
    exchange_url: str
    exchange_auth_method: Literal["globus"] | None = None
    exchange_header_env: dict[str, str] = Field(default_factory=dict)
    run_dir: Path = Path("distributed_runs")
    timeout_s: float = Field(default=30, gt=0)
    peers: dict[Name, PeerConfig]

    @model_validator(mode="after")
    def validate_peers(self):
        if not self.peers:
            raise ValueError("At least one peer is required")
        if any(not header.strip() or not variable.strip() for header, variable in self.exchange_header_env.items()):
            raise ValueError("Exchange header names and environment variable names must not be empty")
        if len({header.lower() for header in self.exchange_header_env}) != len(self.exchange_header_env):
            raise ValueError("Duplicate exchange header names")
        for name, peer in self.peers.items():
            if name == "operator":
                raise ValueError("The peer name 'operator' is reserved")
            if unknown := set(peer.allowed_peers) - self.peers.keys():
                raise ValueError(f"Unknown peers for {name!r}: {sorted(unknown)}")
            if name in peer.allowed_peers:
                raise ValueError(f"Peer {name!r} cannot message itself")
        return self

    def peer_dir(self, name: str) -> Path:
        if name not in self.peers:
            raise ValueError(f"Unknown peer: {name!r}")
        return self.run_dir / self.run_id / self.launch_id / name


def load_config(path: str | Path) -> DistributedConfig:
    path = Path(path).resolve()
    config = DistributedConfig.model_validate(json.loads(path.read_text()))
    if not config.run_dir.is_absolute():
        config.run_dir = (path.parent / config.run_dir).resolve()
    return config
