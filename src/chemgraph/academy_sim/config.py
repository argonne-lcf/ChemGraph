"""User config schema for thin graph-to-graph Academy replacement demos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from chemgraph.academy_sim.errors import AcademySimConfigError
from chemgraph.mcp.config import MCPToolSourceConfig


class ModelConfig(BaseModel):
    """LLM configuration for one graph."""

    model_config = ConfigDict(extra='forbid')

    config_file: Path | None = None


class ExchangeConfig(BaseModel):
    """Academy exchange settings."""

    model_config = ConfigDict(extra='forbid')

    type: Literal['local', 'redis', 'hybrid', 'http', 'globus'] = 'local'
    registration: Literal['file', 'exchange'] = 'file'
    url: str | None = None
    redis_host: str = '127.0.0.1'
    redis_port: int = 6379
    namespace: str | None = None

    @model_validator(mode='after')
    def _normalize_exchange_type(self) -> 'ExchangeConfig':
        if self.type == 'globus':
            self.type = 'http'
        return self


class ArtifactConfig(BaseModel):
    """Run artifact location."""

    model_config = ConfigDict(extra='forbid')

    run_dir: Path = Path('academy_sim_runs')


class PeerToolConfig(BaseModel):
    """Peer communication tool settings."""

    model_config = ConfigDict(extra='forbid')

    enabled: bool = True


class GraphConfig(BaseModel):
    """One ChemGraph graph exposed through Academy transport."""

    model_config = ConfigDict(extra='forbid')

    name: str = ''
    site: str = 'local'
    workflow_type: str = 'single_agent'
    model: ModelConfig | None = None
    startup_prompt: str | None = None
    system_prompt: str | None = None
    planner_prompt: str | None = None
    executor_prompt: str | None = None
    allowed_peers: tuple[str, ...] = ()
    science_tools: tuple[MCPToolSourceConfig, ...] = ()
    peer_tools: PeerToolConfig = Field(default_factory=PeerToolConfig)
    workflow_kwargs: dict[str, Any] = Field(default_factory=dict)
    idle_timeout_s: float = 600.0
    poll_interval_s: float = 2.0


class AcademySimConfig(BaseModel):
    """Top-level config for academy_sim launchers and demos."""

    model_config = ConfigDict(extra='forbid')

    run_id: str = Field(min_length=1)
    task: str = Field(min_length=1)
    initial_graph: str | None = None
    bootstrap_mode: Literal['inline', 'manual'] = 'inline'
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)
    model: ModelConfig | None = None
    graphs: dict[str, GraphConfig]

    @model_validator(mode='after')
    def _validate_graphs(self) -> 'AcademySimConfig':
        if not self.graphs:
            raise ValueError('at least one graph is required')
        graph_names = set(self.graphs)
        if self.initial_graph is not None and self.initial_graph not in graph_names:
            raise ValueError(
                f'initial_graph {self.initial_graph!r} is not a configured graph'
            )
        updated: dict[str, GraphConfig] = {}
        for name, graph in self.graphs.items():
            if graph.name and graph.name != name:
                raise ValueError(
                    f'graph key {name!r} does not match graph.name {graph.name!r}'
                )
            unknown = sorted(set(graph.allowed_peers).difference(graph_names))
            if unknown:
                raise ValueError(
                    f'graph {name!r} references unknown allowed_peers: {unknown}'
                )
            if len(set(graph.allowed_peers)) != len(graph.allowed_peers):
                raise ValueError(f'graph {name!r} has duplicate allowed_peers')
            if graph.model is None and self.model is None:
                raise ValueError(
                    f'graph {name!r} requires graph.model or top-level model'
                )
            updated[name] = graph.model_copy(update={'name': name})
        self.graphs = updated
        return self

    def graph(self, name: str) -> GraphConfig:
        try:
            return self.graphs[name]
        except KeyError as exc:
            raise AcademySimConfigError(
                f'unknown graph {name!r}; available graphs: {sorted(self.graphs)}'
            ) from exc

    def model_for_graph(self, name: str) -> ModelConfig:
        graph_model = self.graph(name).model
        model = graph_model or self.model
        if model is None:
            raise AcademySimConfigError(f'graph {name!r} has no model config')
        return model

    def run_dir(self) -> Path:
        base = self.artifacts.run_dir
        if base.name == self.run_id:
            return base
        return base / self.run_id


def load_config(path: str | Path) -> AcademySimConfig:
    """Load an academy_sim JSON/JSONC config file."""

    config_path = Path(path).resolve()
    data = json.loads(_strip_json_comments(config_path.read_text(encoding='utf-8')))
    if not isinstance(data, dict):
        raise AcademySimConfigError(f'config {config_path} must contain an object')
    try:
        config = AcademySimConfig.model_validate(data)
    except Exception as exc:
        raise AcademySimConfigError(f'invalid academy_sim config {config_path}: {exc}') from exc
    return _resolve_paths(config, base_dir=config_path.parent)


def _resolve_paths(config: AcademySimConfig, *, base_dir: Path) -> AcademySimConfig:
    run_dir = config.artifacts.run_dir
    if not run_dir.is_absolute():
        run_dir = (base_dir / run_dir).resolve()
    model = _resolve_model(config.model, base_dir=base_dir)
    graphs = {
        name: graph.model_copy(
            update={'model': _resolve_model(graph.model, base_dir=base_dir)}
        )
        for name, graph in config.graphs.items()
    }
    return config.model_copy(
        update={
            'artifacts': config.artifacts.model_copy(update={'run_dir': run_dir}),
            'model': model,
            'graphs': graphs,
        }
    )


def _resolve_model(model: ModelConfig | None, *, base_dir: Path) -> ModelConfig | None:
    if model is None or model.config_file is None:
        return model
    path = model.config_file
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return model.model_copy(update={'config_file': path})


def _strip_json_comments(text: str) -> str:
    """Remove JSONC comments while preserving string contents."""

    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        char = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ''
        if in_string:
            out.append(char)
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            i += 1
            continue
        if char == '/' and nxt == '/':
            i += 2
            while i < len(text) and text[i] not in '\r\n':
                i += 1
            continue
        if char == '/' and nxt == '*':
            i += 2
            while i < len(text):
                if text[i] in '\r\n':
                    out.append(text[i])
                    i += 1
                    continue
                if text[i] == '*' and i + 1 < len(text) and text[i + 1] == '/':
                    i += 2
                    break
                i += 1
            continue
        out.append(char)
        i += 1
    return ''.join(out)
