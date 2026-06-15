from __future__ import annotations

import dataclasses
import json
import pathlib
from collections.abc import Mapping
from typing import Any

from chemgraph.academy.campaigns import resolve_campaign
from pydantic import BaseModel, ConfigDict, Field, field_validator


_REMOVED_CAMPAIGN_FIELDS = frozenset(
    {
        'completion_criteria',
        'parameters',
        'routing_policy',
        'workflow_stages',
    },
)
_RESOURCE_KINDS = frozenset({'directory', 'file', 'json'})
_RESOURCE_SCOPES = frozenset(
    {
        'absolute',
        'campaign_file',
        'external',
        'shared_run',
    },
)


class MCPServerSpec(BaseModel):
    """Campaign-declared MCP server subprocess available to agents."""

    model_config = ConfigDict(extra='forbid')

    name: str = Field(min_length=1)
    command: str = Field(
        min_length=1,
        description=(
            "Shell command to launch the MCP server. Tokens after the first "
            "are arguments. Do not include --transport/--host/--port; the "
            "supervisor adds them."
        ),
    )
    env: dict[str, str] = Field(default_factory=dict)

    @field_validator('name', 'command')
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('field must be non-empty')
        return value


class ResourceSpec(BaseModel):
    """Campaign-declared resource or artifact handle.

    The runtime resolves only these explicit ``path`` fields. It never scans
    arbitrary campaign metadata looking for strings that might be paths.
    """

    model_config = ConfigDict(extra='forbid')

    kind: str
    path: str | None = None
    uri: str | None = None
    scope: str = 'campaign_file'
    description: str = ''
    expose_content: bool = False

    @field_validator('kind')
    @classmethod
    def _known_resource_kind(cls, value: str) -> str:
        value = value.strip()
        if value not in _RESOURCE_KINDS:
            raise ValueError(
                f'resource kind must be one of {sorted(_RESOURCE_KINDS)}',
            )
        return value

    @field_validator('scope')
    @classmethod
    def _known_resource_scope(cls, value: str) -> str:
        value = value.strip()
        if value not in _RESOURCE_SCOPES:
            raise ValueError(
                f'resource scope must be one of {sorted(_RESOURCE_SCOPES)}',
            )
        return value

    @field_validator('path', 'uri', 'description')
    @classmethod
    def _strip_optional_resource_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


@dataclasses.dataclass(frozen=True)
class ChemGraphAgentSpec:
    name: str
    role: str
    mission: str
    allowed_peers: tuple[str, ...]
    mcp_servers: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    """Optional per-agent whitelist of MCP tool names.

    Empty (the default) means the agent sees every tool advertised by the
    servers listed in :attr:`mcp_servers`. When non-empty, only tools whose
    name appears in this tuple are exposed to the agent; everything else
    that the servers advertise is filtered out before reaching LangChain.

    The whitelist is flat and server-agnostic: a name matches any tool with
    that name across the agent's connected servers. Duplicate tool names
    across an agent's servers are still rejected by the supervisor (today's
    behavior), so the whitelist does not introduce new ambiguity.
    """
    resources: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ChemGraphCampaign:
    run_id: str
    user_task: str
    initial_agent: str
    prompt_profile: pathlib.Path
    agents: tuple[ChemGraphAgentSpec, ...]
    mcp_servers: tuple[MCPServerSpec, ...] = ()
    resources: Mapping[str, ResourceSpec] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ChemGraphDaemonConfig:
    run_dir: pathlib.Path
    run_token: str
    agent_count: int
    campaign_config: pathlib.Path
    lm_config: pathlib.Path
    max_decisions: int
    poll_timeout_s: float
    idle_timeout_s: float
    startup_timeout_s: float
    completion_timeout_s: float
    status_interval_s: float
    redis_host: str
    redis_port: int
    redis_namespace: str
    rank: int
    local_rank: int | None
    chemgraph_repo_root: pathlib.Path
    exchange_type: str = 'redis'


def namespace_for_run(run_dir: pathlib.Path) -> str:
    return f'academy-chemgraph-swarm:{run_dir.name}'


def resolve_campaign_resources(
    campaign: ChemGraphCampaign,
    run_dir: str | pathlib.Path,
    *,
    shared_dir_name: str = 'shared',
) -> ChemGraphCampaign:
    """Resolve explicit shared-run resource paths for one concrete run."""
    shared_root = (pathlib.Path(run_dir).resolve() / shared_dir_name)
    resources: dict[str, ResourceSpec] = {}

    for name, spec in campaign.resources.items():
        if spec.path is None:
            resources[name] = spec
            continue
        if spec.scope != 'shared_run':
            resources[name] = spec
            continue
        path = pathlib.Path(spec.path)
        resolved = path if path.is_absolute() else shared_root / path
        resources[name] = spec.model_copy(
            update={
                'path': str(resolved.resolve()),
                'uri': spec.uri or _file_uri(resolved.resolve()),
            },
        )

    return dataclasses.replace(campaign, resources=resources)


def _file_uri(path: pathlib.Path) -> str:
    return path.resolve().as_uri()


def _resolve_resource_spec(
    raw: Mapping[str, Any],
    *,
    campaign_path: pathlib.Path,
) -> ResourceSpec:
    spec = ResourceSpec.model_validate(raw)
    if spec.path is None:
        return spec
    if spec.scope == 'campaign_file':
        path = pathlib.Path(spec.path)
        resolved = path if path.is_absolute() else campaign_path.parent / path
        resolved = resolved.resolve()
        return spec.model_copy(
            update={
                'path': str(resolved),
                'uri': spec.uri or _file_uri(resolved),
            },
        )
    if spec.scope == 'absolute':
        path = pathlib.Path(spec.path)
        if not path.is_absolute():
            raise RuntimeError(
                f'absolute resource path must be absolute: {spec.path}',
            )
        resolved = path.resolve()
        return spec.model_copy(
            update={
                'path': str(resolved),
                'uri': spec.uri or _file_uri(resolved),
            },
        )
    if spec.scope in {'shared_run', 'external'}:
        return spec

    raise RuntimeError(f'unsupported resource scope {spec.scope!r}')


def load_campaign(path: str | pathlib.Path) -> ChemGraphCampaign:
    path = resolve_campaign(path)
    data = _load_jsonc(path)
    _reject_removed_campaign_fields(data, campaign_path=path)
    prompt_profile = _resolve_campaign_relative_path(
        data.get('prompt_profile'),
        campaign_path=path,
        field_name='prompt_profile',
    )

    mcp_servers = tuple(
        MCPServerSpec.model_validate(raw)
        for raw in data.get('mcp_servers', ())
    )
    resources = {
        name: _resolve_resource_spec(raw, campaign_path=path)
        for name, raw in dict(data.get('resources', {})).items()
    }
    agents = []
    for item in data['agents']:
        agents.append(
            ChemGraphAgentSpec(
                name=item['name'],
                role=item['role'],
                mission=item['mission'],
                allowed_peers=tuple(item.get('allowed_peers', ())),
                mcp_servers=tuple(item.get('mcp_servers', ())),
                allowed_tools=tuple(item.get('allowed_tools', ())),
                resources=tuple(item.get('resources', ())),
            ),
        )
    return ChemGraphCampaign(
        run_id=data.get('run_id', path.stem),
        user_task=data['user_task'],
        initial_agent=data.get('initial_agent', agents[0].name),
        prompt_profile=prompt_profile,
        agents=tuple(agents),
        mcp_servers=mcp_servers,
        resources=resources,
    )


def _load_jsonc(path: pathlib.Path) -> dict[str, Any]:
    """Load a campaign file that may contain JSONC-style comments."""
    data = json.loads(_strip_json_comments(path.read_text(encoding='utf-8')))
    if not isinstance(data, dict):
        raise RuntimeError(f'campaign {path} must contain a JSON object')
    return data


def _strip_json_comments(text: str) -> str:
    """Remove // and /* */ comments without touching JSON string values."""
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


def _reject_removed_campaign_fields(
    data: Mapping[str, Any],
    *,
    campaign_path: pathlib.Path,
) -> None:
    removed = sorted(_REMOVED_CAMPAIGN_FIELDS.intersection(data))
    if not removed:
        return
    raise RuntimeError(
        f'campaign {campaign_path} uses removed structured orchestration '
        f'field(s): {removed}. Put simple natural-language coordination hints '
        'in agent mission fields and enforce the communication graph with '
        'allowed_peers.',
    )


def _resolve_campaign_relative_path(
    raw: Any,
    *,
    campaign_path: pathlib.Path,
    field_name: str,
) -> pathlib.Path:
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError(f'campaign requires non-empty {field_name!r}')
    path = pathlib.Path(raw.strip())
    if not path.is_absolute():
        path = campaign_path.parent / path
    return path.resolve()


def validate_campaign(campaign: ChemGraphCampaign, agent_count: int) -> None:
    if len(campaign.agents) != agent_count:
        raise RuntimeError(
            f'campaign defines {len(campaign.agents)} agents but '
            f'agent_count={agent_count}',
        )
    names = [agent.name for agent in campaign.agents]
    if len(set(names)) != len(names):
        raise RuntimeError('campaign agent names must be unique')
    if campaign.initial_agent not in names:
        raise RuntimeError(
            f'initial_agent {campaign.initial_agent!r} is not an agent',
        )
    server_names = [server.name for server in campaign.mcp_servers]
    if len(set(server_names)) != len(server_names):
        raise RuntimeError('campaign MCP server names must be unique')
    declared_servers = set(server_names)
    for agent in campaign.agents:
        unknown = sorted(set(agent.allowed_peers).difference(names))
        if unknown:
            raise RuntimeError(
                f'{agent.name} has unknown allowed peers: {unknown}',
            )
        if agent.name in agent.allowed_peers:
            raise RuntimeError(f'{agent.name} must not list itself as a peer')
        unknown_servers = sorted(set(agent.mcp_servers).difference(declared_servers))
        if unknown_servers:
            raise RuntimeError(
                f'{agent.name} references unknown MCP servers: {unknown_servers}',
            )
        if agent.allowed_tools:
            if len(set(agent.allowed_tools)) != len(agent.allowed_tools):
                raise RuntimeError(
                    f'{agent.name} has duplicate allowed_tools entries',
                )
            if not agent.mcp_servers:
                raise RuntimeError(
                    f'{agent.name} declares allowed_tools but no mcp_servers '
                    'to draw them from',
                )
        unknown_resources = sorted(set(agent.resources).difference(campaign.resources))
        if unknown_resources:
            raise RuntimeError(
                f'{agent.name} references unknown resources: {unknown_resources}',
            )


def selected_agent(campaign: ChemGraphCampaign, rank: int) -> ChemGraphAgentSpec:
    if rank < 0 or rank >= len(campaign.agents):
        raise RuntimeError(
            f'MPI rank {rank} has no agent. Launch exactly '
            f'{len(campaign.agents)} ranks for this campaign.',
        )
    return campaign.agents[rank]


def campaign_bootstrap_text(campaign: ChemGraphCampaign) -> str:
    initial_agent = next(
        (agent for agent in campaign.agents if agent.name == campaign.initial_agent),
        None,
    )
    initial_resources = initial_agent.resources if initial_agent is not None else ()
    payload: dict[str, Any] = {
        'user_task': campaign.user_task,
        'resources': _resources_payload(campaign, initial_resources),
        'resource_data': _resource_data_payload(campaign, initial_resources),
    }
    return json.dumps(payload, sort_keys=True)


def _resources_payload(
    campaign: ChemGraphCampaign,
    resource_names: tuple[str, ...] | list[str],
) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for name in resource_names:
        spec = campaign.resources.get(name)
        if spec is None:
            continue
        payload[name] = spec.model_dump(exclude_none=True)
    return payload


def _resource_data_payload(
    campaign: ChemGraphCampaign,
    resource_names: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in resource_names:
        spec = campaign.resources.get(name)
        if spec is None or not spec.expose_content:
            continue
        if spec.kind != 'json' or spec.path is None:
            continue
        path = pathlib.Path(spec.path)
        if not path.exists():
            raise FileNotFoundError(f'campaign resource does not exist: {path}')
        payload[name] = json.loads(path.read_text(encoding='utf-8'))
    return payload


def visible_resources_payload(
    campaign: ChemGraphCampaign,
    agent: ChemGraphAgentSpec,
) -> dict[str, dict[str, Any]]:
    return _resources_payload(campaign, agent.resources)
