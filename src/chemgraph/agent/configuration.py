"""One normalized main-agent configuration for construction and reconstruction."""

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping

from langchain_core.utils.function_calling import convert_to_openai_tool

from chemgraph import __version__
from chemgraph.graphs import workspace as workspace_policy
from chemgraph.graphs.workspace import resolve_workspace_interrupt_policy
from chemgraph.memory.graph_config import (
    GRAPH_SCHEMA_VERSION, NEW_SESSION_GUIDANCE, describe_agent_registry,
    describe_backend, describe_tool_registry, describe_worker_options, fingerprint,
)
from chemgraph.memory.schemas import MainAgentGraphConfig
from chemgraph.registry.agents import AgentRegistry
from chemgraph.registry.tools import RegistryError, ToolRegistry
from chemgraph.skills.runtime import local_skill_workspace


_COMMON_SETTINGS = (
    "model_name", "recursion_limit", "reasoning_effort", "structured_output",
    "generate_report", "max_retries", "human_supervised", "terminal_tool_names",
    "enable_deepagent", "configuration_id",
    "approval_mode",
)
_WORKSPACE_SETTINGS = ("skills", "skill_dirs", "discover_skills", "user_skills_dir")


class _FrozenList(tuple):
    pass


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [_thaw(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    return value


@dataclass(frozen=True)
class MainAgentRuntimeConfig:
    """Frozen options; live tools/backends remain caller-owned opaque objects."""

    saved: MainAgentGraphConfig
    _graph_options: Mapping

    def graph_arguments(self):
        return _thaw(self._graph_options)

    @classmethod
    def from_agent(cls, agent, *, endpoint, default_prompts):
        settings = {name: getattr(agent, name) for name in _COMMON_SETTINGS}
        settings.update({name: getattr(agent, name) for name in _WORKSPACE_SETTINGS})
        settings.update({f"deepagent_{name}": getattr(agent, f"deepagent_{name}")
                         for name in _WORKSPACE_SETTINGS})
        backend, backend_cli, backend_opaque = describe_backend(agent.backend)
        legacy, _, legacy_opaque = describe_backend(agent.deepagent_backend)
        worker_options, options_opaque = describe_worker_options(agent.subagent_options)
        tools, custom_tools = describe_tool_registry(agent.tool_registry)
        workers, custom_workers = describe_agent_registry(agent.agent_registry)
        workspace = local_skill_workspace(agent.backend)
        policy = (None if agent.approval_mode == "bypass"
                  else resolve_workspace_interrupt_policy(agent.tool_registry))
        settings.update(
            artifact_directory=agent.log_dir, model_endpoint=endpoint,
            graph_schema_version=GRAPH_SCHEMA_VERSION, package_version=__version__,
            workspace=str(workspace[0]) if workspace else None,
            deepagent_workspace=legacy.get("workspace"),
            registry_tool_names=agent.tool_registry.names(),
            configured_subagent_names=agent.subagent_names,
            subagent_names=agent.agent_registry.names(),
            main_agent_prompt=agent.main_agent_prompt,
            requires_configuration_id=(backend_opaque or (agent.enable_deepagent and legacy_opaque)
                                       or options_opaque or custom_tools or custom_workers
                                       or bool(agent.tools) or endpoint.caller_owned),
            cli_restorable=(backend_cli and (not agent.enable_deepagent or legacy["type"] == "cli-local-shell-v1")
                            and not agent.subagent_options and not custom_tools and not custom_workers
                            and not agent.tools and default_prompts and not endpoint.caller_owned),
            tool_signatures=tuple(sorted(
                f"{getattr(tool, 'name', type(tool).__name__)}:"
                f"{getattr(getattr(tool, 'args_schema', None), '__name__', '')}"
                for tool in agent.tools or []
            )),
        )
        saved = MainAgentGraphConfig(**settings)
        graph = {
            "main_tools": agent.tools, "agent_registry": agent.agent_registry,
            "agent_options": agent.subagent_options, "backend": agent.backend,
            "tool_registry": agent.tool_registry, "interrupt_on": policy,
            "system_prompt": saved.main_agent_prompt, "recursion_limit": saved.recursion_limit,
            "human_supervised": saved.human_supervised,
            **{name: getattr(saved, name) for name in _WORKSPACE_SETTINGS},
            **{f"subagent_{name}": getattr(saved, name) for name in (
                "structured_output", "generate_report", "max_retries", "human_supervised", "terminal_tool_names",
            )},
            **{f"subagent_{name}": getattr(agent, name) for name in (
                "system_prompt", "formatter_prompt", "report_prompt",
            )},
            "enable_deepagent": saved.enable_deepagent,
            "deepagent_backend": agent.deepagent_backend,
            "deepagent_system_prompt": agent.deepagent_prompt,
            "deepagent_recursion_limit": saved.recursion_limit,
            **{f"deepagent_{name}": getattr(saved, f"deepagent_{name}") for name in _WORKSPACE_SETTINGS},
        }
        # The effective review_policy already identifies the mode. Excluding
        # this new descriptive field preserves existing reviewed checkpoints.
        identity = saved.model_dump(exclude={"package_version", "topology_fingerprint", "approval_mode"})
        identity.update(
            main_backend=backend, legacy_worker_backend=legacy if saved.enable_deepagent else None,
            registry_specs=tools, agent_specs=workers, subagent_options=worker_options,
            review_policy={"version": workspace_policy.WORKSPACE_REVIEW_POLICY_VERSION, "interrupt_on": graph["interrupt_on"]},
            prompts={key: value for key, value in graph.items() if "prompt" in key},
            custom_tool_schemas=[(convert_to_openai_tool(tool), getattr(tool, "return_direct", False))
                                for tool in [*(agent.tools or []), *[
                                    agent.tool_registry.get(spec.name) for spec in agent.tool_registry.specs()
                                    if spec.import_path is None
                                ]]],
        )
        saved = saved.model_copy(update={"topology_fingerprint": fingerprint(identity)})
        return cls(saved, _freeze(graph))


def validate_cli_configuration(config):
    """Fail closed before constructing a model or requesting workspace access."""
    if config.graph_schema_version != GRAPH_SCHEMA_VERSION:
        raise ValueError(f"This main-agent graph is incompatible. {NEW_SESSION_GUIDANCE}")
    if not config.artifact_directory or config.model_endpoint is None:
        raise ValueError(f"The saved execution context is incomplete. {NEW_SESSION_GUIDANCE}")
    if (not config.cli_restorable or not config.topology_fingerprint or config.requires_configuration_id
            or config.model_endpoint.caller_owned):
        raise ValueError("This session requires caller-provided Python configuration. Reconstruct it through the Python API.")


def workspace_arguments(config):
    """Translate the normalized saved workspace into the existing CLI adapter."""
    catalog = ToolRegistry()
    try:
        registry = ToolRegistry(catalog.get_spec(name) for name in config.registry_tool_names)
        workers = AgentRegistry()
        agent_registry = AgentRegistry(workers.get_spec(name) for name in config.subagent_names)
    except RegistryError as exc:
        raise ValueError(f"Cannot reconstruct the stored catalog: {exc}. {NEW_SESSION_GUIDANCE}") from exc
    return {
        "workspace": config.workspace,
        **{name: getattr(config, name) for name in _WORKSPACE_SETTINGS},
        "tool_registry": registry,
        "agent_registry": agent_registry,
        "subagent_names": config.configured_subagent_names,
        "main_agent_prompt": config.main_agent_prompt, "configuration_id": config.configuration_id,
    }


def restoration_arguments(config):
    """The only translation from saved configuration to CLI initialization."""
    validate_cli_configuration(config)
    return {
        **{name: getattr(config, name) for name in _COMMON_SETTINGS},
        **workspace_arguments(config),
        "workflow_type": "main_agent", "log_dir": config.artifact_directory,
        "base_url": config.model_endpoint.base_url, "model_endpoint": config.model_endpoint,
        "deepagent_workspace": config.deepagent_workspace,
        **{f"deepagent_{name}": getattr(config, f"deepagent_{name}") for name in _WORKSPACE_SETTINGS},
    }
