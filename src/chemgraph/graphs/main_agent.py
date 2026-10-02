"""Compatibility entry point for the durable DeepAgent-based main agent."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware.subagents import CompiledSubAgent
from langchain_core.tools import BaseTool

from chemgraph.graphs.deep_agent import DEFAULT_DEEPAGENT_PROMPT, construct_deep_agent_graph
from chemgraph.graphs.subagents import (
    latest_assistant_text as latest_assistant_text,
    _preserve_terminal_tool_output as _preserve_terminal_tool_output,
    _validate_subagents,
)
from chemgraph.graphs.workspace import _DEFAULT_INTERRUPT_POLICY, default_tool_registry
from chemgraph.memory.subagent_recorder import SubagentRunRecorder
from chemgraph.registry.agents import AgentRegistry
from chemgraph.registry.tools import ToolRegistry


DEFAULT_MAIN_AGENT_PROMPT = DEFAULT_DEEPAGENT_PROMPT


def _validate_main_tools(main_tools: Sequence[BaseTool]) -> None:
    names = [getattr(item, "name", "") for item in main_tools]
    if any(not name for name in names):
        raise ValueError("Every supervisor tool must have a non-empty name.")
    reserved = {
        "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep",
        "execute", "task", "write_todos", "search_tools", "load_tools", "search_agents", "load_agents",
    }
    if conflicts := sorted(reserved.intersection(names)):
        formatted = ", ".join(repr(name) for name in conflicts)
        raise ValueError(
            f"Supervisor tool name(s) {formatted} are reserved for middleware."
        )
    if len(names) != len(set(names)):
        raise ValueError("Supervisor tool names must be unique.")


def construct_main_agent_graph(
    llm: Any,
    *,
    subagents: Sequence[CompiledSubAgent] | None = None,
    main_tools: list[BaseTool] | None = None,
    agent_registry: Any | None = None,
    agent_options: dict[str, dict[str, Any]] | None = None,
    backend: BackendProtocol | None = None,
    tool_registry: ToolRegistry | None = None,
    skills: Sequence[str] | None = None,
    skill_dirs: Sequence[str] | None = None,
    discover_skills: bool = True,
    user_skills_dir: str | None = None,
    interrupt_on: dict[str, Any] | None | object = _DEFAULT_INTERRUPT_POLICY,
    recursion_limit: int = 200,
    human_supervised: bool = False,
    subagent_tools: list[BaseTool] | None = None,
    subagent_system_prompt: str | None = None,
    subagent_formatter_prompt: str | None = None,
    subagent_report_prompt: str | None = None,
    subagent_structured_output: bool = False,
    subagent_generate_report: bool = False,
    subagent_max_retries: int = 1,
    subagent_human_supervised: bool = False,
    subagent_terminal_tool_names: Collection[str] = (),
    enable_deepagent: bool = False,
    deepagent_backend: BackendProtocol | None = None,
    deepagent_skills: Sequence[str] | None = None,
    deepagent_discover_skills: bool = True,
    deepagent_user_skills_dir: str | None = None,
    deepagent_skill_dirs: Sequence[str] | None = None,
    deepagent_recursion_limit: int = 200,
    deepagent_system_prompt: str = DEFAULT_DEEPAGENT_PROMPT,
    system_prompt: str = DEFAULT_MAIN_AGENT_PROMPT,
    checkpointer: Any | None = None,
    subagent_recorder: SubagentRunRecorder | None = None,
):
    """Build the durable main agent through the shared DeepAgent factory.

    Specialists are discovered and loaded on demand. Explicit compiled workers
    remain attached; legacy ``enable_deepagent`` explicitly activates its worker.
    Without a backend, file tools use checkpoints and no shell is exposed.
    """
    if deepagent_recursion_limit <= 0:
        raise ValueError("deepagent_recursion_limit must be positive.")
    if not enable_deepagent and any((deepagent_backend is not None, deepagent_skills, deepagent_skill_dirs)):
        raise ValueError("deepagent_backend/skills/skill_dirs requires enable_deepagent=True.")
    supervisor_tools = list(main_tools or [])
    _validate_main_tools(supervisor_tools)
    workers = _validate_subagents(list(subagents or []), subagent_recorder)
    attached_names = {worker["name"] for worker in workers}
    registry = agent_registry if agent_registry is not None else AgentRegistry(
        spec for spec in AgentRegistry().specs()
        if not spec.test_only and not attached_names.intersection((spec.name, *spec.aliases))
    )
    if not isinstance(registry, AgentRegistry):
        raise TypeError("agent_registry must be an AgentRegistry.")
    options = {}
    for name, value in (agent_options or {}).items():
        canonical = registry.resolve_name(name)
        if canonical in options:
            raise ValueError("Worker options repeat a name or alias.")
        if not isinstance(value, dict):
            raise TypeError("Worker options must be mappings.")
        options[canonical] = dict(value)
    legacy_conflict = "deep_agent" in options
    if "single_agent" in registry.names():
        chemistry = {
            "structured_output": subagent_structured_output,
            "generate_report": subagent_generate_report,
            "max_retries": subagent_max_retries,
            "human_supervised": subagent_human_supervised,
            "terminal_tool_names": subagent_terminal_tool_names,
        }
        for key, value in (("tools", subagent_tools), ("system_prompt", subagent_system_prompt),
                           ("formatter_prompt", subagent_formatter_prompt), ("report_prompt", subagent_report_prompt)):
            if value is not None:
                chemistry[key] = value
        options["single_agent"] = {**chemistry, **options.get("single_agent", {})}
    if "deep_agent" in registry.names():
        options["deep_agent"] = {
            "backend": backend, "skills": skills, "skill_dirs": skill_dirs,
            "discover_skills": discover_skills, "user_skills_dir": user_skills_dir,
            "recursion_limit": recursion_limit, **options.get("deep_agent", {}),
        }
    if enable_deepagent:
        if "deep_agent" not in registry.names():
            registry = AgentRegistry([*registry.specs(), AgentRegistry().get_spec("deep_agent")])
        if legacy_conflict:
            raise ValueError("deep_agent is already configured; omit enable_deepagent.")
        options["deep_agent"] = {
            "tools": [], "backend": deepagent_backend, "skills": deepagent_skills,
            "skill_dirs": deepagent_skill_dirs, "discover_skills": deepagent_discover_skills,
            "user_skills_dir": deepagent_user_skills_dir,
            "recursion_limit": deepagent_recursion_limit, "system_prompt": deepagent_system_prompt,
        }
    return construct_deep_agent_graph(
        llm, tools=supervisor_tools,
        tool_registry=(tool_registry if tool_registry is not None else
                       default_tool_registry(supervisor_tools, human_supervised=human_supervised)),
        agent_registry=registry, agent_options=options, subagents=workers,
        initial_agents=("deep_agent",) if enable_deepagent else (),
        subagent_recorder=subagent_recorder, restrict_delegation=True,
        backend=backend, skills=skills, skill_dirs=skill_dirs,
        discover_skills=discover_skills, user_skills_dir=user_skills_dir,
        interrupt_on=interrupt_on, system_prompt=system_prompt,
        recursion_limit=recursion_limit,
        **({"checkpointer": checkpointer} if checkpointer is not None else {}),
        name="main_agent",
    )


__all__ = ["DEFAULT_DEEPAGENT_PROMPT", "DEFAULT_MAIN_AGENT_PROMPT",
           "construct_main_agent_graph", "latest_assistant_text"]
