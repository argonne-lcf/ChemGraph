"""Reusable Deep Agent workflow for workspace and coding tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends.protocol import BackendProtocol
from langgraph.checkpoint.memory import InMemorySaver

from chemgraph.registry.tools import ToolRegistry
from chemgraph.graphs.workspace import (
    DEFAULT_WORKSPACE_INTERRUPT_ON as DEFAULT_DEEPAGENT_INTERRUPT_ON,
    _DEFAULT_INTERRUPT_POLICY,
    _normalize_backend as _normalize_backend,
    normalize_skill_sources as normalize_skill_sources,
    prepare_workspace_runtime,
)


DEFAULT_DEEPAGENT_PROMPT = """\
You are ChemGraph's Deep Agent. Complete workspace tasks and use attached
chemistry tools or write scripts guided by the available skills for simulations. Read
the relevant available skill before carrying out a specialized workflow.
Inspect tool schemas and report actual results; never invent chemistry results.
Preserve the user's execution method and calculator choices. Skill-guided
scripts can use the existing chemistry Python APIs without attached MCP tools.
Follow existing action approvals and report missing execution capabilities.

Treat `/workspace` as the project root when that mount exists. Follow the
"Shell paths vs. virtual paths" mappings for execution. Packaged skills at
`/chemgraph-skills/` are readable resources, not paths in the execution
filesystem. Copy any required template/helper into that filesystem before
using it in a command. Remote MCP servers and compute workers may have yet
another filesystem; establish input visibility before submitting work.

Use direct tools for focused work. When agent discovery is available, use
search_agents and load_agents before delegating substantial specialist work
through task. Give workers self-contained inputs and constraints and review
actual results. Do not run workspace mutations in parallel.
Return a self-contained report of results, paths, job IDs, and unresolved work.
"""


_DEFAULT_CHECKPOINTER = object()


def construct_deep_agent_graph(
    llm: Any,
    *,
    tools: Sequence[Any] | None = None,
    tool_registry: ToolRegistry | None = None,
    skills: Sequence[str] | None = None,
    discover_skills: bool = True,
    user_skills_dir: str | None = None,
    skill_dirs: Sequence[str] | None = None,
    system_prompt: str = DEFAULT_DEEPAGENT_PROMPT,
    backend: BackendProtocol | None = None,
    interrupt_on: dict[str, Any] | None | object = _DEFAULT_INTERRUPT_POLICY,
    recursion_limit: int = 200,
    checkpointer: Any = _DEFAULT_CHECKPOINTER,
    name: str = "deepagent",
    subagents: Sequence[Any] | None = None,
    agent_registry: Any | None = None,
    agent_options: dict[str, dict[str, Any]] | None = None,
    initial_agents: Sequence[str] = (),
    subagent_recorder: Any | None = None,
    restrict_delegation: bool = False,
):
    """Construct a standalone or parent-checkpointed workspace Deep Agent.

    Standalone construction receives an in-memory checkpointer so approval
    interrupts can be resumed. Orchestrators should explicitly pass
    ``checkpointer=None`` so the worker inherits the parent graph's checkpoint.
    Bundled skills are always available. ``discover_skills`` also discovers
    personal and project sources for supported local workspaces. ``skills``
    adds ordered backend-relative sources, overriding discovered/bundled names.
    ``skill_dirs`` mounts explicit host directories, independently of the workspace
    and discovery setting, before backend-relative ``skills`` sources.
    ``user_skills_dir`` fixes the personal root when restoring a session.
    ``tool_registry`` adds metadata discovery and on-demand local tools; ``tools``
    remains the always-attached tool list. Registry tools run on the agent host.
    Passing ``interrupt_on=None`` disables approval interrupts and should be
    reserved for an externally isolated, explicitly trusted execution context.
    """
    if recursion_limit <= 0:
        raise ValueError("recursion_limit must be positive.")

    effective_checkpointer = (
        InMemorySaver()
        if checkpointer is _DEFAULT_CHECKPOINTER
        else checkpointer
    )
    effective_backend, sources, middleware, effective_interrupt_on = prepare_workspace_runtime(
        backend=backend, tools=tools, tool_registry=tool_registry, skills=skills,
        discover_skills=discover_skills, user_skills_dir=user_skills_dir,
        skill_dirs=skill_dirs, interrupt_on=interrupt_on,
    )
    registered = list(subagents or [])
    if agent_registry is not None or restrict_delegation:
        from chemgraph.registry.agent_middleware import RegistryAgentsMiddleware
        from chemgraph.registry.agents import AgentRegistry

        loader = RegistryAgentsMiddleware(
            agent_registry if agent_registry is not None else AgentRegistry([]),
            llm=llm, options=agent_options, interrupt_on=effective_interrupt_on,
            attached_workers=registered, initial_agents=initial_agents,
            recorder=subagent_recorder, tool_middleware=middleware,
        )
        registered.extend(loader.proxies())
        middleware.append(loader)
    workflow = create_deep_agent(
        model=llm,
        subagents=registered,
        tools=list(tools or []),
        skills=sources,
        middleware=middleware,
        system_prompt=system_prompt,
        backend=effective_backend,
        interrupt_on=effective_interrupt_on,
        checkpointer=effective_checkpointer,
        name=name,
    )
    return workflow.with_config({"recursion_limit": recursion_limit})


__all__ = [
    "DEFAULT_DEEPAGENT_INTERRUPT_ON",
    "DEFAULT_DEEPAGENT_PROMPT",
    "construct_deep_agent_graph",
    "normalize_skill_sources",
]
