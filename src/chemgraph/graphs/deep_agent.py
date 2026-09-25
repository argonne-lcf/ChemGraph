"""Reusable Deep Agent workflow for workspace and coding tasks."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol
from langgraph.checkpoint.memory import InMemorySaver

from chemgraph.registry.middleware import RegistryToolsMiddleware
from chemgraph.registry.tools import ToolRegistry
from chemgraph.skills.runtime import ChemGraphSkillsMiddleware, prepare_skill_backend


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

Return a self-contained report of results, paths, job IDs, and unresolved work.
"""


DEFAULT_DEEPAGENT_INTERRUPT_ON = {
    "execute": {"allowed_decisions": ["approve", "reject"]},
    "write_file": {"allowed_decisions": ["approve", "reject"]},
    "edit_file": {"allowed_decisions": ["approve", "reject"]},
    "delete": {"allowed_decisions": ["approve", "reject"]},
    "smiles_to_coordinate_file": {"allowed_decisions": ["approve", "reject"]},
    "save_atomsdata_to_file": {"allowed_decisions": ["approve", "reject"]},
}

# Additional built-ins that read host files, write artifacts, or launch calculations.
# Apply only to registry tools, preserving the policy for explicitly attached tools.
_REGISTRY_REVIEW_TOOLS = {
    "run_ase", "run_docking", "run_graspa", "run_xanes", "generate_html",
    "fetch_xanes_data", "plot_xanes_data",
    "load_document", "file_to_atomsdata", "extract_output_json",
}


_DEFAULT_CHECKPOINTER = object()
_DEFAULT_INTERRUPT_POLICY = object()
_WORKSPACE_MOUNT = "/workspace/"


def normalize_skill_sources(skills: Sequence[str] | None) -> tuple[str, ...]:
    """Validate and freeze ordered backend-relative skill source paths."""
    if skills is None:
        return ()
    if isinstance(skills, (str, bytes)):
        raise TypeError("skills must be a sequence of path strings, not a string.")

    normalized: list[str] = []
    for source in skills:
        if not isinstance(source, str):
            raise TypeError("Every skill source must be a string.")
        if not source.strip():
            raise ValueError("Skill source paths must not be empty.")
        normalized.append(source)
    return tuple(normalized)


class _WorkspaceShellBackend(StateBackend, LocalShellBackend):
    """Use checkpoint files at the root while retaining local-shell execution.

    StateBackend precedes LocalShellBackend so file operations never expose a
    second host-file namespace. The local-shell type also preserves Deep Agents'
    execution detection and virtual-to-host path guidance.
    """

    def __init__(self, shell: LocalShellBackend):
        StateBackend.__init__(self)
        self.shell = shell

    @property
    def id(self) -> str:
        return self.shell.id

    def execute(self, command: str, *, timeout: int | None = None):
        return self.shell.execute(command, timeout=timeout)

    async def aexecute(self, command: str, *, timeout: int | None = None):
        return await self.shell.aexecute(command, timeout=timeout)

    async def agrep(self, pattern, path=None, glob=None, *, max_count=None):
        # FilesystemBackend's async override searches the host directly.
        return await BackendProtocol.agrep(
            self, pattern, path, glob, max_count=max_count,
        )


def _normalize_backend(backend: BackendProtocol) -> BackendProtocol:
    """Mount a virtual local workspace at the path Deep Agent expects."""
    if isinstance(backend, LocalShellBackend) and backend.virtual_mode:
        return CompositeBackend(
            default=_WorkspaceShellBackend(backend),
            routes={_WORKSPACE_MOUNT: backend},
        )
    return backend


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
    effective_interrupt_on = (
        deepcopy(DEFAULT_DEEPAGENT_INTERRUPT_ON)
        if interrupt_on is _DEFAULT_INTERRUPT_POLICY
        else interrupt_on
    )
    effective_backend = _normalize_backend(
        backend if backend is not None else StateBackend()
    )
    skill_sources = normalize_skill_sources(skills)
    effective_backend, sources, optional = prepare_skill_backend(
        effective_backend, skill_sources, discover_skills=discover_skills,
        user_skills_dir=user_skills_dir, skill_dirs=skill_dirs,
    )
    middleware = [ChemGraphSkillsMiddleware(
        backend=effective_backend, sources=sources, optional=optional,
    )]
    if tool_registry is not None and tool_registry.names():
        if interrupt_on is _DEFAULT_INTERRUPT_POLICY:
            effective_interrupt_on.update({
                tool_name: {"allowed_decisions": ["approve", "reject"]}
                for tool_name in _REGISTRY_REVIEW_TOOLS.intersection(tool_registry.names())
            })
        loader = RegistryToolsMiddleware(tool_registry, attached_tools=tools or ())
        attached_names = {
            entry.get("function", entry).get("name", entry.get("type"))
            if isinstance(entry, dict)
            else getattr(entry, "name", getattr(entry, "__name__", None))
            for entry in tools or []
        }
        loader.validate_names(attached_names)
        if attached_names & {"search_tools", "load_tools"}:
            raise ValueError("Attached tools conflict with registry discovery tool names.")
        middleware.append(loader)
    workflow = create_deep_agent(
        model=llm,
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
