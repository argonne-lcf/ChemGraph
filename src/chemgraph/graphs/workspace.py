"""Workspace capabilities shared by independently constructed agent graphs."""

from collections.abc import Sequence
from copy import deepcopy

from deepagents.backends import CompositeBackend, LocalShellBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol

from chemgraph.agent.deepagent_backend import (
    DEEPAGENT_ENV_ALLOWLIST, host_shell_environment, resolve_workspace,
)
from chemgraph.registry.middleware import RegistryToolsMiddleware
from chemgraph.registry.tools import ToolRegistry
from chemgraph.skills.runtime import ChemGraphSkillsMiddleware, prepare_skill_backend


# Bump when approval semantics change, including removal of reviewed tools.
WORKSPACE_REVIEW_POLICY_VERSION = 4


DEFAULT_WORKSPACE_INTERRUPT_ON = {
    "execute": {"allowed_decisions": ["approve", "reject"]},
    "write_file": {"allowed_decisions": ["approve", "reject"]},
    "edit_file": {"allowed_decisions": ["approve", "reject"]},
    "delete": {"allowed_decisions": ["approve", "reject"]},
    "smiles_to_coordinate_file": {"allowed_decisions": ["approve", "reject"]},
    "save_atomsdata_to_file": {"allowed_decisions": ["approve", "reject"]},
}

# Additional built-ins that read host files, write artifacts, or launch calculations.
# Apply the same mandatory policy to direct tools and registry workers.
_REGISTRY_REVIEW_TOOLS = {
    "run_ase", "run_docking", "run_graspa", "run_xanes", "generate_html",
    "fetch_xanes_data", "plot_xanes_data",
    "load_document", "file_to_atomsdata", "extract_output_json",
}


_DEFAULT_INTERRUPT_POLICY = object()
_WORKSPACE_MOUNT = "/workspace/"


class _CLIWorkspaceBackend(LocalShellBackend):
    """Host backend with an explicit, reproducible CLI construction policy."""

    def __init__(self, root):
        environment = host_shell_environment()
        super().__init__(root_dir=root, virtual_mode=True, env=environment, inherit_env=False)
        self._cli_environment = dict(environment)
        self._cli_descriptor = {
            "type": "cli-local-shell-v1", "workspace": str(self.cwd.resolve()),
            "virtual_mode": True, "timeout": self._default_timeout,
            "max_output_bytes": self._max_output_bytes,
            "environment_policy": list(DEEPAGENT_ENV_ALLOWLIST),
        }

    def with_artifact_directory(self, directory):
        """Copy a supported backend so sessions never share mutable shell state."""
        from copy import copy

        backend = copy(self)
        backend._env = {**self._env, "CHEMGRAPH_LOG_DIR": directory}
        backend._cli_environment = dict(backend._env)
        return backend


def bind_artifact_directory(backend, directory):
    """Bind reproducible CLI backends without modifying caller-owned environments."""
    if cli_backend_descriptor(backend) is not None:
        return backend.with_artifact_directory(directory)
    return backend


def create_cli_workspace_backend(workspace):
    """Construct the CLI policy after the caller has handled host-access approval."""
    if workspace is None:
        raise ValueError("Workspace must be a non-empty host directory path.")
    return _CLIWorkspaceBackend(resolve_workspace(workspace))


def cli_backend_descriptor(backend):
    """Recognize only unchanged instances produced by the shared CLI factory."""
    if type(backend) is not _CLIWorkspaceBackend:
        return None
    descriptor = backend._cli_descriptor
    if (backend._env != backend._cli_environment
            or str(backend.cwd.resolve()) != descriptor["workspace"]
            or backend.virtual_mode is not True
            or backend._default_timeout != descriptor["timeout"]
            or backend._max_output_bytes != descriptor["max_output_bytes"]):
        return None
    return dict(descriptor)


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


def default_tool_registry(tools=(), *, human_supervised=False) -> ToolRegistry:
    """Create a lazy catalog excluding attached and non-opted-in interactive tools."""
    attached_names = {
        entry.get("function", entry).get("name", entry.get("type"))
        if isinstance(entry, dict)
        else getattr(entry, "name", getattr(entry, "__name__", None))
        for entry in tools or ()
    }
    return ToolRegistry(
        spec for spec in ToolRegistry().specs()
        if spec.name not in attached_names
        and (human_supervised or not spec.interactive)
    )


def resolve_workspace_interrupt_policy(tool_registry=None, interrupt_on=_DEFAULT_INTERRUPT_POLICY):
    """Freeze the effective approval policy for graph construction and identity."""
    if interrupt_on is not _DEFAULT_INTERRUPT_POLICY:
        return deepcopy(interrupt_on)
    policy = deepcopy(DEFAULT_WORKSPACE_INTERRUPT_ON)
    policy.update({
        name: {"allowed_decisions": ["approve", "reject"]}
        for name in _REGISTRY_REVIEW_TOOLS
    })
    return policy


def prepare_workspace_runtime(
    *, backend=None, tools=(), tool_registry=None, skills=None,
    discover_skills=True, user_skills_dir=None, skill_dirs=None,
    interrupt_on=_DEFAULT_INTERRUPT_POLICY,
):
    """Return the backend, skill sources, middleware, and approval policy."""
    effective_interrupt_on = resolve_workspace_interrupt_policy(tool_registry, interrupt_on)
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
    return effective_backend, sources, middleware, effective_interrupt_on
