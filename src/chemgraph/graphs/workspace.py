"""Workspace capabilities shared by independently constructed agent graphs."""

import os
import shutil
import stat
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from deepagents.backends import CompositeBackend, LocalShellBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol, DeleteResult
from deepagents.backends.utils import validate_path

from chemgraph.agent.deepagent_backend import (
    DEEPAGENT_ENV_ALLOWLIST, host_shell_environment, resolve_workspace,
)
from chemgraph.graphs.tool_reasons import ToolReasonMiddleware
from chemgraph.registry.middleware import RegistryToolsMiddleware
from chemgraph.registry.tools import ToolRegistry
from chemgraph.skills.runtime import (
    BUNDLED_SKILLS_PATH, EXTERNAL_SKILLS_PATH, USER_SKILLS_PATH,
    ChemGraphSkillsMiddleware, local_skill_workspace, prepare_skill_backend,
)


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
# Deep Agents file tools reject drive-letter paths, so Windows keeps the virtual mount.
_CLI_VIRTUAL_MODE = os.name == "nt"
# Deep Agents' offloaded tool results and summarized history stay in checkpoint state,
# at the same paths a checkpoint-only supervisor uses.
_CLI_STATE_PREFIXES = ("/large_tool_results/", "/conversation_history/")


class _CLIWorkspaceBackend(LocalShellBackend):
    """Host backend with an explicit, reproducible CLI construction policy."""

    def __init__(self, root):
        environment = host_shell_environment()
        super().__init__(root_dir=root, virtual_mode=_CLI_VIRTUAL_MODE, env=environment, inherit_env=False)
        self._cli_environment = dict(environment)
        self._cli_descriptor = {
            "type": "cli-local-shell-v1", "workspace": str(self.cwd.resolve()),
            "virtual_mode": self.virtual_mode, "timeout": self._default_timeout,
            "max_output_bytes": self._max_output_bytes,
            "environment_policy": list(DEEPAGENT_ENV_ALLOWLIST),
        }

    def delete(self, file_path):
        """Unlink symlinks; refuse to remove the filesystem root, the workspace, or its ancestors."""
        if self.virtual_mode:
            return super().delete(file_path)
        target = Path(file_path)
        if not target.is_absolute():
            target = self.cwd / target
        refused = DeleteResult(
            error=f"Error: refusing to delete '{file_path}' because it contains the workspace."
        )
        try:
            found = os.lstat(target)  # One inspection; fail closed if it is impossible.
            if not stat.S_ISDIR(found.st_mode):
                # Symlinks (even cyclic ones) and files: unlink never recurses, and it fails
                # safely if a directory replaced the entry meanwhile.
                target.unlink()
                return DeleteResult(path=file_path)
            # Compare identities: /tmp/x and /private/tmp/x name the same directory. Skip
            # ancestors that no longer exist.
            protected = set()
            for candidate in (Path("/"), self.cwd, *self.cwd.parents):
                try:
                    other = os.stat(candidate)
                except FileNotFoundError:
                    continue
                protected.add((other.st_dev, other.st_ino))
            if (found.st_dev, found.st_ino) in protected:
                return refused
            # The workspace may also sit inside the tree under another name (a bind mount).
            for directory, _subdirs, _files in os.walk(target, followlinks=False):
                inner = os.lstat(directory)
                if (inner.st_dev, inner.st_ino) in protected:
                    return refused
            # Remove the inspected path itself; rmtree refuses a top-level symlink.
            shutil.rmtree(target)
            return DeleteResult(path=file_path)
        except (OSError, RuntimeError) as exc:
            return DeleteResult(error=f"Error deleting '{file_path}': {exc}")

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
    root = resolve_workspace(workspace)
    if not _CLI_VIRTUAL_MODE:
        _require_file_tool_path(root)
    return _CLIWorkspaceBackend(root)


def _require_file_tool_path(root):
    """Reject roots that file tools would rewrite or route to a ChemGraph mount."""
    text = root.as_posix()
    try:
        unchanged = validate_path(text) == text
    except ValueError:
        unchanged = False
    if not unchanged:
        raise ValueError(f"Workspace path cannot be used unchanged by file tools: {text}")
    for prefix in (BUNDLED_SKILLS_PATH, USER_SKILLS_PATH, EXTERNAL_SKILLS_PATH, *_CLI_STATE_PREFIXES):
        if (text.rstrip("/") + "/").startswith(prefix):
            raise ValueError(f"Workspace {text} overlaps the reserved file-tool path {prefix}.")


def cli_backend_descriptor(backend):
    """Recognize only unchanged instances produced by the shared CLI factory."""
    if type(backend) is not _CLIWorkspaceBackend:
        return None
    descriptor = backend._cli_descriptor
    if (backend._env != backend._cli_environment
            or str(backend.cwd.resolve()) != descriptor["workspace"]
            or backend.virtual_mode is not descriptor["virtual_mode"]
            or backend._default_timeout != descriptor["timeout"]
            or backend._max_output_bytes != descriptor["max_output_bytes"]):
        return None
    return dict(descriptor)


def _real_path_cli(backend):
    """Recognize the CLI backend that gives file tools real host paths."""
    descriptor = cli_backend_descriptor(backend)
    return descriptor is not None and descriptor["virtual_mode"] is False


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


def real_path_workspace_note(backend):
    """Tell the model where a real-path CLI workspace is and how file-tool paths resolve."""
    if not _real_path_cli(backend):
        return ""
    root = backend.cwd.as_posix()
    return (
        f"\n\nThe workspace is `{root}`, the shell's starting directory. File tools and "
        "the shell use real host paths, except the ChemGraph mounts listed under \"Shell "
        "paths vs. virtual paths\" (`/chemgraph-.../`), which are not host paths. Always give "
        "file tools absolute paths; a relative file-tool path resolves from `/`. For `glob`, "
        f"pass the directory as `path` and a relative pattern, e.g. `glob(path=\"{root}\", "
        "pattern=\"**/*.py\")` for the project or `glob(path=\"/opt/data\", pattern=\"*.xyz\")` "
        "elsewhere. An omitted `path`, or `path=\"/\"`, searches the workspace plus every mount.\n"
    )


class _PrefixedStateBackend(StateBackend):
    """Keep routed checkpoint files under the full path the agent was shown.

    Workers and a checkpoint-only supervisor share the ``files`` channel, so keys must
    stay ``/large_tool_results/...``, not the key the composite route strips.
    """

    def __init__(self, prefix):
        super().__init__()
        self._prefix = prefix.rstrip("/")

    def _read_files(self):
        start = self._prefix + "/"
        return {key[len(self._prefix):]: value for key, value in super()._read_files().items()
                if key.startswith(start)}

    def _send_files_update(self, update):
        super()._send_files_update({self._prefix + key: value for key, value in update.items()})


def _normalize_backend(backend: BackendProtocol) -> BackendProtocol:
    """Mount a virtual local workspace at the path Deep Agent expects."""
    if _real_path_cli(backend):
        # Keep offloaded results off the host root, under their unchanged state keys.
        return CompositeBackend(default=backend, routes={
            prefix: _PrefixedStateBackend(prefix) for prefix in _CLI_STATE_PREFIXES
        })
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


def tool_reason_names(interrupt_on=_DEFAULT_INTERRUPT_POLICY):
    """Name tools that must state a reason, including under approval bypass."""
    names = set(resolve_workspace_interrupt_policy())
    if isinstance(interrupt_on, dict):
        names.update(name for name, policy in interrupt_on.items() if policy)
    return frozenset(names)


def prepare_workspace_runtime(
    *, backend=None, tools=(), tool_registry=None, skills=None,
    discover_skills=True, user_skills_dir=None, skill_dirs=None,
    interrupt_on=_DEFAULT_INTERRUPT_POLICY, scoped=False,
):
    """Return the backend, skill sources, middleware, and approval policy.

    ``scoped`` graphs keep files in checkpoint state and review every tool call
    through :class:`chemgraph.runtime.middleware.ScopedToolsMiddleware` instead
    of approval interrupts.
    """
    if scoped and (backend is not None or interrupt_on is not None):
        raise ValueError("Scoped graphs use checkpoint files and no approval interrupts.")
    effective_interrupt_on = resolve_workspace_interrupt_policy(tool_registry, interrupt_on)
    effective_backend = _normalize_backend(
        backend if backend is not None else StateBackend()
    )
    skill_sources = normalize_skill_sources(skills)
    effective_backend, sources, optional = prepare_skill_backend(
        effective_backend, skill_sources, discover_skills=discover_skills,
        user_skills_dir=user_skills_dir, skill_dirs=skill_dirs,
        # The real-path CLI workspace is the composite's default, not a /workspace route.
        local_workspace=local_skill_workspace(backend) if _real_path_cli(backend) else None,
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
    if scoped:
        from chemgraph.runtime.middleware import ScopedToolsMiddleware

        middleware.append(ScopedToolsMiddleware())
    # Innermost, so registry-loaded tools are extended and executed without the reason.
    middleware.append(ToolReasonMiddleware(tool_reason_names(interrupt_on)))
    return effective_backend, sources, middleware, effective_interrupt_on
