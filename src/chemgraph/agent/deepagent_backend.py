"""Host-shell backend construction shared by the CLI and the Streamlit UI.

The experimental Deep Agent runs shell commands on the host and edits
files under a workspace directory.  Both front ends build the same
``LocalShellBackend`` (virtual-mode paths, a minimal environment
allowlist, no inherited environment); the CLI adds an interactive
confirmation prompt around this helper and the UI an explicit
acknowledgment checkbox.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

#: Environment variables forwarded to Deep Agent shell commands.  Everything
#: else (API keys in particular) is withheld from the subprocess.
DEEPAGENT_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "TMPDIR",
    "CHEMGRAPH_LOG_DIR",
)


def resolve_workspace(workspace: Optional[str]) -> Path:
    """Return the absolute Deep Agent workspace directory.

    Parameters
    ----------
    workspace : str, optional
        Directory path; ``None``/empty selects the current working directory.

    Raises
    ------
    ValueError
        When the path is not an existing directory.
    """
    root = Path(workspace or Path.cwd()).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Deep Agent workspace is not a directory: {root}")
    return root


def host_shell_environment() -> dict[str, str]:
    """Return the allowlisted subset of the current environment."""
    return {
        name: os.environ[name]
        for name in DEEPAGENT_ENV_ALLOWLIST
        if name in os.environ
    }


def update_shell_environment(backend: Any, **values: Optional[str]) -> bool:
    """Change allowlisted variables seen by *backend*'s later shell commands.

    ``LocalShellBackend`` snapshots its environment when it is built.  Front
    ends that move ``CHEMGRAPH_LOG_DIR`` between turns (the Streamlit UI
    gives every query its own artifact directory) call this so commands
    run by the agent write to the current turn instead of an earlier one.

    Parameters
    ----------
    backend : Any
        Backend returned by :func:`create_host_shell_backend`.
    **values : str or None
        Allowlisted variable names; ``None`` removes the variable.

    Returns
    -------
    bool
        ``False`` when *backend* has no mutable shell environment (e.g. not
        a host-shell backend), ``True`` otherwise.

    Raises
    ------
    ValueError
        For a name outside :data:`DEEPAGENT_ENV_ALLOWLIST`.
    """
    for name in values:
        if name not in DEEPAGENT_ENV_ALLOWLIST:
            raise ValueError(f"{name} is not an allowlisted shell variable.")
    env = getattr(backend, "_env", None)
    if not isinstance(env, dict):
        return False
    for name, value in values.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = str(value)
    return True


def create_host_shell_backend(workspace: Optional[str]) -> Any:
    """Create the development-only host-shell backend for *workspace*.

    Parameters
    ----------
    workspace : str, optional
        Workspace directory (see :func:`resolve_workspace`).

    Returns
    -------
    deepagents.backends.LocalShellBackend
        Backend rooted at the workspace with virtual-mode paths and the
        allowlisted environment only.
    """
    from deepagents.backends import LocalShellBackend

    root = resolve_workspace(workspace)
    return LocalShellBackend(
        root_dir=root,
        virtual_mode=True,
        env=host_shell_environment(),
        inherit_env=False,
    )


__all__ = [
    "DEEPAGENT_ENV_ALLOWLIST",
    "create_host_shell_backend",
    "host_shell_environment",
    "resolve_workspace",
    "update_shell_environment",
]
