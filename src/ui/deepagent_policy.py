"""Server-side policy for the experimental Deep Agent in the web UI.

The CLI's Deep Agent runs in the operator's own terminal.  In the web UI the
person typing into the browser need not be the person who started the
server (``--server.address 0.0.0.0``, Docker, Kubernetes), yet the Deep
Agent can run shell commands as the server's user once *they* approve them.
Two server-side controls therefore apply, neither of which a browser user
can change:

* The ``deep_agent`` workflow is **off** unless the operator enables it
  (``chemgraph ui --enable-deep-agent`` or ``CHEMGRAPH_UI_DEEPAGENT=1``).
* Workspace and extra skill directories entered in the UI (or in the raw
  TOML editor) must resolve inside operator-approved roots
  (``--deep-agent-root`` / ``CHEMGRAPH_UI_DEEPAGENT_ROOTS``, defaulting to
  the directory the UI was launched from).  Paths are resolved with
  :func:`os.path.realpath` (symlinks and ``..`` collapsed) before the
  containment check, and nothing touches the filesystem for a path that
  fails it.

Streamlit-free so it can be unit-tested.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence

#: Set to ``1``/``true``/``yes``/``on`` to offer the deep_agent workflow.
ENABLE_ENV = "CHEMGRAPH_UI_DEEPAGENT"
#: ``os.pathsep``-separated directories that UI-entered paths must stay in.
ROOTS_ENV = "CHEMGRAPH_UI_DEEPAGENT_ROOTS"

_TRUE = frozenset({"1", "true", "yes", "on"})

DISABLED_MESSAGE = (
    "The deep_agent workflow is disabled on this server. It gives the model "
    "a shell on the host, so the operator must enable it when starting the "
    "UI: `chemgraph ui --enable-deep-agent` (or set "
    f"`{ENABLE_ENV}=1`)."
)


def deep_agent_enabled(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether the operator enabled the Deep Agent for this server."""
    env = os.environ if environ is None else environ
    return env.get(ENABLE_ENV, "").strip().lower() in _TRUE


def allowed_roots(environ: Optional[Mapping[str, str]] = None) -> tuple[str, ...]:
    """Return the resolved directories UI-supplied paths must stay within.

    Defaults to the server's working directory (where the UI was launched).
    """
    env = os.environ if environ is None else environ
    raw = [item for item in env.get(ROOTS_ENV, "").split(os.pathsep) if item.strip()]
    roots = raw or [os.getcwd()]
    resolved: list[str] = []
    for item in roots:
        root = os.path.realpath(os.path.expanduser(item.strip()))
        if root not in resolved:
            resolved.append(root)
    return tuple(resolved)


def confine_directory(
    value: str,
    *,
    roots: Optional[Sequence[str]] = None,
    label: str = "Directory",
) -> str:
    """Resolve a UI-supplied directory and require it inside an allowed root.

    Relative paths are anchored at the first root (the launch directory by
    default), matching the CLI's "relative to the invocation directory".

    Parameters
    ----------
    value : str
        Path text from the browser or the saved config.
    roots : sequence of str, optional
        Allowed roots; defaults to :func:`allowed_roots`.
    label : str, optional
        Name used in error messages.

    Returns
    -------
    str
        The resolved absolute directory.

    Raises
    ------
    ValueError
        When the path escapes every allowed root or is not a directory.
    """
    allowed = tuple(roots) if roots is not None else allowed_roots()
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is empty.")
    if "\x00" in text:
        raise ValueError(f"{label} contains a NUL character.")
    base = allowed[0] if allowed else os.getcwd()
    candidate = os.path.realpath(os.path.join(base, os.path.expanduser(text)))
    for root in allowed:
        # Containment is decided on the fully resolved path, before the
        # filesystem is consulted about it.
        if candidate == root:
            safe = root
        elif candidate.startswith(root.rstrip(os.sep) + os.sep):
            safe = candidate
        else:
            continue
        if not os.path.isdir(safe):
            raise ValueError(f"{label} does not exist: {safe}")
        return safe
    raise ValueError(
        f"{label} {text!r} resolves outside the directories this server "
        f"allows ({', '.join(allowed)}). The operator can widen them with "
        f"--deep-agent-root or {ROOTS_ENV}."
    )


__all__ = [
    "DISABLED_MESSAGE",
    "ENABLE_ENV",
    "ROOTS_ENV",
    "allowed_roots",
    "confine_directory",
    "deep_agent_enabled",
]
