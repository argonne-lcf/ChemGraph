from __future__ import annotations

import os
from pathlib import Path


BUILTIN_SYSTEM_PROFILES = {
    "aurora": "aurora.template.json",
    "polaris": "polaris.template.json",
    "crux": "crux.template.json",
}

# Same env var as chemgraph.academy.campaigns -- points at the repo's
# ``examples/`` directory. When set, profiles resolve under
# ``$CHEMGRAPH_EXAMPLES_ROOT/profiles/``.
EXAMPLES_ROOT_ENV = "CHEMGRAPH_EXAMPLES_ROOT"


def _examples_profiles_root() -> Path:
    """Return the directory that holds the shipped system profiles.

    Order of resolution:
      1. ``$CHEMGRAPH_EXAMPLES_ROOT/profiles`` if the env var is set.
      2. ``<repo_root>/examples/profiles`` walking up from this file
         (works for both editable installs and running against the
         checkout directly).
    """
    env_root = os.environ.get(EXAMPLES_ROOT_ENV)
    if env_root:
        return Path(env_root) / "profiles"
    # This file is src/chemgraph/academy/runtime/profiles/__init__.py.
    # parents[5] == repo root (../../../../../../).
    return Path(__file__).resolve().parents[5] / "examples" / "profiles"


def resolve_builtin_system_profile(path_or_name: str | Path) -> Path:
    value = str(path_or_name)
    path = Path(value)
    if path.exists():
        return path.resolve()
    relative = BUILTIN_SYSTEM_PROFILES.get(value)
    if relative is None:
        return path
    # ponytail: profile JSON now lives under ``<repo>/examples/profiles/``
    # instead of inside the package. Site-specific config (ports,
    # usernames, allocation paths) doesn't belong shipped as importable
    # data.
    return (_examples_profiles_root() / relative).resolve()


def list_builtin_system_profiles() -> list[str]:
    return sorted(BUILTIN_SYSTEM_PROFILES)


from chemgraph.academy.runtime.profiles.system import SystemProfile  # noqa: E402
from chemgraph.academy.runtime.profiles.system import load_system_profile  # noqa: E402


__all__ = [
    "BUILTIN_SYSTEM_PROFILES",
    "SystemProfile",
    "list_builtin_system_profiles",
    "load_system_profile",
    "resolve_builtin_system_profile",
]
