from __future__ import annotations

from importlib import resources
from pathlib import Path


BUILTIN_SYSTEM_PROFILES = {
    "aurora": "aurora.template.json",
    "polaris": "polaris.template.json",
    "crux": "crux.template.json",
}


def resolve_builtin_system_profile(path_or_name: str | Path) -> Path:
    value = str(path_or_name)
    path = Path(value)
    if path.exists():
        return path.resolve()
    relative = BUILTIN_SYSTEM_PROFILES.get(value)
    if relative is None:
        return path
    return Path(str(resources.files(__package__).joinpath(relative)))


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
