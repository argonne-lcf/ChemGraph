"""Resolve operator-configured executables without invoking a shell."""

import os
from pathlib import Path
import shutil


def resolve_executable(name: str, env_var: str | None = None) -> str:
    """Resolve an environment override or a command on PATH at call time."""
    command = os.environ.get(env_var, name) if env_var else name
    resolved = shutil.which(os.path.expanduser(command))
    if resolved is None:
        hint = f"; set {env_var} to an executable path" if env_var else ""
        raise FileNotFoundError(f"Executable {command!r} is unavailable{hint}")
    return str(Path(resolved).resolve())
