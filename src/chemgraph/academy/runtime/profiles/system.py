from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from chemgraph.academy.runtime.profiles import resolve_builtin_system_profile


class SystemProfile(BaseModel):
    """Site/runtime paths for launching ChemGraph Academy on an HPC system."""

    model_config = ConfigDict(extra="forbid")

    name: str
    remote_host: str
    remote_root: str
    academy_repo_root: str
    repo_root: str
    run_root: str
    relay_host_file: str
    relay_port: int
    venv_python: str
    redis_bin_dir: str
    redis_port: int
    redis_bind: str
    redis_protected_mode: str
    mpiexec: str
    pythonpath_entries: list[str]
    path_entries: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    unset_env: list[str] = Field(default_factory=list)
    no_proxy: str


def load_system_profile(path_or_name: str | Path) -> SystemProfile:
    profile_path = resolve_builtin_system_profile(path_or_name)
    text = os.path.expandvars(profile_path.read_text(encoding="utf-8"))
    unresolved = sorted(set(re.findall(r"\$\{([^}]+)\}", text)))
    if unresolved:
        raise ValueError(
            f"System profile {profile_path} contains unresolved environment "
            f"variables: {', '.join(unresolved)}",
        )
    data = json.loads(text)
    return SystemProfile.model_validate(data)
