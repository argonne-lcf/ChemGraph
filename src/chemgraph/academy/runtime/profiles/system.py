from __future__ import annotations

import json
import os
import re
from pathlib import Path

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from chemgraph.academy.runtime.profiles import resolve_builtin_system_profile


class SubmitDefaults(BaseModel):
    """Per-site PBS defaults used when the dashboard's launch buttons
    build a submit-mode --site argv on the fly.

    Previously hardcoded as ``_V1_SITE_DEFAULTS`` in
    ``swarm.dashboard.server``. Moved here so per-system
    profile JSON is the single source of truth for site-level PBS
    knobs, and so operators can override without editing Python.
    """

    model_config = ConfigDict(extra="forbid")

    queue: str
    walltime: str
    nodes: int = 1
    filesystems: str
    bundle_root_override: str | None = None


class SystemProfile(BaseModel):
    """Site/runtime paths for launching ChemGraph Academy on an HPC system."""

    model_config = ConfigDict(extra="forbid")

    name: str
    remote_host: str
    remote_root: str
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
    submit_defaults: SubmitDefaults


def load_system_profile(path_or_name: str | Path) -> SystemProfile:
    profile_path = resolve_builtin_system_profile(path_or_name)
    # Default ALCF_SSH_USER to ALCF_USER when unset. This separates the
    # *SSH login* (used in ``remote_host``) from the *path component*
    # (used everywhere else), which matters for accounts whose login
    # differs from their workspace dir name -- e.g. login ``jinchuli``
    # but workspace under ``/flare/.../jinchu/``. Most users have one
    # equal to the other and the default keeps their setup unchanged.
    env = os.environ.copy()
    if "ALCF_USER" in env and not env.get("ALCF_SSH_USER"):
        env["ALCF_SSH_USER"] = env["ALCF_USER"]
    text = _expand_with(profile_path.read_text(encoding="utf-8"), env)
    unresolved = sorted(set(re.findall(r"\$\{([^}]+)\}", text)))
    if unresolved:
        raise ValueError(
            f"System profile {profile_path} contains unresolved environment "
            f"variables: {', '.join(unresolved)}",
        )
    data = json.loads(text)
    return SystemProfile.model_validate(data)


def _expand_with(text: str, env: dict[str, str]) -> str:
    """``os.path.expandvars`` but reading from a caller-supplied env dict.

    The stdlib's ``expandvars`` always reads ``os.environ`` directly,
    which means ``ALCF_SSH_USER`` defaulted to ``ALCF_USER`` only by
    mutating the process environment. That'd leak the default into
    every subsequent caller. Substituting via regex keeps the change
    local.
    """
    return re.sub(
        r"\$\{([^}]+)\}",
        lambda m: env.get(m.group(1), m.group(0)),
        text,
    )
