"""Unified loader for HPC-specific Parsl configurations.

This consolidates the ``load_parsl_config()`` function that was
previously duplicated across ``graspa_mcp_parsl.py`` and
``xanes_mcp_parsl.py``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def resolve_worker_init(run_dir: str, fallback: str) -> str:
    """Build a Parsl ``worker_init`` shell snippet with layered precedence.

    Precedence (highest first):

    1. Environment variable ``CHEMGRAPH_WORKER_INIT`` -- if set and non-empty,
       used verbatim. Lets a user point Parsl workers at any env without
       editing code.
    2. Auto-detect the submitting process's Python env and emit an activate
       line for it (``VIRTUAL_ENV`` then ``CONDA_PREFIX``). The agent / MCP
       subprocess runs from this env, so workers should too.
    3. The system-specific *fallback* string passed by the caller (e.g.
       ``"module load conda; conda activate base"`` on Crux).

    The returned string is always prefixed with ``export TMPDIR=/tmp;
    cd {run_dir};`` so Parsl workers land in the same directory the
    submitter chose.
    """
    override = os.environ.get("CHEMGRAPH_WORKER_INIT", "").strip()
    if override:
        activate = override
    else:
        venv = os.environ.get("VIRTUAL_ENV", "").strip()
        conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
        if venv:
            activate = f"source {venv}/bin/activate"
        elif conda_prefix and conda_env:
            activate = (
                f"source {conda_prefix}/etc/profile.d/conda.sh && "
                f"conda activate {conda_env}"
            )
        else:
            activate = fallback
    return f"export TMPDIR=/tmp; cd {run_dir}; {activate}"


def load_parsl_config(system_name: str, run_dir: str | None = None, **kwargs):
    """Dynamically import and return a Parsl ``Config`` for the given HPC system.

    Parameters
    ----------
    system_name : str
        Target system name.  Supported: ``"local"``, ``"polaris"``,
        ``"aurora"``, ``"crux"``.
    run_dir : str, optional
        Parsl run directory.  Defaults to the current working directory.
    **kwargs
        Extra keyword arguments forwarded to the system-specific
        config factory (e.g. ``worker_init``, ``max_workers``).

    Returns
    -------
    parsl.config.Config
        A ready-to-use Parsl configuration object.

    Raises
    ------
    ValueError
        If *system_name* is not recognised.
    """
    system_name = system_name.lower().strip()
    if run_dir is None:
        run_dir = os.getcwd()

    logger.info("Loading Parsl config for system: %s", system_name)

    if system_name == "local":
        from chemgraph.hpc_configs.local_parsl import get_local_config

        return get_local_config(run_dir=run_dir, **kwargs)

    elif system_name == "polaris":
        from chemgraph.hpc_configs.polaris_parsl import get_polaris_config

        return get_polaris_config(run_dir=run_dir, **kwargs)

    elif system_name == "aurora":
        from chemgraph.hpc_configs.aurora_parsl import get_aurora_config

        return get_aurora_config(run_dir=run_dir, **kwargs)

    elif system_name == "crux":
        from chemgraph.hpc_configs.crux_parsl import get_crux_config

        return get_crux_config(run_dir=run_dir, **kwargs)

    else:
        raise ValueError(
            f"Unknown HPC system: '{system_name}'. "
            f"Supported systems: local, polaris, aurora, crux"
        )
