"""Unified loader for HPC-specific Parsl configurations.

This consolidates the ``load_parsl_config()`` function that was
previously duplicated across ``graspa_mcp_parsl.py`` and
``xanes_mcp_parsl.py``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def load_parsl_config(system_name: str, run_dir: str | None = None, **kwargs):
    """Dynamically import and return a Parsl ``Config`` for the given HPC system.

    Parameters
    ----------
    system_name : str
        Target system name.  Supported: ``"polaris"``, ``"aurora"``.
    run_dir : str, optional
        Parsl run directory.  Defaults to the current working directory.
    **kwargs
        Extra keyword arguments forwarded to the system-specific
        config factory (e.g. ``worker_init``).

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

    if system_name == "polaris":
        from chemgraph.hpc_configs.polaris_parsl import get_polaris_config

        return get_polaris_config(run_dir=run_dir, **kwargs)

    elif system_name == "aurora":
        from chemgraph.hpc_configs.aurora_parsl import get_aurora_config

        return get_aurora_config(run_dir=run_dir, **kwargs)

    else:
        raise ValueError(
            f"Unknown HPC system: '{system_name}'. "
            f"Supported systems: polaris, aurora"
        )
