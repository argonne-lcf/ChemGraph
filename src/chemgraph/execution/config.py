"""Execution backend configuration and factory.

Reads the ``[execution]`` section from ``config.toml`` (or env-var
overrides) and returns an initialised :class:`ExecutionBackend` instance.

Environment variables
---------------------
``CHEMGRAPH_EXECUTION_BACKEND``
    Override the backend name (``"parsl"``, ``"ensemble_launcher"``,
    ``"globus_compute"``, ``"local"``).
``COMPUTE_SYSTEM``
    Override the target HPC system (``"polaris"``, ``"aurora"``,
    ``"local"``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from chemgraph.execution.base import ExecutionBackend

logger = logging.getLogger(__name__)

# Supported backend names (keep in sync with the ``elif`` chain below)
SUPPORTED_BACKENDS = ("parsl", "ensemble_launcher", "globus_compute", "local")


def _load_execution_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """Read the ``[execution]`` table from ``config.toml``.

    Returns an empty dict if the section is missing or the file is not
    found, so callers always get sensible defaults.
    """
    if config_path is None:
        # Walk upward from CWD to find config.toml (same heuristic the
        # rest of ChemGraph uses).
        candidate = Path.cwd() / "config.toml"
        if candidate.is_file():
            config_path = str(candidate)
        else:
            # Try the repo root (two levels up from this file).
            repo_root = Path(__file__).resolve().parents[3]
            candidate = repo_root / "config.toml"
            if candidate.is_file():
                config_path = str(candidate)

    if config_path is None:
        return {}

    try:
        import toml

        full_config = toml.load(config_path)
        return full_config.get("execution", {})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read [execution] from %s: %s", config_path, exc)
        return {}


def get_backend(
    config_path: Optional[str] = None,
    backend_name: Optional[str] = None,
    system: Optional[str] = None,
    **kwargs: Any,
) -> ExecutionBackend:
    """Create and initialise an :class:`ExecutionBackend`.

    Resolution order for ``backend_name``:

    1. Explicit ``backend_name`` argument
    2. ``CHEMGRAPH_EXECUTION_BACKEND`` environment variable
    3. ``config.toml`` ``[execution] backend`` key
    4. ``"local"`` (safe fallback)

    Resolution order for ``system``:

    1. Explicit ``system`` argument
    2. ``COMPUTE_SYSTEM`` environment variable
    3. ``config.toml`` ``[execution] system`` key
    4. ``"local"``

    Parameters
    ----------
    config_path : str, optional
        Path to ``config.toml``.  Auto-detected when omitted.
    backend_name : str, optional
        Force a specific backend.
    system : str, optional
        Target HPC system name.
    **kwargs
        Extra keyword arguments forwarded to
        :meth:`ExecutionBackend.initialize`.

    Returns
    -------
    ExecutionBackend
        A ready-to-use backend instance.
    """
    cfg = _load_execution_config(config_path)

    # -- resolve backend name -------------------------------------------------
    resolved_backend = (
        backend_name
        or os.getenv("CHEMGRAPH_EXECUTION_BACKEND")
        or cfg.get("backend", "local")
    )
    resolved_backend = resolved_backend.lower().strip()

    if resolved_backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown execution backend '{resolved_backend}'. "
            f"Supported: {', '.join(SUPPORTED_BACKENDS)}"
        )

    # -- resolve system -------------------------------------------------------
    resolved_system = (
        system or os.getenv("COMPUTE_SYSTEM") or cfg.get("system", "local")
    )

    # -- merge backend-specific config ----------------------------------------
    backend_cfg = cfg.get(resolved_backend, {})
    merged_kwargs = {**backend_cfg, **kwargs}

    # -- instantiate ----------------------------------------------------------
    logger.info(
        "Creating execution backend '%s' for system '%s'",
        resolved_backend,
        resolved_system,
    )

    if resolved_backend == "parsl":
        from chemgraph.execution.parsl_backend import ParslBackend

        backend = ParslBackend()

    elif resolved_backend == "ensemble_launcher":
        from chemgraph.execution.ensemble_launcher_backend import (
            EnsembleLauncherBackend,
        )

        backend = EnsembleLauncherBackend()

    elif resolved_backend == "globus_compute":
        from chemgraph.execution.globus_compute_backend import (
            GlobusComputeBackend,
        )

        backend = GlobusComputeBackend()

    elif resolved_backend == "local":
        from chemgraph.execution.local_backend import LocalBackend

        backend = LocalBackend()

    else:
        # Should be unreachable thanks to the validation above.
        raise ValueError(f"Unsupported backend: {resolved_backend}")

    backend.initialize(system=resolved_system, **merged_kwargs)
    return backend
