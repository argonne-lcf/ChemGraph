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
    ``"crux"``, ``"local"``).
``GLOBUS_COMPUTE_AMQP_PORT``
    Override the Globus Compute result-streaming port when it is not set in
    ``config.toml`` or passed explicitly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from chemgraph.execution.base import ExecutionBackend

if TYPE_CHECKING:
    from chemgraph.execution.globus_transfer import GlobusTransferManager

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

    # Globus Compute: fall back to GLOBUS_COMPUTE_ENDPOINT_ID env var
    if resolved_backend == "globus_compute" and not merged_kwargs.get("endpoint_id"):
        env_id = os.getenv("GLOBUS_COMPUTE_ENDPOINT_ID")
        if env_id:
            merged_kwargs["endpoint_id"] = env_id
    if resolved_backend == "globus_compute" and not merged_kwargs.get("amqp_port"):
        env_port = os.getenv("GLOBUS_COMPUTE_AMQP_PORT")
        if env_port:
            merged_kwargs["amqp_port"] = env_port

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
            SYSTEM_CONFIG_REGISTRY,
            EnsembleLauncherBackend,
            get_launcher_config,
        )

        backend = EnsembleLauncherBackend()

        if merged_kwargs.get("client_only", False):
            # Client-only mode: pass through as-is (needs checkpoint_dir).
            pass
        else:
            # Managed mode: start orchestrator locally.
            assert resolved_system in SYSTEM_CONFIG_REGISTRY, (
                f"Unknown system {resolved_system}: "
                f"only know {list(SYSTEM_CONFIG_REGISTRY.keys())}"
            )
            # System-appropriate MPI flavour: multi-node HPC systems need
            # mpich/hydra so child-spec JSON actually lands on remote /tmp;
            # "test" only works for single-host runs.
            launcher_cfg_kwargs = dict(backend_cfg)
            if "mpi_flavour" not in launcher_cfg_kwargs:
                _system_mpi_flavour = {
                    "aurora": "mpich",
                    "polaris": "mpich",
                    "crux": "mpich",
                    "local": "test",
                }
                launcher_cfg_kwargs["mpi_flavour"] = _system_mpi_flavour.get(
                    resolved_system, "mpich"
                )
            merged_kwargs = {
                "system_config": SYSTEM_CONFIG_REGISTRY[resolved_system],
                "launcher_config": get_launcher_config(**launcher_cfg_kwargs),
            }

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


def get_transfer_manager(
    config_path: Optional[str] = None,
    system: Optional[str] = None,
    *,
    default_manager: GlobusTransferManager | None = None,
    **kwargs: Any,
):
    """Create a :class:`GlobusTransferManager` from config, or ``None``.

    Reads the ``[execution.globus_transfer]`` section from
    ``config.toml``. An explicit destination ID takes priority over *system*,
    which takes priority over the configured destination. ``default_manager``
    supplies existing server settings instead of reading config/environment;
    overrides create an independent manager. Returns ``None`` when required
    settings are missing. Authentication remains lazy.

    Environment variable fallbacks
    ------------------------------
    ``GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID``
    ``GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID``
    ``GLOBUS_TRANSFER_DESTINATION_BASE_PATH``
    ``GLOBUS_TRANSFER_DESTINATION_COMPUTE_BASE_PATH``
    """
    from chemgraph.hpc_configs import (
        get_facility_transfer_profile,
        list_facility_transfer_profiles,
    )

    if default_manager is not None:
        defaults = {
            key: getattr(default_manager, key)
            for key in (
                "source_endpoint_id", "destination_endpoint_id",
                "destination_base_path", "destination_compute_base_path",
                "source_base_path", "allow_interactive_auth",
            )
        }
        defaults["client_id"] = default_manager._client_id
        default_system = default_manager.system
    else:
        cfg = _load_execution_config(config_path)
        defaults = dict(cfg.get("globus_transfer", {}))
        for key in (
            "source_endpoint_id", "destination_endpoint_id",
            "destination_base_path", "destination_compute_base_path",
        ):
            if not defaults.get(key):
                defaults[key] = os.getenv(f"GLOBUS_TRANSFER_{key.upper()}")
        default_system = os.getenv("COMPUTE_SYSTEM") or cfg.get("system")

    merged = {**defaults, **{k: v for k, v in kwargs.items() if v is not None}}
    resolved_system = system or default_system
    resolved_system_name = (
        resolved_system.strip().lower()
        if isinstance(resolved_system, str) and resolved_system.strip()
        else None
    )
    profile = None
    if resolved_system_name is not None:
        profile = get_facility_transfer_profile(resolved_system_name)

    if system is not None and not kwargs.get("destination_endpoint_id"):
        if profile is None or profile.has_placeholder_collection_id:
            raise ValueError(
                f"No bundled Transfer collection for system {system!r}. "
                "Pass destination_endpoint_id explicitly or select a system "
                "from list_transfer_facilities."
            )
        merged["destination_endpoint_id"] = profile.collection_id
    elif not merged.get("destination_endpoint_id") and profile is not None:
        if not profile.has_placeholder_collection_id:
            merged["destination_endpoint_id"] = profile.collection_id

    destination = merged.get("destination_endpoint_id")
    default_profile = (
        get_facility_transfer_profile(default_system) if default_system else None
    )
    default_destination = defaults.get("destination_endpoint_id") or (
        default_profile.collection_id if default_profile else None
    )
    if default_destination and destination != default_destination:
        # A compute path for a previous destination must not follow an override.
        merged["destination_compute_base_path"] = kwargs.get(
            "destination_compute_base_path"
        )
        if system is None:
            resolved_system_name = None

    # Keep the selected system when several systems share a collection.
    # Otherwise derive path mapping from the actual destination UUID.
    if profile is None or profile.collection_id != destination:
        profile = next(
            (
                p for p in list_facility_transfer_profiles()
                if not p.has_placeholder_collection_id
                and p.collection_id == destination
            ),
            None,
        )
    if profile is not None:
        resolved_system_name = profile.system
    if (
        profile is not None
        and merged.get("destination_base_path")
        and not merged.get("destination_compute_base_path")
    ):
        merged["destination_compute_base_path"] = profile.compute_path(
            merged["destination_base_path"]
        )

    required = (
        "source_endpoint_id",
        "destination_endpoint_id",
        "destination_base_path",
    )
    if not all(merged.get(k) for k in required):
        logger.debug(
            "Globus Transfer not configured (missing %s). "
            "Supply the missing settings before transferring files.",
            [k for k in required if not merged.get(k)],
        )
        return None

    from chemgraph.execution.globus_transfer import GlobusTransferManager

    manager = GlobusTransferManager(
        source_endpoint_id=merged["source_endpoint_id"],
        destination_endpoint_id=merged["destination_endpoint_id"],
        destination_base_path=merged["destination_base_path"],
        destination_compute_base_path=merged.get(
            "destination_compute_base_path"
        ),
        source_base_path=merged.get("source_base_path"),
        client_id=merged.get("client_id"),
        allow_interactive_auth=bool(merged.get("allow_interactive_auth", True)),
        system=resolved_system_name,
    )
    if default_manager is not None and manager._client_id == default_manager._client_id:
        manager._transfer_client = default_manager._transfer_client
    logger.info(
        "GlobusTransferManager created: %s -> %s",
        merged["source_endpoint_id"],
        merged["destination_endpoint_id"],
    )
    return manager
