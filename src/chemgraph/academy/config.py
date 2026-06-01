"""Bridge between ChemGraph config.toml and Academy Manager/Exchange/Launcher.

Reads the ``[academy]`` section from ``config.toml`` and builds the
corresponding Academy objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import toml

logger = logging.getLogger(__name__)

# Exchange and launcher types supported by this bridge.
ExchangeType = Literal["local", "redis", "hybrid"]
LauncherType = Literal["thread", "process", "parsl", "globus_compute"]


@dataclass
class AcademyConfig:
    """Parsed ``[academy]`` configuration section.

    Attributes
    ----------
    exchange : ExchangeType
        Message exchange backend (default ``"local"``).
    launcher : LauncherType
        Agent deployment mechanism (default ``"thread"``).
    num_agents : int
        Number of worker agents to spawn (default ``1``).
    redis_hostname : str
        Redis host when ``exchange="redis"`` (default ``"localhost"``).
    redis_port : int
        Redis port (default ``6379``).
    parsl_system : str
        HPC system name for Parsl config (default ``"local"``).
    globus_endpoint_id : str
        Globus Compute endpoint UUID.
    max_concurrency : int
        Max concurrent LLM calls per provider (default ``50``).
    log_dir : str or None
        Base log directory for agent output.
    extra : dict
        Any additional keys from the config section.
    """

    exchange: ExchangeType = "local"
    launcher: LauncherType = "thread"
    num_agents: int = 1
    redis_hostname: str = "localhost"
    redis_port: int = 6379
    parsl_system: str = "local"
    globus_endpoint_id: str = ""
    max_concurrency: int = 50
    log_dir: Optional[str] = None
    extra: dict = field(default_factory=dict)


def load_academy_config(config_path: str = "config.toml") -> AcademyConfig:
    """Load the ``[academy]`` section from a TOML config file.

    Missing keys are filled with defaults.  Unknown keys are stored
    in ``extra``.
    """
    try:
        data = toml.load(config_path)
    except FileNotFoundError:
        logger.warning("Config file %s not found, using defaults", config_path)
        return AcademyConfig()

    section = data.get("academy", {})

    known_keys = {f.name for f in AcademyConfig.__dataclass_fields__.values()}
    known = {k: v for k, v in section.items() if k in known_keys}
    extra = {k: v for k, v in section.items() if k not in known_keys}

    return AcademyConfig(**known, extra=extra)


def _build_exchange_factory(cfg: AcademyConfig) -> Any:
    """Create the Academy ExchangeFactory matching the config."""
    if cfg.exchange == "local":
        from academy.exchange import LocalExchangeFactory

        return LocalExchangeFactory()

    if cfg.exchange == "redis":
        from academy.exchange import RedisExchangeFactory

        return RedisExchangeFactory(
            hostname=cfg.redis_hostname,
            port=cfg.redis_port,
        )

    if cfg.exchange == "hybrid":
        from academy.exchange import HybridExchangeFactory

        return HybridExchangeFactory()

    raise ValueError(f"Unsupported exchange type: {cfg.exchange}")


def _build_executor(cfg: AcademyConfig) -> Any:
    """Create the executor matching the configured launcher type."""
    if cfg.launcher == "thread":
        from concurrent.futures import ThreadPoolExecutor

        return ThreadPoolExecutor(max_workers=cfg.num_agents)

    if cfg.launcher == "process":
        from concurrent.futures import ProcessPoolExecutor

        return ProcessPoolExecutor(max_workers=cfg.num_agents)

    if cfg.launcher == "parsl":
        try:
            from academy.executor import ParslExecutor
        except ImportError as exc:
            raise ImportError(
                "Parsl launcher requires: pip install chemgraphagent[academy,parsl]"
            ) from exc
        return ParslExecutor()

    if cfg.launcher == "globus_compute":
        try:
            from academy.executor import GlobusComputeExecutor
        except ImportError as exc:
            raise ImportError(
                "Globus Compute launcher requires: "
                "pip install chemgraphagent[academy,globus_compute]"
            ) from exc
        return GlobusComputeExecutor(endpoint_id=cfg.globus_endpoint_id)

    raise ValueError(f"Unsupported launcher type: {cfg.launcher}")


async def build_manager(
    cfg: AcademyConfig | None = None,
    config_path: str = "config.toml",
) -> Any:
    """Build an Academy Manager from ChemGraph config.

    Returns an async context manager. Usage::

        async with await build_manager() as manager:
            handle = await manager.launch(ScreeningAgent, ...)
            result = await handle.screen_molecule("CCO", "optimize")

    Parameters
    ----------
    cfg : AcademyConfig, optional
        Pre-loaded config.  If ``None``, loads from *config_path*.
    config_path : str
        Path to config.toml (used only when *cfg* is ``None``).

    Returns
    -------
    Manager
        An Academy Manager ready for ``async with``.
    """
    from academy.manager import Manager

    if cfg is None:
        cfg = load_academy_config(config_path)

    factory = _build_exchange_factory(cfg)
    executor = _build_executor(cfg)

    return await Manager.from_exchange_factory(
        factory=factory,
        executors=executor,
    )
