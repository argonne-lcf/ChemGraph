"""Build the Academy exchange factory matching a daemon config."""

from __future__ import annotations

from typing import Any

from chemgraph.academy.core.campaign import ChemGraphDaemonConfig


def build_exchange_factory(config: ChemGraphDaemonConfig) -> Any:
    """Return the Academy exchange factory matching ``config.exchange_type``."""
    exchange_type = config.exchange_type

    if exchange_type == 'redis':
        from academy.exchange.redis import RedisExchangeFactory

        return RedisExchangeFactory(
            hostname=config.redis_host,
            port=config.redis_port,
        )

    if exchange_type == 'local':
        from academy.exchange.local import LocalExchangeFactory

        return LocalExchangeFactory()

    if exchange_type == 'hybrid':
        from academy.exchange.hybrid import HybridExchangeFactory

        return HybridExchangeFactory(
            redis_host=config.redis_host,
            redis_port=config.redis_port,
            namespace=config.redis_namespace,
        )

    raise ValueError(
        f"Unsupported exchange type {exchange_type!r}; expected one of "
        "'redis', 'local', 'hybrid'.",
    )
