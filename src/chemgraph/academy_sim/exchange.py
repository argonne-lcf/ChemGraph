"""Academy exchange factory construction for academy_sim."""

from __future__ import annotations

from typing import Any

from chemgraph.academy_sim.config import ExchangeConfig


def build_exchange_factory(config: ExchangeConfig) -> Any:
    """Return an Academy exchange factory for an academy_sim exchange config."""

    if config.type == 'local':
        from academy.exchange.local import LocalExchangeFactory

        return LocalExchangeFactory()

    if config.type == 'redis':
        from academy.exchange.redis import RedisExchangeFactory

        return RedisExchangeFactory(
            hostname=config.redis_host,
            port=config.redis_port,
        )

    if config.type == 'hybrid':
        from academy.exchange.hybrid import HybridExchangeFactory

        return HybridExchangeFactory(
            redis_host=config.redis_host,
            redis_port=config.redis_port,
            namespace=config.namespace,
        )

    if config.type == 'http':
        try:
            from academy.exchange.cloud import HttpExchangeFactory
        except ImportError:
            from academy.exchange import HttpExchangeFactory

        if config.url:
            try:
                return HttpExchangeFactory(url=config.url)
            except TypeError:
                return HttpExchangeFactory(config.url)
        return HttpExchangeFactory()

    raise ValueError(f'unsupported exchange type {config.type!r}')
