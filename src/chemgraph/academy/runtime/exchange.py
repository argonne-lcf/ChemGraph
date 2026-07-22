"""Build the Academy exchange factory matching a daemon config."""

from __future__ import annotations

from typing import Any

from chemgraph.academy.core.campaign import ChemGraphDaemonConfig


SUPPORTED_EXCHANGE_TYPES: tuple[str, ...] = ('redis', 'local', 'hybrid', 'http')
"""All exchange types this module knows how to build.

Used by the CLI to enforce ``--exchange-type`` choices and by tests
to assert the supported set stays in sync with the dispatch table
below.
"""


def exchange_uses_redis(exchange_type: str) -> bool:
    """Return True when the exchange type requires a running Redis server.

    The compute launcher uses this to decide whether to start a Redis
    subprocess on rank 0. Exchanges that don't talk to Redis (``local``,
    ``http``) don't need one and skipping the subprocess avoids a port-
    binding failure when Redis isn't installed on the compute node.
    """
    return exchange_type in {'redis', 'hybrid'}


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

    if exchange_type == 'http':
        # Academy's HTTP exchange. Passing url=None selects the
        # hosted default (https://exchange.academy-agents.org/v1)
        # with Globus Auth. The bearer token is read from
        # $XDG_DATA_HOME/academy/storage.db -- the user (or the
        # launcher's env-prep step) must have logged in already
        # via the device flow before any agent constructs this.
        # On Aurora compute nodes, http_proxy / https_proxy must be
        # set to the ALCF proxy (http://proxy.alcf.anl.gov:3128)
        # before the daemon starts; otherwise the first PUT will
        # hang at the connection-timeout boundary.
        from academy.exchange.cloud import HttpExchangeFactory

        kwargs: dict[str, Any] = {}
        if config.http_exchange_url:
            kwargs['url'] = config.http_exchange_url
        return HttpExchangeFactory(**kwargs)

    raise ValueError(
        f"Unsupported exchange type {exchange_type!r}; expected one of "
        f"{sorted(SUPPORTED_EXCHANGE_TYPES)}.",
    )
