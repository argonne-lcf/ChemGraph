"""Make academy-py's hosted-exchange listener resilient to truncated
long-poll responses.

Upstream ``academy.exchange.cloud.client.HttpExchangeTransport.listen``
raises ``aiohttp.ClientPayloadError`` (from an underlying
``TransferEncodingError: Not enough data to satisfy transfer length
header``) whenever the exchange server (or an intervening reverse proxy)
closes a long-poll response with an unfinished chunked-transfer body.
Once raised, the ``Runtime`` background task dies and the agent never
sees any subsequent inject / peer message -- the daemon stays up but is
functionally deaf.

The right long-term fix is in academy-py itself. Until that lands, this
module monkeypatches ``listen()`` to catch that specific exception,
sleep briefly, and reconnect. All other exceptions propagate unchanged
so real bugs are still surfaced.

Imported from ``chemgraph.academy.runtime.daemon`` before the Runtime
starts.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


def _patch() -> None:
    try:
        from academy.exchange.cloud import client as _client
        from aiohttp.client_exceptions import ClientPayloadError
    except ImportError:
        return

    if getattr(_client.HttpExchangeTransport.listen, "__cg_patched__", False):
        return

    _orig_listen = _client.HttpExchangeTransport.listen

    async def _listen_reconnecting(self, timeout=None):
        while True:
            try:
                async for message in _orig_listen(self, timeout=timeout):
                    yield message
                return
            except ClientPayloadError as exc:
                # Transient truncated long-poll -- reconnect. Sleep a
                # touch to avoid tight-looping if the server is broken.
                logger.warning(
                    "hosted-exchange listen() lost the long-poll stream "
                    "(%s); reconnecting.",
                    exc,
                )
                await asyncio.sleep(1.0)

    _listen_reconnecting.__cg_patched__ = True  # type: ignore[attr-defined]
    _client.HttpExchangeTransport.listen = _listen_reconnecting  # type: ignore[assignment]


_patch()
