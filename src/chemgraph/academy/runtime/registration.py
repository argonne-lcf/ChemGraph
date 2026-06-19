"""Peer-agent discovery against the Academy exchange.

The runtime used to coordinate per-rank registrations via a
shared-filesystem JSON file (``<run_dir>/academy_registrations.json``):
rank 0 registered every agent on the campaign and wrote the resulting
``AgentRegistration`` objects to disk; other ranks polled the file. That
mechanism cannot span machines with separate filesystems, which blocks
the federated ``spawn-site`` flow.

The replacement uses the exchange itself as the lookup service. Each
rank registers ONLY its own local agent (returning an
``AgentRegistration`` that goes straight into ``Runtime``), and looks
up the ``AgentId`` of every cross-site peer by polling
``transport.discover(ChemGraphLogicalAgent)`` until the expected names
appear. There is no rank-0 special-casing for registration anymore: any
rank can come up in any order, on any host, as long as eventually
every peer's mailbox is reachable through the exchange before
``startup_timeout_s`` elapses.

Name collisions: ``discover()`` returns every agent of the given class
registered against the exchange, across all operators and campaigns. To
keep federated campaigns from accidentally seeing each other's
agents, operators should prefix agent names with a campaign-unique
run-id (e.g. ``demo-001-coordinator-agent``). Auto-prefixing is a
future ergonomic improvement; for now it's an operator-runbook
convention.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from typing import Any

from academy.exchange.transport import ExchangeTransportT
from academy.identifier import AgentId

logger = logging.getLogger(__name__)


async def discover_peer_agent_ids(
    transport: ExchangeTransportT,
    peer_names: Iterable[str],
    *,
    agent_class: type,
    timeout_s: float,
    poll_interval_s: float = 1.0,
) -> dict[str, AgentId[Any]]:
    """Poll ``transport.discover()`` until every named peer is found.

    Args:
        transport: An open exchange transport already registered for the
            local rank's own agent. Discovery is read-only from this
            rank's perspective -- it does not create or mutate mailboxes.
        peer_names: Names of agents this rank intends to message. Each
            name must match the ``AgentId.name`` of an agent previously
            registered against the same exchange (potentially by a
            different process on a different host).
        agent_class: Concrete ``Agent`` subclass to scope the discovery
            query (``transport.discover`` is class-typed). All ChemGraph
            agents are ``ChemGraphLogicalAgent``, so callers pass that.
        timeout_s: Wall-clock budget. On expiry a ``TimeoutError`` is
            raised whose message lists the peers we never saw, so
            operators can immediately tell which remote site is missing
            or whose agent failed to register.
        poll_interval_s: Backoff between ``discover()`` retries. The
            default keeps startup snappy without hammering the exchange.

    Returns:
        Mapping from peer name to the discovered ``AgentId``. Empty
        ``peer_names`` short-circuits to an empty dict.
    """
    wanted = set(peer_names)
    if not wanted:
        return {}
    found: dict[str, AgentId[Any]] = {}
    deadline = time.monotonic() + timeout_s
    while True:
        agent_ids = await transport.discover(agent_class)
        for agent_id in agent_ids:
            name = getattr(agent_id, 'name', None)
            if isinstance(name, str) and name in wanted and name not in found:
                found[name] = agent_id
        missing = wanted.difference(found)
        if not missing:
            return found
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f'Timed out after {timeout_s:.1f}s waiting to discover '
                f'peer agents on the exchange: missing={sorted(missing)}. '
                f'Confirm every site of the federated campaign has '
                f'started and registered its agents under the expected '
                f'names (run-id-prefixed names are required when the '
                f'hosted exchange is shared across operators).',
            )
        logger.debug(
            'discover() missing %d peers (%s); sleeping %.1fs',
            len(missing), sorted(missing), poll_interval_s,
        )
        await asyncio.sleep(poll_interval_s)
