"""Peer-agent identity + readiness for federated ChemGraph campaigns.

Cross-site peer rendezvous needs each rank to be able to address every
other site's agents without a shared filesystem. The first cut polled
``transport.discover(ChemGraphLogicalAgent)`` and filtered by
``AgentId.name``, which works on the local/redis/hybrid exchanges but
breaks on Academy's hosted HTTP exchange: ``discover()`` returns
``AgentId`` objects with ``name=None`` and ``role='agent'`` only.
Names round-trip through the server's mailbox state but are not
echoed back in the discovery response.

The replacement: agree on a **deterministic UID** for every (run-id,
agent-name) pair. Both sites compute the same UID from the same
inputs, so each side knows the recipient's UID before either rank
boots. ``discover()`` is still useful as a liveness probe (matching
on UID, which IS preserved across the server round-trip) so a rank
can wait until its peers have actually registered before proceeding.

Side effect of this scheme: agent names become campaign-scoped. Two
operators running the SAME ``federated-hello`` campaign concurrently
would clash on the same UIDs and crash the registration POST with
"mailbox already exists". The run-id is part of the UID namespace,
so as long as operators bump ``--run-id`` (federated-hello-001 vs
federated-hello-002) the UIDs differ and the campaigns don't see
each other.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Iterable
from typing import Any

from academy.exchange.transport import ExchangeTransportT
from academy.identifier import AgentId

logger = logging.getLogger(__name__)


# Stable namespace UUID used as the seed for uuid5 derivation. The
# value itself doesn't matter -- only that every site computes the
# same UID for the same (run_id, agent_name) pair. Bumping this
# constant would invalidate every running deployment, so don't.
_PEER_UID_NAMESPACE = uuid.UUID('1e7eda44-1b34-4f5a-b2a1-cf5ca5db8e8b')


def deterministic_agent_uid(*, run_id: str, agent_name: str) -> uuid.UUID:
    """Derive the AgentId.uid that every site will use for ``agent_name``.

    Same inputs on Aurora and Crux ⇒ same UID. The recipient side
    registers with this UID; the sender side constructs an
    ``AgentId`` with the same UID locally and uses it to build a
    ``Handle`` without ever calling ``discover()``.
    """
    return uuid.uuid5(_PEER_UID_NAMESPACE, f"{run_id}/{agent_name}")


def deterministic_agent_id(*, run_id: str, agent_name: str) -> AgentId[Any]:
    """Construct the ``AgentId`` peers can reconstruct from name alone."""
    return AgentId(
        uid=deterministic_agent_uid(run_id=run_id, agent_name=agent_name),
        name=agent_name,
        role='agent',
    )


async def register_agent_with_uid(
    transport: ExchangeTransportT,
    agent_class: type,
    agent_id: AgentId[Any],
) -> AgentId[Any]:
    """Register ``agent_id`` on the exchange, reusing the supplied UID.

    Bypasses ``transport.register_agent`` (which always calls
    ``AgentId.new`` and generates a random UID) by POSTing directly
    to the same mailbox endpoint with our pre-built AgentId. Returns
    the same AgentId on success so callers can hand it to Runtime.
    """
    # Reach into the transport for the same session + URL the SDK uses.
    # The shape mirrors HttpExchangeTransport.register_agent exactly,
    # we just swap the auto-generated AgentId for the deterministic one.
    session = transport._session
    mailbox_url = transport._mailbox_url
    payload = {
        'mailbox': agent_id.model_dump_json(),
        'agent': ','.join(agent_class._agent_mro()),
    }
    async with session.post(mailbox_url, json=payload) as response:
        # _raise_for_status is what the SDK uses; reach in for it too.
        from academy.exchange.cloud.client import _raise_for_status
        _raise_for_status(response, transport.mailbox_id, agent_id)
    return agent_id


async def wait_for_peers_alive(
    transport: ExchangeTransportT,
    peer_ids: Iterable[AgentId[Any]],
    *,
    agent_class: type,
    timeout_s: float,
    poll_interval_s: float = 1.0,
) -> None:
    """Block until every peer in ``peer_ids`` is visible to ``discover()``.

    UID-based matching: ``discover()`` strips names but preserves
    UIDs, so we filter the discover response by UID and wait until
    every expected peer's mailbox shows up. If ``peer_ids`` is empty
    (single-agent or self-only slice), return immediately.

    Raises ``TimeoutError`` listing the missing peers' UIDs after
    ``timeout_s`` so the operator can correlate with their other
    site's launch logs.
    """
    wanted = {peer.uid: peer for peer in peer_ids}
    if not wanted:
        return
    seen: set[uuid.UUID] = set()
    deadline = time.monotonic() + timeout_s
    while True:
        agent_ids = await transport.discover(agent_class)
        for aid in agent_ids:
            if aid.uid in wanted:
                seen.add(aid.uid)
        missing = set(wanted).difference(seen)
        if not missing:
            return
        if time.monotonic() >= deadline:
            missing_descs = sorted(f'{wanted[u].name}({u})' for u in missing)
            raise TimeoutError(
                f'Timed out after {timeout_s:.1f}s waiting for peer agents '
                f'to register on the exchange: missing={missing_descs}. '
                f'Confirm every site of the federated campaign has started '
                f'and that all sites are using the same --run-id (the run-id '
                f'is part of the UID namespace; mismatches make the peers '
                f'invisible to each other).',
            )
        logger.debug(
            'wait_for_peers_alive: missing %d (%s); sleeping %.1fs',
            len(missing), sorted(missing), poll_interval_s,
        )
        await asyncio.sleep(poll_interval_s)
