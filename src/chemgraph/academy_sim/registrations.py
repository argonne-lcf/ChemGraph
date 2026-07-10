"""Peer AgentId publication for academy_sim."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from chemgraph.academy_sim.artifacts import write_json_atomic
from chemgraph.academy_sim.errors import PeerRegistrationError

_REGISTRATION_FILE = 'academy_sim_registrations.json'
_GRAPH_UID_NAMESPACE = uuid.UUID('b621031d-5559-4f1b-8af3-b6f118af52ec')


def registrations_path(run_dir: Path) -> Path:
    return run_dir / _REGISTRATION_FILE


def deterministic_graph_uid(*, run_id: str, graph_name: str) -> uuid.UUID:
    """Return the stable Academy mailbox UID for one graph in one run."""

    return uuid.uuid5(_GRAPH_UID_NAMESPACE, f'{run_id}/{graph_name}')


def deterministic_graph_agent_id(*, run_id: str, graph_name: str) -> Any:
    """Construct the Academy AgentId every site can derive locally."""

    from academy.identifier import AgentId

    return AgentId(
        uid=deterministic_graph_uid(run_id=run_id, graph_name=graph_name),
        name=graph_name,
        role='agent',
    )


async def register_http_agent_with_id(
    *,
    transport: Any,
    agent_class: type,
    agent_id: Any,
) -> Any:
    """Register an Academy HTTP mailbox using a caller-supplied AgentId.

    Academy's public registration helper creates random AgentIds. Cross-HPC
    graph rendezvous needs stable UIDs, so this mirrors the HTTP transport's
    mailbox registration call while swapping in the deterministic AgentId.
    """

    try:
        session = transport._session
        mailbox_url = transport._mailbox_url
        mailbox_id = transport.mailbox_id
    except AttributeError as exc:
        raise PeerRegistrationError(
            'exchange registration requires Academy HTTP transport internals; '
            'use exchange.type="http" or exchange.registration="file"'
        ) from exc
    payload = {
        'mailbox': agent_id.model_dump_json(),
        'agent': ','.join(agent_class._agent_mro()),
    }
    async with session.post(mailbox_url, json=payload) as response:
        from academy.exchange.cloud.client import _raise_for_status

        _raise_for_status(response, mailbox_id, agent_id)
    return agent_id


def http_agent_registration(agent_id: Any) -> Any:
    """Return an Academy Runtime registration object for an HTTP AgentId."""

    from academy.exchange.cloud.client import HttpAgentRegistration

    return HttpAgentRegistration(agent_id=agent_id)


async def wait_for_peer_uids(
    transport: Any,
    peer_agent_ids: Iterable[Any],
    *,
    agent_class: type,
    timeout_s: float,
    poll_interval_s: float = 1.0,
) -> None:
    """Wait until all peer AgentId UIDs are visible on the exchange."""

    wanted = {peer.uid: peer for peer in peer_agent_ids}
    if not wanted:
        return
    seen: set[uuid.UUID] = set()
    deadline = time.monotonic() + timeout_s
    while True:
        for agent_id in await transport.discover(agent_class):
            if agent_id.uid in wanted:
                seen.add(agent_id.uid)
        missing = set(wanted).difference(seen)
        if not missing:
            return
        if time.monotonic() >= deadline:
            missing_desc = sorted(f'{wanted[uid].name}({uid})' for uid in missing)
            raise TimeoutError(
                'timed out waiting for peer graph AgentIds on the exchange: '
                f'{missing_desc}. Check that every site uses the same run_id, '
                'exchange URL, and Academy/Globus credentials.'
            )
        await asyncio.sleep(poll_interval_s)


def publish_agent_id(
    *,
    run_dir: Path,
    run_id: str,
    run_token: str,
    launch_token: str,
    exchange_type: str,
    graph: str,
    agent_id: Any,
) -> None:
    """Publish one graph's Academy AgentId into the shared run file."""

    path = registrations_path(run_dir)
    if path.exists():
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('run_token') != run_token:
            raise PeerRegistrationError(
                f'registration file {path} belongs to a different run token'
            )
        if payload.get('run_id') != run_id:
            raise PeerRegistrationError(
                f'registration file {path} belongs to run {payload.get("run_id")!r}'
            )
    else:
        payload = {
            'run_id': run_id,
            'run_token': run_token,
            'exchange_type': exchange_type,
            'agents': {},
        }
    payload.setdefault('agents', {})[graph] = {
        'agent_id': _dump_agent_id(agent_id),
        'launch_token': launch_token,
        'published_at': time.time(),
    }
    write_json_atomic(path, payload)


def load_agent_ids(
    run_dir: Path,
    *,
    run_token: str,
    launch_token: str | None = None,
) -> dict[str, Any]:
    """Load graph name -> Academy AgentId from the shared run file."""

    path = registrations_path(run_dir)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('run_token') != run_token:
        raise PeerRegistrationError(
            f'registration file {path} belongs to a different run token'
        )
    agents = payload.get('agents')
    if not isinstance(agents, dict):
        raise PeerRegistrationError(f'registration file {path} is malformed')

    from academy.identifier import AgentId

    loaded: dict[str, Any] = {}
    for name, entry in agents.items():
        if _entry_launch_token(entry) != launch_token and launch_token is not None:
            continue
        loaded[name] = AgentId[Any].model_validate(_entry_agent_id(entry))
    return loaded


async def wait_for_agent_ids(
    run_dir: Path,
    *,
    run_token: str,
    launch_token: str | None = None,
    names: set[str],
    timeout_s: float,
) -> dict[str, Any]:
    """Wait until all requested peer AgentIds have been published."""

    deadline = time.monotonic() + timeout_s
    while True:
        if registrations_path(run_dir).exists():
            ids = load_agent_ids(
                run_dir,
                run_token=run_token,
                launch_token=launch_token,
            )
            if names.issubset(ids):
                return {name: ids[name] for name in names}
        if time.monotonic() > deadline:
            missing = sorted(names.difference(ids if 'ids' in locals() else {}))
            raise TimeoutError(f'timed out waiting for peer AgentIds: {missing}')
        await asyncio.sleep(0.25)


def _dump_agent_id(agent_id: Any) -> dict[str, Any]:
    if hasattr(agent_id, 'model_dump'):
        return agent_id.model_dump(mode='json')
    if isinstance(agent_id, dict):
        return agent_id
    raise TypeError(f'cannot serialize AgentId object {type(agent_id).__name__}')


def _entry_agent_id(entry: Any) -> dict[str, Any]:
    """Return AgentId payload from current or legacy registration entries."""

    if isinstance(entry, dict) and isinstance(entry.get('agent_id'), dict):
        return entry['agent_id']
    if isinstance(entry, dict):
        return entry
    raise PeerRegistrationError(f'malformed AgentId registration entry: {entry!r}')


def _entry_launch_token(entry: Any) -> str | None:
    if isinstance(entry, dict) and isinstance(entry.get('launch_token'), str):
        return entry['launch_token']
    return None
