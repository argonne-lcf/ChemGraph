from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

# Skip when the optional 'academy' extra is absent; this module
# imports academy.exchange.* directly at top level.
pytest.importorskip("academy")

from academy.identifier import AgentId

from chemgraph.academy.core.campaign import ChemGraphDaemonConfig
from chemgraph.academy.runtime.exchange import build_exchange_factory
from chemgraph.academy.runtime.registration import deterministic_agent_id
from chemgraph.academy.runtime.registration import deterministic_agent_uid
from chemgraph.academy.runtime.registration import wait_for_peers_alive


def _config(tmp_path: Path, exchange_type: str) -> ChemGraphDaemonConfig:
    return ChemGraphDaemonConfig(
        run_dir=tmp_path,
        run_token='token-1',
        agent_count=1,
        campaign_config=tmp_path / 'campaign.jsonc',
        lm_config=tmp_path / 'lm.json',
        max_decisions=1,
        poll_timeout_s=1.0,
        idle_timeout_s=1.0,
        startup_timeout_s=1.0,
        completion_timeout_s=1.0,
        status_interval_s=1.0,
        redis_host='localhost',
        redis_port=6392,
        redis_namespace='ns',
        rank=0,
        local_rank=0,
        chemgraph_repo_root=tmp_path,
        exchange_type=exchange_type,
    )


@pytest.mark.parametrize(
    ('exchange_type', 'expected_class'),
    [
        ('redis', 'RedisExchangeFactory'),
        ('local', 'LocalExchangeFactory'),
        ('hybrid', 'HybridExchangeFactory'),
    ],
)
def test_build_exchange_factory_dispatches_by_config(
    tmp_path,
    exchange_type,
    expected_class,
) -> None:
    factory = build_exchange_factory(_config(tmp_path, exchange_type))

    assert type(factory).__name__ == expected_class


def test_build_exchange_factory_rejects_unknown_exchange(tmp_path) -> None:
    with pytest.raises(ValueError, match='Unsupported exchange type'):
        build_exchange_factory(_config(tmp_path, 'bad'))


class _FakeTransport:
    def __init__(self, rounds: list[list[AgentId[Any]]]) -> None:
        self._rounds = rounds
        self._calls = 0

    async def discover(self, agent_class):
        index = min(self._calls, len(self._rounds) - 1)
        self._calls += 1
        return tuple(self._rounds[index])


def test_deterministic_agent_identity_is_stable_and_namespaced() -> None:
    first = deterministic_agent_uid(run_id='run-1', agent_name='worker')
    repeated = deterministic_agent_uid(run_id='run-1', agent_name='worker')
    other_run = deterministic_agent_uid(run_id='run-2', agent_name='worker')
    other_agent = deterministic_agent_uid(
        run_id='run-1',
        agent_name='coordinator',
    )

    assert first == repeated
    assert first != other_run
    assert first != other_agent

    agent_id = deterministic_agent_id(
        run_id='run-1',
        agent_name='worker',
    )
    assert agent_id.uid == first
    assert agent_id.name == 'worker'


def test_wait_for_peers_alive_short_circuits_without_peers() -> None:
    transport = _FakeTransport(rounds=[[]])

    asyncio.run(
        wait_for_peers_alive(
            transport,
            [],
            agent_class=object,
            timeout_s=0.01,
        ),
    )

    assert transport._calls == 0


def test_wait_for_peers_alive_matches_hosted_exchange_uids() -> None:
    worker = deterministic_agent_id(
        run_id='run-1',
        agent_name='worker',
    )
    coordinator = deterministic_agent_id(
        run_id='run-1',
        agent_name='coordinator',
    )
    worker_seen = AgentId(uid=worker.uid, name=None, role='agent')
    coordinator_seen = AgentId(
        uid=coordinator.uid,
        name=None,
        role='agent',
    )
    transport = _FakeTransport(
        rounds=[
            [worker_seen],
            [worker_seen, coordinator_seen],
        ],
    )

    asyncio.run(
        wait_for_peers_alive(
            transport,
            [worker, coordinator],
            agent_class=object,
            timeout_s=1.0,
            poll_interval_s=0.01,
        ),
    )

    assert transport._calls == 2


def test_wait_for_peers_alive_times_out_with_missing_peer_name() -> None:
    visible = deterministic_agent_id(
        run_id='run-1',
        agent_name='visible',
    )
    missing = deterministic_agent_id(
        run_id='run-1',
        agent_name='missing',
    )
    visible_seen = AgentId(uid=visible.uid, name=None, role='agent')
    transport = _FakeTransport(rounds=[[visible_seen]])

    with pytest.raises(TimeoutError, match='missing'):
        asyncio.run(
            wait_for_peers_alive(
                transport,
                [visible, missing],
                agent_class=object,
                timeout_s=0.05,
                poll_interval_s=0.01,
            ),
        )
