from __future__ import annotations

from pathlib import Path

import pytest
from academy.exchange.hybrid import HybridAgentRegistration
from academy.exchange.local import LocalAgentRegistration
from academy.exchange.redis import RedisAgentRegistration
from academy.identifier import AgentId

from chemgraph.academy.core.campaign import ChemGraphDaemonConfig
from chemgraph.academy.runtime.exchange import build_exchange_factory
from chemgraph.academy.runtime.registration import load_academy_registrations
from chemgraph.academy.runtime.registration import registration_payload
from chemgraph.academy.runtime.registration import write_academy_registrations


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


@pytest.mark.parametrize(
    'registration_cls',
    [
        RedisAgentRegistration,
        LocalAgentRegistration,
        HybridAgentRegistration,
    ],
)
def test_academy_registration_round_trips_by_exchange_type(
    tmp_path,
    registration_cls,
) -> None:
    registration = registration_cls(agent_id=AgentId.new('agent-a'))
    write_academy_registrations(
        run_dir=tmp_path,
        run_token='token-1',
        registrations={'agent-a': registration},
    )

    loaded = load_academy_registrations(tmp_path, run_token='token-1')

    assert isinstance(loaded['agent-a'], registration_cls)
    assert loaded['agent-a'].agent_id == registration.agent_id


def test_registration_payload_rejects_mixed_exchange_types() -> None:
    with pytest.raises(ValueError, match='mixed exchange types'):
        registration_payload(
            run_token='token-1',
            registrations={
                'redis-agent': RedisAgentRegistration(
                    agent_id=AgentId.new('redis-agent'),
                ),
                'local-agent': LocalAgentRegistration(
                    agent_id=AgentId.new('local-agent'),
                ),
            },
        )


def test_registration_payload_rejects_empty_registrations() -> None:
    with pytest.raises(ValueError, match='at least one registration'):
        registration_payload(run_token='token-1', registrations={})
