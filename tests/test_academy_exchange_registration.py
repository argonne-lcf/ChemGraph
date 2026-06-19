from __future__ import annotations

from pathlib import Path

import pytest
from academy.exchange.cloud.client import HttpAgentRegistration
from academy.exchange.hybrid import HybridAgentRegistration
from academy.exchange.local import LocalAgentRegistration
from academy.exchange.redis import RedisAgentRegistration
from academy.identifier import AgentId

from chemgraph.academy.core.campaign import ChemGraphDaemonConfig
from chemgraph.academy.runtime.exchange import build_exchange_factory
from chemgraph.academy.runtime.exchange import exchange_uses_redis
from chemgraph.academy.runtime.exchange import SUPPORTED_EXCHANGE_TYPES
from chemgraph.academy.runtime.registration import load_academy_registrations
from chemgraph.academy.runtime.registration import registration_payload
from chemgraph.academy.runtime.registration import write_academy_registrations


def _config(
    tmp_path: Path,
    exchange_type: str,
    *,
    http_exchange_url: str | None = None,
) -> ChemGraphDaemonConfig:
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
        http_exchange_url=http_exchange_url,
    )


@pytest.mark.parametrize(
    ('exchange_type', 'expected_class'),
    [
        ('redis', 'RedisExchangeFactory'),
        ('local', 'LocalExchangeFactory'),
        ('hybrid', 'HybridExchangeFactory'),
        ('http', 'HttpExchangeFactory'),
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


def test_supported_exchange_types_exposes_full_dispatch_table() -> None:
    """SUPPORTED_EXCHANGE_TYPES drives both the CLI ``choices`` argument
    on ``compute_launcher.parse_args`` and ``daemon.parse_args``. If the
    set drifts from what ``build_exchange_factory`` actually handles,
    the CLI happily accepts a value that then raises at runtime."""
    assert set(SUPPORTED_EXCHANGE_TYPES) == {'redis', 'local', 'hybrid', 'http'}


def test_exchange_uses_redis_helper_matches_dispatch_table() -> None:
    """The compute launcher uses this helper to decide whether to start a
    Redis subprocess on rank 0. Pin the answers for every supported type
    so adding a new exchange forces a conscious yes/no decision here."""
    assert exchange_uses_redis('redis') is True
    assert exchange_uses_redis('hybrid') is True
    assert exchange_uses_redis('local') is False
    assert exchange_uses_redis('http') is False


def test_http_exchange_factory_uses_hosted_default_when_url_omitted(
    tmp_path,
) -> None:
    """A ``None`` ``http_exchange_url`` must select Academy's hosted
    default (https://exchange.academy-agents.org/v1). This is the path
    every cross-HPC campaign takes unless the operator stands up a
    self-hosted exchange."""
    from academy.exchange.cloud.client import DEFAULT_EXCHANGE_URL
    factory = build_exchange_factory(_config(tmp_path, 'http'))

    # Upstream stores connection details on factory._info; reach into
    # it to make sure we hand off the URL we mean to.
    assert factory._info.url == DEFAULT_EXCHANGE_URL


def test_http_exchange_factory_honors_custom_url(tmp_path) -> None:
    """Operators must be able to point at a self-hosted HTTP exchange
    server (``python -m academy.exchange.cloud``). This is the escape
    hatch when the public Academy server is unavailable or undesired."""
    custom = 'https://my-private-exchange.example.com/v1'
    factory = build_exchange_factory(
        _config(tmp_path, 'http', http_exchange_url=custom),
    )

    assert factory._info.url == custom


@pytest.mark.parametrize(
    'registration_cls',
    [
        RedisAgentRegistration,
        LocalAgentRegistration,
        HybridAgentRegistration,
        HttpAgentRegistration,
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
