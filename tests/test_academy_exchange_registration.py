from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from academy.identifier import AgentId

from chemgraph.academy.core.campaign import ChemGraphDaemonConfig
from chemgraph.academy.runtime.exchange import build_exchange_factory
from chemgraph.academy.runtime.exchange import exchange_uses_redis
from chemgraph.academy.runtime.exchange import SUPPORTED_EXCHANGE_TYPES
from chemgraph.academy.runtime.registration import discover_peer_agent_ids


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


# ---------------------------------------------------------------------------
# Exchange factory dispatch
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# discover_peer_agent_ids (federated peer discovery)
#
# Real exchange transports require a running broker. We exercise the
# discovery helper against a fake transport whose ``discover`` returns
# a sequence we control across successive polls. This keeps the tests
# fast and deterministic while still pinning the behavior that matters:
# wait-for-peers, success when all are present, timeout listing the
# missing ones.
# ---------------------------------------------------------------------------


class _FakeTransport:
    """Minimal ``transport.discover()`` stand-in for the discovery tests.

    Configure with a list of "rounds"; each call to ``discover()``
    returns (and consumes) one round. After the configured rounds run
    out the last one keeps being returned, so timeout tests can assert
    'never converged'.
    """

    def __init__(self, rounds: list[list[AgentId[Any]]]) -> None:
        self._rounds = rounds
        self._calls = 0

    async def discover(self, agent_class):  # noqa: ARG002 - sig match only
        index = min(self._calls, len(self._rounds) - 1)
        self._calls += 1
        return tuple(self._rounds[index])


def _agent_id(name: str) -> AgentId[Any]:
    return AgentId.new(name)


def test_discover_peer_agent_ids_returns_empty_for_empty_peer_list() -> None:
    """When the local agent has no allowed_peers the helper short-circuits
    -- it must not poll the exchange unnecessarily (the network round-trip
    would block daemon startup for nothing)."""
    transport = _FakeTransport(rounds=[[_agent_id('anyone')]])
    result = asyncio.run(
        discover_peer_agent_ids(
            transport, [], agent_class=object, timeout_s=0.01,
        ),
    )
    assert result == {}
    assert transport._calls == 0


def test_discover_peer_agent_ids_finds_all_peers_on_first_poll() -> None:
    """Happy path: every peer is already on the exchange. The helper
    must return promptly with a name->AgentId mapping in the same
    direction the daemon will use it (peer name -> AgentId for Handle
    construction)."""
    a = _agent_id('worker-a')
    b = _agent_id('worker-b')
    c = _agent_id('coordinator')
    transport = _FakeTransport(rounds=[[a, b, c]])
    result = asyncio.run(
        discover_peer_agent_ids(
            transport, ['worker-a', 'worker-b'],
            agent_class=object, timeout_s=1.0,
        ),
    )
    assert result == {'worker-a': a, 'worker-b': b}
    # Did NOT include the un-requested coordinator -- filtering by name
    # is what keeps cross-operator agents on the shared hosted exchange
    # from leaking into each other's peer dicts.
    assert 'coordinator' not in result


def test_discover_peer_agent_ids_waits_for_late_peers() -> None:
    """The federated convergence story: site A registers at t=0 and
    polls; site B doesn't register its agent until poll #3; the helper
    must keep polling and succeed once B is visible. This is the
    behavior that lets operators bring sites up in any order."""
    a = _agent_id('worker-a')
    b = _agent_id('worker-b')
    rounds = [
        [a],         # poll 1: only A visible
        [a],         # poll 2: still waiting for B
        [a, b],      # poll 3: B comes up
    ]
    transport = _FakeTransport(rounds=rounds)
    result = asyncio.run(
        discover_peer_agent_ids(
            transport, ['worker-a', 'worker-b'],
            agent_class=object,
            timeout_s=2.0,
            poll_interval_s=0.01,  # keep the test fast
        ),
    )
    assert result == {'worker-a': a, 'worker-b': b}


def test_discover_peer_agent_ids_times_out_naming_missing_peers() -> None:
    """When a remote site never shows up, the helper must raise with a
    message that names the missing peers. Operators reading the log
    should immediately know which site to bring up / debug."""
    transport = _FakeTransport(rounds=[[_agent_id('worker-a')]])
    with pytest.raises(TimeoutError, match=r"missing=\['coordinator', 'worker-b'\]"):
        asyncio.run(
            discover_peer_agent_ids(
                transport, ['worker-a', 'worker-b', 'coordinator'],
                agent_class=object,
                timeout_s=0.05,
                poll_interval_s=0.01,
            ),
        )


def test_discover_peer_agent_ids_skips_already_found_peers_on_re_poll() -> None:
    """If poll N saw peer A, poll N+1 must not overwrite A's AgentId
    even if discover returns a fresh AgentId object with the same name
    (which the hosted exchange doesn't actually do, but the helper's
    behavior shouldn't depend on that). Pin the 'first found wins'
    invariant explicitly."""
    a_first = _agent_id('worker-a')
    a_again = _agent_id('worker-a')  # different uuid, same name
    b = _agent_id('worker-b')
    rounds = [
        [a_first],
        [a_again, b],
    ]
    transport = _FakeTransport(rounds=rounds)
    result = asyncio.run(
        discover_peer_agent_ids(
            transport, ['worker-a', 'worker-b'],
            agent_class=object,
            timeout_s=2.0,
            poll_interval_s=0.01,
        ),
    )
    assert result['worker-a'] is a_first
    assert result['worker-b'] is b
