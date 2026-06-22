from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

# Skip when the optional 'academy' extra is absent; this module
# imports academy.* directly at top level.
pytest.importorskip("academy")

from academy.identifier import AgentId
from academy.exchange.cloud.client import DEFAULT_EXCHANGE_URL

from chemgraph.academy.core.campaign import ChemGraphDaemonConfig
from chemgraph.academy.runtime.exchange import build_exchange_factory
from chemgraph.academy.runtime.exchange import exchange_uses_redis
from chemgraph.academy.runtime.exchange import SUPPORTED_EXCHANGE_TYPES
from chemgraph.academy.runtime.registration import deterministic_agent_id
from chemgraph.academy.runtime.registration import deterministic_agent_uid
from chemgraph.academy.runtime.registration import wait_for_peers_alive


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


class HttpExchangeFactory:
    def __init__(self, url: str = DEFAULT_EXCHANGE_URL, **kwargs: Any) -> None:
        self._info = SimpleNamespace(url=url)
        self.kwargs = kwargs


def _stub_http_exchange_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    import academy.exchange.cloud as cloud

    monkeypatch.setattr(cloud, 'HttpExchangeFactory', HttpExchangeFactory)


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
    monkeypatch,
) -> None:
    if exchange_type == 'http':
        _stub_http_exchange_factory(monkeypatch)

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
    monkeypatch,
) -> None:
    """A ``None`` ``http_exchange_url`` must select Academy's hosted
    default (https://exchange.academy-agents.org/v1). This is the path
    every cross-HPC campaign takes unless the operator stands up a
    self-hosted exchange."""
    _stub_http_exchange_factory(monkeypatch)
    factory = build_exchange_factory(_config(tmp_path, 'http'))

    # Upstream stores connection details on factory._info; reach into
    # it to make sure we hand off the URL we mean to.
    assert factory._info.url == DEFAULT_EXCHANGE_URL


def test_http_exchange_factory_honors_custom_url(tmp_path, monkeypatch) -> None:
    """Operators must be able to point at a self-hosted HTTP exchange
    server (``python -m academy.exchange.cloud``). This is the escape
    hatch when the public Academy server is unavailable or undesired."""
    custom = 'https://my-private-exchange.example.com/v1'
    _stub_http_exchange_factory(monkeypatch)
    factory = build_exchange_factory(
        _config(tmp_path, 'http', http_exchange_url=custom),
    )

    assert factory._info.url == custom


# ---------------------------------------------------------------------------
# Deterministic peer identity + wait_for_peers_alive
#
# The hosted HttpExchange strips AgentId.name from discover() responses
# (only ``uid`` and ``role`` round-trip). Name-based discovery was
# silently never finding any peer across sites. The replacement: derive
# each peer's mailbox UID deterministically from (run_id, agent_name)
# so every site can construct the same AgentId locally without needing
# discover() to echo the name back. discover() stays useful as a
# liveness probe (matching on UID, which IS preserved).
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


def test_deterministic_agent_uid_is_stable() -> None:
    """Same (run_id, agent_name) inputs must produce the same UID,
    every call, on every machine. This is the load-bearing invariant
    of the federated rendezvous: Aurora and Crux compute the same
    UID locally and addressing works without any shared lookup."""
    a = deterministic_agent_uid(run_id='r-001', agent_name='worker')
    b = deterministic_agent_uid(run_id='r-001', agent_name='worker')
    assert a == b


def test_deterministic_agent_uid_differs_by_run_id() -> None:
    """Different run-ids must yield different UIDs so two operators
    running the SAME campaign with different --run-ids don't collide
    on the same mailboxes."""
    a = deterministic_agent_uid(run_id='r-001', agent_name='worker')
    b = deterministic_agent_uid(run_id='r-002', agent_name='worker')
    assert a != b


def test_deterministic_agent_uid_differs_by_agent_name() -> None:
    """Different agent names within the same run must yield different
    UIDs so per-agent mailboxes don't collide."""
    a = deterministic_agent_uid(run_id='r-001', agent_name='worker-a')
    b = deterministic_agent_uid(run_id='r-001', agent_name='worker-b')
    assert a != b


def test_deterministic_agent_id_preserves_name_locally() -> None:
    """The AgentId we build for our OWN registration carries the
    name so it shows up in trace events; the name is only stripped
    when the AgentId is round-tripped through the hosted exchange's
    discover() response."""
    aid = deterministic_agent_id(run_id='r-001', agent_name='worker-a')
    assert aid.name == 'worker-a'
    assert aid.uid == deterministic_agent_uid(
        run_id='r-001', agent_name='worker-a',
    )


def test_wait_for_peers_alive_returns_immediately_for_empty_list() -> None:
    """When the local agent has no allowed_peers the helper short-
    circuits -- it must not poll the exchange unnecessarily."""
    transport = _FakeTransport(rounds=[[]])
    asyncio.run(
        wait_for_peers_alive(
            transport, [], agent_class=object, timeout_s=0.01,
        ),
    )
    assert transport._calls == 0


def test_wait_for_peers_alive_succeeds_when_all_uids_present() -> None:
    """Happy path: every expected peer's mailbox is on the exchange.
    Match by UID (the field discover() preserves), not name."""
    a = deterministic_agent_id(run_id='r-001', agent_name='worker-a')
    b = deterministic_agent_id(run_id='r-001', agent_name='worker-b')
    # Simulate what the hosted exchange actually returns: AgentIds
    # with the right UID but name stripped. Tests would have caught
    # the original bug if they'd matched this shape.
    a_seen = AgentId(uid=a.uid, name=None, role='agent')
    b_seen = AgentId(uid=b.uid, name=None, role='agent')
    transport = _FakeTransport(rounds=[[a_seen, b_seen]])
    asyncio.run(
        wait_for_peers_alive(
            transport, [a, b], agent_class=object, timeout_s=1.0,
        ),
    )


def test_wait_for_peers_alive_waits_across_polls_for_late_peer() -> None:
    """The federated convergence story: bring sites up in any order;
    the wait keeps polling and unblocks the moment all UIDs are seen."""
    a = deterministic_agent_id(run_id='r-001', agent_name='worker-a')
    b = deterministic_agent_id(run_id='r-001', agent_name='worker-b')
    a_seen = AgentId(uid=a.uid, name=None, role='agent')
    b_seen = AgentId(uid=b.uid, name=None, role='agent')
    rounds = [
        [a_seen],          # poll 1: only A visible
        [a_seen],          # poll 2: still waiting for B
        [a_seen, b_seen],  # poll 3: B comes up
    ]
    transport = _FakeTransport(rounds=rounds)
    asyncio.run(
        wait_for_peers_alive(
            transport, [a, b],
            agent_class=object,
            timeout_s=2.0,
            poll_interval_s=0.01,
        ),
    )


def test_wait_for_peers_alive_times_out_naming_missing_uids() -> None:
    """When a remote site never registers, raise with a message
    naming the missing peers (name + uid). Operators reading the
    log can correlate with the missing site's launch logs."""
    a = deterministic_agent_id(run_id='r-001', agent_name='worker-a')
    missing = deterministic_agent_id(run_id='r-001', agent_name='no-such-peer')
    a_seen = AgentId(uid=a.uid, name=None, role='agent')
    transport = _FakeTransport(rounds=[[a_seen]])
    with pytest.raises(TimeoutError, match='no-such-peer'):
        asyncio.run(
            wait_for_peers_alive(
                transport, [a, missing],
                agent_class=object,
                timeout_s=0.05,
                poll_interval_s=0.01,
            ),
        )


def test_wait_for_peers_alive_ignores_unrelated_agents_with_same_class() -> None:
    """The hosted exchange returns every ChemGraphLogicalAgent registered
    across all operators / campaigns. The wait must filter strictly by
    UID and not get confused by other operators' agents."""
    a = deterministic_agent_id(run_id='r-001', agent_name='worker-a')
    a_seen = AgentId(uid=a.uid, name=None, role='agent')
    # Lots of noise from other operators / runs:
    noise = [
        AgentId.new('stranger-1'),
        AgentId.new('stranger-2'),
        AgentId.new('stranger-3'),
    ]
    transport = _FakeTransport(rounds=[noise + [a_seen]])
    asyncio.run(
        wait_for_peers_alive(
            transport, [a],
            agent_class=object,
            timeout_s=1.0,
            poll_interval_s=0.01,
        ),
    )
