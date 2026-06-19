from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from academy.identifier import AgentId

from chemgraph.academy.runtime import bootstrap
from chemgraph.academy.runtime.registration import deterministic_agent_id


# ---------------------------------------------------------------------------
# parse_args -- CLI surface
# ---------------------------------------------------------------------------


def test_parse_args_requires_campaign() -> None:
    """``--campaign`` is one of the two required fields. Bootstrap is
    useless without knowing which campaign's user_task to send."""
    with pytest.raises(SystemExit):
        bootstrap.parse_args(['--run-id', 'r-001'])


def test_parse_args_requires_run_id() -> None:
    """``--run-id`` is required because the recipient's mailbox UID
    is derived deterministically from (run_id, agent_name). Without
    it the bootstrap would address a different mailbox than the
    spawn-site invocations registered."""
    with pytest.raises(SystemExit):
        bootstrap.parse_args(['--campaign', 'mace-ensemble-screening-20'])


def test_parse_args_defaults_exchange_type_to_http() -> None:
    """Federated bootstrap is the main use case so http is the right
    default. Operators on single-machine runs can override but they
    rarely need this command at all (run-compute auto-bootstraps)."""
    args = bootstrap.parse_args([
        '--campaign', 'mace-ensemble-screening-20',
        '--run-id', 'r-001',
    ])
    assert args.exchange_type == 'http'
    assert args.recipient is None  # defaults to campaign.initial_agent
    # discover-timeout-s default now matches spawn-site's 600s.
    assert args.discover_timeout_s == pytest.approx(600.0)


def test_parse_args_accepts_recipient_override() -> None:
    """Operator should be able to bootstrap a non-initial agent for
    e.g. partial re-runs or debugging."""
    args = bootstrap.parse_args([
        '--campaign', 'foo.jsonc',
        '--run-id', 'r-001',
        '--recipient', 'worker-a',
    ])
    assert args.recipient == 'worker-a'


# ---------------------------------------------------------------------------
# dispatch_bootstrap -- the core async path
#
# The hosted HttpExchange strips AgentId.name from discover() responses,
# so our fake transport returns UID-only AgentIds to mirror that. The
# bootstrap path constructs the recipient AgentId deterministically
# from (run_id, recipient_name) -- no name lookup happens.
# ---------------------------------------------------------------------------


class _FakeTransport:
    """``transport.discover()`` returns a fixed agent list."""
    def __init__(self, agents):
        self._agents = tuple(agents)

    async def discover(self, agent_class):  # noqa: ARG002 - sig match only
        return self._agents


class _FakeClient:
    def __init__(self, transport):
        self._transport = transport
        self.close = AsyncMock()


class _FakeFactory:
    def __init__(self, client):
        self._client = client

    async def create_user_client(self, *, name, start_listener):  # noqa: ARG002
        return self._client


class _FakeCampaign:
    """Minimal stand-in for ChemGraphCampaign with just what
    ``campaign_bootstrap_text`` reads. Avoids the full file-load path."""
    def __init__(self, user_task: str = 'do the thing'):
        self.user_task = user_task
        self.initial_agent = 'coordinator-agent'
        self.agents = ()
        self.resources = {}


def _seen_agent_id(name: str, run_id: str) -> AgentId[Any]:
    """Mirror what the hosted exchange returns from discover():
    deterministic UID, but with the name stripped to None."""
    aid = deterministic_agent_id(run_id=run_id, agent_name=name)
    return AgentId(uid=aid.uid, name=None, role='agent')


def test_dispatch_bootstrap_sends_one_message_to_deterministic_recipient(
    monkeypatch,
) -> None:
    """Happy path: recipient's mailbox is visible on the exchange,
    the wait succeeds, one Handle.action call gets made. The
    recipient AgentId is built deterministically from (run_id,
    recipient_name), NOT discovered by name."""
    run_id = 'demo-001'
    seen = _seen_agent_id('coordinator-agent', run_id)
    transport = _FakeTransport(agents=[
        seen,
        _seen_agent_id('some-other-campaign-agent', 'unrelated'),
    ])
    client = _FakeClient(transport)
    factory = _FakeFactory(client)

    sent: list[tuple[Any, str, dict]] = []

    class _RecordingHandle:
        def __init__(self, agent_id):
            self._agent_id = agent_id

        async def action(self, name, message):
            sent.append((self._agent_id, name, message))

    monkeypatch.setattr(bootstrap, 'Handle', _RecordingHandle)

    message_id = asyncio.run(
        bootstrap.dispatch_bootstrap(
            campaign=_FakeCampaign(),
            run_id=run_id,
            recipient='coordinator-agent',
            exchange_factory=factory,
            discover_timeout_s=1.0,
        ),
    )

    assert len(sent) == 1
    agent_id, action_name, message = sent[0]
    # Handle is built with the DETERMINISTIC AgentId -- same UID as
    # what the recipient daemon registered, so the message routes to
    # the right mailbox.
    expected_uid = deterministic_agent_id(
        run_id=run_id, agent_name='coordinator-agent',
    ).uid
    assert agent_id.uid == expected_uid
    assert action_name == 'receive_message'
    assert message['recipient'] == 'coordinator-agent'
    assert message['sender'] == 'campaign'
    assert message['message_id'] == message_id
    assert 'do the thing' in message['content']
    # Client must be closed on the happy path so we don't leak the
    # aiohttp session that backs the http exchange transport.
    client.close.assert_awaited_once()


def test_dispatch_bootstrap_closes_client_on_recipient_timeout(
    monkeypatch,
) -> None:
    """If the recipient's mailbox never appears on the exchange the
    helper must raise TimeoutError -- AND the client must still be
    closed so we don't leak the underlying network resources."""
    # Transport returns SOME unrelated agent but not our recipient.
    transport = _FakeTransport(agents=[
        _seen_agent_id('not-our-recipient', 'unrelated-run'),
    ])
    client = _FakeClient(transport)
    factory = _FakeFactory(client)

    monkeypatch.setattr(
        bootstrap, 'Handle',
        lambda agent_id: pytest.fail("Handle must not be built on timeout"),
    )

    with pytest.raises(TimeoutError):
        asyncio.run(
            bootstrap.dispatch_bootstrap(
                campaign=_FakeCampaign(),
                run_id='demo-001',
                recipient='coordinator-agent',
                exchange_factory=factory,
                discover_timeout_s=0.05,
            ),
        )
    client.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# main() -- end-to-end exit codes
# ---------------------------------------------------------------------------


def test_main_returns_2_on_recipient_timeout(monkeypatch, capsys) -> None:
    """Operators need a non-zero exit so wrapping shell scripts know
    bootstrap didn't actually happen. The stderr message should be the
    TimeoutError's text (which names the missing recipient)."""
    async def _raise(*args, **kwargs):
        raise TimeoutError("Timed out ... missing=['coordinator-agent']")
    monkeypatch.setattr(bootstrap, 'dispatch_bootstrap', _raise)
    monkeypatch.setattr(bootstrap, 'load_campaign',
                        lambda path: _FakeCampaign())
    monkeypatch.setattr(bootstrap, 'build_exchange_factory',
                        lambda config: None)

    code = bootstrap.main([
        '--campaign', 'mace-ensemble-screening-20',
        '--run-id', 'demo-001',
        '--exchange-type', 'local',
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert 'coordinator-agent' in err
