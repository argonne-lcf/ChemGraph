from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from academy.identifier import AgentId

from chemgraph.academy.runtime import bootstrap


# ---------------------------------------------------------------------------
# parse_args -- CLI surface
# ---------------------------------------------------------------------------


def test_parse_args_requires_campaign() -> None:
    """``--campaign`` is the only field that doesn't have a default --
    bootstrap is useless without knowing which campaign's
    ``user_task`` to send."""
    with pytest.raises(SystemExit):
        bootstrap.parse_args([])


def test_parse_args_defaults_exchange_type_to_http() -> None:
    """Federated bootstrap is the main use case so http is the right
    default. Operators on single-machine runs can override but they
    rarely need this command at all (run-compute auto-bootstraps)."""
    args = bootstrap.parse_args(['--campaign', 'mace-ensemble-screening-20'])
    assert args.exchange_type == 'http'
    assert args.recipient is None  # defaults to campaign.initial_agent
    assert args.discover_timeout_s == pytest.approx(120.0)


def test_parse_args_accepts_recipient_override() -> None:
    """Operator should be able to bootstrap a non-initial agent for
    e.g. partial re-runs or debugging."""
    args = bootstrap.parse_args([
        '--campaign', 'foo.jsonc',
        '--recipient', 'worker-a',
    ])
    assert args.recipient == 'worker-a'


# ---------------------------------------------------------------------------
# dispatch_bootstrap -- the core async path
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


def test_dispatch_bootstrap_sends_one_message_to_discovered_recipient(
    monkeypatch,
) -> None:
    """Happy path: recipient is on the exchange, helper discovers them,
    one Handle.action call gets made, the message_id returned matches
    what was sent."""
    target = AgentId.new('coordinator-agent')
    transport = _FakeTransport(agents=[target, AgentId.new('other-agent')])
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
            recipient='coordinator-agent',
            exchange_factory=factory,
            discover_timeout_s=1.0,
        ),
    )

    assert len(sent) == 1
    agent_id, action_name, message = sent[0]
    assert agent_id is target
    assert action_name == 'receive_message'
    assert message['recipient'] == 'coordinator-agent'
    assert message['sender'] == 'campaign'
    assert message['message_id'] == message_id
    # The bootstrap content embeds the campaign's user_task; the
    # recipient agent's first round will parse this content.
    assert 'do the thing' in message['content']
    # Client must be closed on the happy path so we don't leak the
    # aiohttp session that backs the http exchange transport.
    client.close.assert_awaited_once()


def test_dispatch_bootstrap_closes_client_on_discover_timeout(
    monkeypatch,
) -> None:
    """If the recipient never appears on the exchange the helper must
    raise TimeoutError -- AND the client must still be closed so we
    don't leak the underlying network resources."""
    transport = _FakeTransport(agents=[AgentId.new('other-agent')])
    client = _FakeClient(transport)
    factory = _FakeFactory(client)

    monkeypatch.setattr(bootstrap, 'Handle',
                        lambda agent_id: pytest.fail("Handle must not be built on timeout"))

    with pytest.raises(TimeoutError):
        asyncio.run(
            bootstrap.dispatch_bootstrap(
                campaign=_FakeCampaign(),
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
        raise TimeoutError('Timed out ... missing=[\'coordinator-agent\']')
    monkeypatch.setattr(bootstrap, 'dispatch_bootstrap', _raise)
    # Bypass the campaign-file load to keep the test offline.
    monkeypatch.setattr(bootstrap, 'load_campaign',
                        lambda path: _FakeCampaign())
    monkeypatch.setattr(bootstrap, 'build_exchange_factory',
                        lambda config: None)

    code = bootstrap.main([
        '--campaign', 'mace-ensemble-screening-20',
        '--exchange-type', 'local',
    ])
    assert code == 2
    err = capsys.readouterr().err
    assert 'coordinator-agent' in err
