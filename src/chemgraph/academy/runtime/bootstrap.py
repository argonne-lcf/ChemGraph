"""Standalone campaign-bootstrap dispatch for federated runs.

In single-machine campaigns rank 0 of the daemon dispatches the
``campaign`` -> ``initial_agent`` bootstrap message in-process as the
last step of startup. The federated ``spawn-site`` flow can't do
that: at startup time the agent that owns ``initial_agent`` may live
on a different machine that hasn't come up yet, so each site skips
the inline dispatch (``--no-bootstrap``) and the operator triggers
kickoff once every site is up by running ``chemgraph academy
bootstrap`` from anywhere with the cached Globus token.

This module is intentionally light: it does not load a system profile,
does not need a run-dir, and does not invoke ``mpiexec``. It just
opens an exchange user-client, discovers the recipient by name, and
sends one message.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from academy.handle import Handle

from chemgraph.academy.campaigns import resolve_campaign
from chemgraph.academy.core.agent import ChemGraphLogicalAgent
from chemgraph.academy.core.campaign import campaign_bootstrap_text
from chemgraph.academy.core.campaign import ChemGraphCampaign
from chemgraph.academy.core.campaign import ChemGraphDaemonConfig
from chemgraph.academy.core.campaign import load_campaign
from chemgraph.academy.core.campaign import namespace_for_run
from chemgraph.academy.core.peer_protocol import build_message
from chemgraph.academy.runtime.exchange import build_exchange_factory
from chemgraph.academy.runtime.exchange import SUPPORTED_EXCHANGE_TYPES
from chemgraph.academy.runtime.registration import deterministic_agent_id
from chemgraph.academy.runtime.registration import wait_for_peers_alive

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='chemgraph academy bootstrap',
        description=(
            'Dispatch the campaign bootstrap message to the initial agent. '
            'Run this once every site of a federated campaign is up; the '
            'recipient is looked up by name on the exchange.'
        ),
    )
    parser.add_argument(
        '--campaign', required=True,
        help='Campaign config (packaged name or path to campaign.jsonc).',
    )
    parser.add_argument(
        '--run-id', required=True,
        help=(
            "The run-id used by the spawn-site invocations. The bootstrap "
            "recipient's mailbox UID is derived deterministically from "
            "(run-id, agent-name); the same run-id must be passed here "
            "and to every spawn-site in the campaign."
        ),
    )
    parser.add_argument(
        '--recipient',
        help=(
            "Name of the agent that should receive the bootstrap "
            "message. Defaults to the campaign's ``initial_agent``."
        ),
    )
    parser.add_argument(
        '--exchange-type',
        choices=SUPPORTED_EXCHANGE_TYPES,
        default='http',
        help=(
            "Academy exchange backend. Defaults to 'http' since "
            "federated bootstrap is the main use case; pass 'local' / "
            "'redis' / 'hybrid' if you're re-bootstrapping a single-"
            "machine campaign for some reason."
        ),
    )
    parser.add_argument(
        '--http-exchange-url',
        help='Override URL for --exchange-type=http (defaults to Academy-hosted).',
    )
    parser.add_argument(
        '--redis-host', default='127.0.0.1',
        help='Redis host (only used for redis / hybrid exchanges).',
    )
    parser.add_argument(
        '--redis-port', type=int, default=6379,
        help='Redis port (only used for redis / hybrid exchanges).',
    )
    parser.add_argument(
        '--redis-namespace',
        help='Redis namespace (only used for hybrid; defaults from run-id).',
    )
    parser.add_argument(
        '--discover-timeout-s', type=float, default=600.0,
        help=(
            "How long to wait for the recipient agent's mailbox to be "
            "visible on the exchange. Defaults to 10 minutes to match "
            "spawn-site's startup_timeout_s; bump it higher if a "
            "federated site is unusually slow to come up."
        ),
    )
    return parser.parse_args(argv)


def _config_for_factory(args: argparse.Namespace) -> ChemGraphDaemonConfig:
    """Build the minimal DaemonConfig that ``build_exchange_factory`` reads.

    Most fields are unused for bootstrap and get throwaway values; what
    matters is ``exchange_type``, ``http_exchange_url``, and the redis
    triple. ``run_dir`` is a placeholder because the factory builder
    only consults a couple of fields.
    """
    run_dir = Path.cwd() / '.bootstrap-tmp'
    return ChemGraphDaemonConfig(
        run_dir=run_dir,
        run_token='bootstrap',
        agent_count=0,
        campaign_config=Path(args.campaign),
        lm_config=run_dir / 'lm.json',
        max_decisions=0,
        poll_timeout_s=1.0,
        idle_timeout_s=1.0,
        startup_timeout_s=args.discover_timeout_s,
        completion_timeout_s=1.0,
        status_interval_s=1.0,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_namespace=args.redis_namespace or namespace_for_run(run_dir),
        rank=0,
        local_rank=0,
        chemgraph_repo_root=Path.cwd(),
        exchange_type=args.exchange_type,
        http_exchange_url=args.http_exchange_url,
    )


async def dispatch_bootstrap(
    *,
    campaign: ChemGraphCampaign,
    run_id: str,
    recipient: str,
    exchange_factory: Any,
    discover_timeout_s: float,
) -> str:
    """Send the campaign bootstrap message to ``recipient`` over the exchange.

    Returns the dispatched message_id so the operator can correlate it
    with what shows up on the recipient site's event log.

    The recipient's AgentId is constructed deterministically from
    ``(run_id, recipient_name)`` -- same scheme spawn-site uses on
    the daemon side -- so no name-based discovery is needed (the
    hosted exchange strips names from discover() responses, which
    made the old discover-by-name approach silently fail).
    """
    client = await exchange_factory.create_user_client(
        name='chemgraph-bootstrap',
        start_listener=False,
    )
    # ``Handle.action`` reads its outbound exchange from a
    # ``ContextVar`` that ``UserExchangeClient.__aenter__`` sets to
    # self. Without entering the client as an async-context-manager
    # the contextvar stays unset and Handle.action raises
    # ``ExchangeClientNotFoundError``. The daemon-side path gets this
    # for free because Academy's Runtime enters the client; the
    # standalone bootstrap command has to do it explicitly.
    async with client:
        try:
            recipient_id = deterministic_agent_id(
                run_id=run_id, agent_name=recipient,
            )
            # Liveness probe: wait for the recipient's mailbox to
            # actually be registered on the exchange before sending.
            # Without this we'd happily POST a message to a mailbox
            # that doesn't exist yet -- the exchange would reject it.
            await wait_for_peers_alive(
                client._transport,
                [recipient_id],
                agent_class=ChemGraphLogicalAgent,
                timeout_s=discover_timeout_s,
            )

            message = build_message(
                sender='campaign',
                recipient=recipient,
                content=campaign_bootstrap_text(campaign),
                kind='message',
                tldr='Campaign bootstrap',
                reason='Initial campaign task dispatch (operator-triggered).',
                confidence=1.0,
            )
            handle: Handle[Any] = Handle(recipient_id)
            await handle.action('receive_message', message)
            logger.info(
                'Bootstrap message dispatched: recipient=%s message_id=%s',
                recipient, message['message_id'],
            )
            return message['message_id']
        finally:
            # __aexit__ does close + clear the contextvar; close()
            # here would double-close. The async with handles it.
            pass


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args(argv)

    campaign_path = resolve_campaign(args.campaign)
    if not campaign_path.exists():
        campaign_path = Path(args.campaign).resolve()
    campaign = load_campaign(campaign_path)

    recipient = args.recipient or campaign.initial_agent
    config = _config_for_factory(args)
    factory = build_exchange_factory(config)

    try:
        message_id = asyncio.run(
            dispatch_bootstrap(
                campaign=campaign,
                run_id=args.run_id,
                recipient=recipient,
                exchange_factory=factory,
                discover_timeout_s=args.discover_timeout_s,
            ),
        )
    except TimeoutError as exc:
        print(f'bootstrap failed: {exc}', file=sys.stderr)
        return 2
    print(f'ok: sent bootstrap to {recipient} (message_id={message_id})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
