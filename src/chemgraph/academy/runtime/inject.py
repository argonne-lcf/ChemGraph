"""Operator-originated message injection into an agent mailbox.

Single primitive for "send any message to any agent, at any time"
from an out-of-cluster process (the operator's laptop). Both
campaign kickoff and mid-run nudges go through this path; no
separate bootstrap concept.

Intentionally light: no system-profile load, no run-dir, no mpiexec.
Opens an exchange user-client, resolves the recipient's deterministic
mailbox uid, sends one message.
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
        prog='swarm inject',
        description=(
            'Inject an operator-originated message into an agent mailbox '
            'over the exchange. Used for both campaign kickoff and mid-run '
            'nudges -- one primitive, no separate bootstrap concept.'
        ),
    )
    parser.add_argument('--campaign', required=True)
    parser.add_argument(
        '--run-id', required=True,
        help=(
            "The run-id used by spawn-site. The recipient's mailbox UID "
            "is derived from (run-id, agent-name); must match spawn-site."
        ),
    )
    parser.add_argument('--recipient', required=True, help='Target agent name.')
    parser.add_argument('--content', required=True, help='Message content.')
    parser.add_argument(
        '--exchange-type',
        choices=SUPPORTED_EXCHANGE_TYPES,
        default='http',
    )
    parser.add_argument('--http-exchange-url')
    parser.add_argument('--redis-host', default='127.0.0.1')
    parser.add_argument('--redis-port', type=int, default=6379)
    parser.add_argument('--redis-namespace')
    parser.add_argument(
        '--discover-timeout-s', type=float, default=600.0,
        help=(
            "How long to wait for the recipient agent's mailbox to be "
            "visible on the exchange."
        ),
    )
    parser.add_argument('--tldr', default='operator message')
    parser.add_argument('--reason', default='operator-triggered inject')
    parser.add_argument('--sender', default='operator')
    parser.add_argument(
        '--kind', default='message',
        choices=('message', 'question', 'nudge'),
    )
    return parser.parse_args(argv)


def _config_for_factory(args: argparse.Namespace) -> ChemGraphDaemonConfig:
    """Minimal DaemonConfig for build_exchange_factory (most fields unused)."""
    run_dir = Path.cwd() / '.inject-tmp'
    return ChemGraphDaemonConfig(
        run_dir=run_dir,
        run_token='inject',
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


async def dispatch_message(
    *,
    run_id: str,
    recipient: str,
    content: str,
    exchange_factory: Any,
    discover_timeout_s: float,
    tldr: str = 'operator message',
    reason: str = 'operator-triggered inject',
    sender: str = 'operator',
    kind: str = 'message',
) -> str:
    """Send one operator message to ``recipient``. Returns the message_id.

    Recipient's AgentId is deterministic on (run_id, recipient) -- same
    scheme spawn-site uses on the daemon side, so no name-based
    discovery needed.
    """
    client = await exchange_factory.create_user_client(
        name='swarm-operator',
        start_listener=False,
    )
    async with client:
        recipient_id = deterministic_agent_id(
            run_id=run_id, agent_name=recipient,
        )
        # Liveness probe: wait for the recipient's mailbox to be
        # registered on the exchange before sending.
        await wait_for_peers_alive(
            client._transport,
            [recipient_id],
            agent_class=ChemGraphLogicalAgent,
            timeout_s=discover_timeout_s,
        )
        message = build_message(
            sender=sender,
            recipient=recipient,
            content=content,
            kind=kind,
            tldr=tldr,
            reason=reason,
            confidence=1.0,
        )
        handle: Handle[Any] = Handle(recipient_id)
        await handle.action('receive_message', message)
        logger.info(
            'inject dispatched: sender=%s recipient=%s message_id=%s',
            sender, recipient, message['message_id'],
        )
        return message['message_id']


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = parse_args(argv)

    # Campaign load is only used to validate the campaign exists; the
    # inject itself is purely (run_id, recipient, content). We keep
    # the flag so the CLI mirrors the launcher's argv shape.
    campaign_path = resolve_campaign(args.campaign)
    if not campaign_path.exists():
        campaign_path = Path(args.campaign).resolve()
    campaign = load_campaign(campaign_path)  # noqa: F841 -- validation only

    config = _config_for_factory(args)
    factory = build_exchange_factory(config)

    try:
        message_id = asyncio.run(
            dispatch_message(
                run_id=args.run_id,
                recipient=args.recipient,
                content=args.content,
                exchange_factory=factory,
                discover_timeout_s=args.discover_timeout_s,
                tldr=args.tldr,
                reason=args.reason,
                sender=args.sender,
                kind=args.kind,
            ),
        )
    except TimeoutError as exc:
        print(f'inject failed: {exc}', file=sys.stderr)
        return 2
    print(f'ok: sent to {args.recipient} (message_id={message_id})')
    return 0


if __name__ == '__main__':
    sys.exit(main())
