"""Manual startup message dispatch for academy_sim exchange rendezvous."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from chemgraph.academy_sim.agent import ChemGraphSimAgent
from chemgraph.academy_sim.config import AcademySimConfig, load_config
from chemgraph.academy_sim.envelopes import build_envelope
from chemgraph.academy_sim.exchange import build_exchange_factory
from chemgraph.academy_sim.registrations import (
    deterministic_graph_agent_id,
    wait_for_peer_uids,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Send the initial academy_sim message to one graph.',
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--recipient')
    parser.add_argument('--timeout-s', type=float, default=600.0)
    return parser.parse_args(argv)


async def dispatch_bootstrap(
    *,
    config: AcademySimConfig,
    recipient: str,
    timeout_s: float,
) -> str:
    if config.exchange.registration != 'exchange':
        raise RuntimeError(
            'academy_sim bootstrap requires exchange.registration="exchange"'
        )
    if config.exchange.type != 'http':
        raise RuntimeError(
            'academy_sim bootstrap currently requires exchange.type="http"'
        )

    graph = config.graph(recipient)
    content = graph.startup_prompt or config.task
    recipient_id = deterministic_graph_agent_id(
        run_id=config.run_id,
        graph_name=recipient,
    )
    exchange_factory = build_exchange_factory(config.exchange)
    client = await exchange_factory.create_user_client(
        name=f'{config.run_id}-bootstrap',
        start_listener=False,
    )
    async with client:
        from academy.handle import Handle

        await wait_for_peer_uids(
            client._transport,
            [recipient_id],
            agent_class=ChemGraphSimAgent,
            timeout_s=timeout_s,
        )
        envelope = build_envelope(
            run_id=config.run_id,
            sender='startup',
            recipient=recipient,
            content=content,
            metadata={'task': config.task, 'bootstrap': 'manual'},
        )
        handle: Handle[Any] = Handle(recipient_id)
        await handle.action('receive_message', envelope.model_dump())
    return envelope.message_id


async def _main_async(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    recipient = args.recipient or config.initial_graph
    if recipient is None:
        print(
            'bootstrap failed: pass --recipient or set initial_graph in config',
            file=sys.stderr,
        )
        return 2
    try:
        message_id = await dispatch_bootstrap(
            config=config,
            recipient=recipient,
            timeout_s=args.timeout_s,
        )
    except TimeoutError as exc:
        print(f'bootstrap failed: {exc}', file=sys.stderr)
        return 2
    print(f'ok: sent bootstrap to {recipient} (message_id={message_id})')
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main_async(argv))


if __name__ == '__main__':
    raise SystemExit(main())
