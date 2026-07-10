"""Command-line launcher for one academy_sim graph process."""

from __future__ import annotations

import argparse
import asyncio
import signal
from pathlib import Path
from typing import Any

from academy.handle import Handle
from academy.runtime import Runtime, RuntimeConfig

from chemgraph.academy_sim.agent import ChemGraphSimAgent
from chemgraph.academy_sim.artifacts import emit_event
from chemgraph.academy_sim.config import load_config
from chemgraph.academy_sim.envelopes import build_envelope
from chemgraph.academy_sim.exchange import build_exchange_factory
from chemgraph.academy_sim.peer_tools import build_peer_tools, peer_prompt_section
from chemgraph.academy_sim.registrations import (
    deterministic_graph_agent_id,
    http_agent_registration,
    publish_agent_id,
    register_http_agent_with_id,
    wait_for_agent_ids,
    wait_for_peer_uids,
)
from chemgraph.agent.graph_runtime import make_configured_graph_runner
from chemgraph.models.settings import load_lm_settings


async def run_graph_process(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    graph = config.graph(args.graph)
    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    exchange_factory = build_exchange_factory(config.exchange)
    registration, peer_handles = await _register_and_resolve_peers(
        exchange_factory=exchange_factory,
        config=config,
        graph=graph,
        run_dir=run_dir,
        run_token=args.run_token,
        launch_token=args.launch_token,
        startup_timeout_s=args.startup_timeout_s,
    )

    def trace(event: str, payload: dict[str, Any]) -> None:
        emit_event(
            run_dir,
            event=event,
            run_id=config.run_id,
            graph=graph.name,
            payload=payload,
        )

    done_event = asyncio.Event()
    peer_tools = (
        build_peer_tools(
            run_id=config.run_id,
            sender=graph.name,
            allowed_peers=graph.allowed_peers,
            peer_handles=peer_handles,
            trace=trace,
        )
        if graph.peer_tools.enabled
        else []
    )
    prompt_suffix = peer_prompt_section(
        graph_name=graph.name,
        allowed_peers=graph.allowed_peers if graph.peer_tools.enabled else (),
    )
    model_config = config.model_for_graph(graph.name)
    if model_config.config_file is None:
        raise RuntimeError(f'graph {graph.name!r} requires model.config_file')
    run_graph = await make_configured_graph_runner(
        spec=graph,
        llm_settings=load_lm_settings(model_config.config_file),
        extra_tools=peer_tools,
        prompt_suffix=prompt_suffix,
        trace=trace,
        log_dir=str(run_dir / 'cg_logs' / graph.name),
        terminal_tool_names=tuple(tool.name for tool in peer_tools),
    )
    agent = ChemGraphSimAgent(
        config=config,
        graph=graph,
        run_graph=run_graph,
        run_dir=run_dir,
        done_event=done_event,
    )
    runtime = Runtime(
        agent,
        exchange_factory=exchange_factory,
        registration=registration,
        config=RuntimeConfig(terminate_on_success=False, terminate_on_error=False),
    )
    async with runtime:
        if config.bootstrap_mode == 'inline' and graph.startup_prompt:
            self_handle: Handle[Any] = Handle(registration.agent_id)
            envelope = build_envelope(
                run_id=config.run_id,
                sender='startup',
                recipient=graph.name,
                content=graph.startup_prompt,
                metadata={'task': config.task},
            )
            await self_handle.action('receive_message', envelope.model_dump())
        await runtime.wait_shutdown()
    return 0


async def _register_and_resolve_peers(
    *,
    exchange_factory: Any,
    config: Any,
    graph: Any,
    run_dir: Path,
    run_token: str,
    launch_token: str,
    startup_timeout_s: float,
) -> tuple[Any, dict[str, Handle[Any]]]:
    if config.exchange.registration == 'exchange':
        return await _register_exchange_peers(
            exchange_factory=exchange_factory,
            config=config,
            graph=graph,
            startup_timeout_s=startup_timeout_s,
        )
    return await _register_file_peers(
        exchange_factory=exchange_factory,
        config=config,
        graph=graph,
        run_dir=run_dir,
        run_token=run_token,
        launch_token=launch_token,
        startup_timeout_s=startup_timeout_s,
    )


async def _register_file_peers(
    *,
    exchange_factory: Any,
    config: Any,
    graph: Any,
    run_dir: Path,
    run_token: str,
    launch_token: str,
    startup_timeout_s: float,
) -> tuple[Any, dict[str, Handle[Any]]]:
    registrar = await exchange_factory.create_user_client(
        name=f'{config.run_id}-{graph.name}-registrar',
        start_listener=False,
    )
    try:
        (registration,) = await registrar.register_agents(
            [(ChemGraphSimAgent, graph.name)]
        )
    finally:
        await registrar.close()

    publish_agent_id(
        run_dir=run_dir,
        run_id=config.run_id,
        run_token=run_token,
        launch_token=launch_token,
        exchange_type=config.exchange.type,
        graph=graph.name,
        agent_id=registration.agent_id,
    )

    peer_handles: dict[str, Handle[Any]] = {}
    if graph.peer_tools.enabled and graph.allowed_peers:
        peer_ids = await wait_for_agent_ids(
            run_dir,
            run_token=run_token,
            launch_token=launch_token,
            names=set(graph.allowed_peers),
            timeout_s=startup_timeout_s,
        )
        peer_handles = {name: Handle(peer_ids[name]) for name in graph.allowed_peers}
    return registration, peer_handles


async def _register_exchange_peers(
    *,
    exchange_factory: Any,
    config: Any,
    graph: Any,
    startup_timeout_s: float,
) -> tuple[Any, dict[str, Handle[Any]]]:
    if config.exchange.type != 'http':
        raise RuntimeError(
            'exchange.registration="exchange" currently requires '
            'exchange.type="http"'
        )
    registrar = await exchange_factory.create_user_client(
        name=f'{config.run_id}-{graph.name}-registrar',
        start_listener=False,
    )
    try:
        agent_id = deterministic_graph_agent_id(
            run_id=config.run_id,
            graph_name=graph.name,
        )
        await register_http_agent_with_id(
            transport=registrar._transport,
            agent_class=ChemGraphSimAgent,
            agent_id=agent_id,
        )
        registration = http_agent_registration(agent_id)
        peer_agent_ids = {
            peer: deterministic_graph_agent_id(
                run_id=config.run_id,
                graph_name=peer,
            )
            for peer in graph.allowed_peers
        }
        if graph.peer_tools.enabled and peer_agent_ids:
            await wait_for_peer_uids(
                registrar._transport,
                peer_agent_ids.values(),
                agent_class=ChemGraphSimAgent,
                timeout_s=startup_timeout_s,
            )
    finally:
        await registrar.close()

    peer_handles = {
        name: Handle(agent_id)
        for name, agent_id in peer_agent_ids.items()
        if graph.peer_tools.enabled
    }
    return registration, peer_handles


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run one academy_sim ChemGraph graph process.',
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--run-token', required=True)
    parser.add_argument(
        '--launch-token',
        help=(
            'Per-launch rendezvous token. Use the same fresh value for all '
            'graphs in one run to avoid stale peer AgentIds.'
        ),
    )
    parser.add_argument('--startup-timeout-s', type=float, default=120.0)
    args = parser.parse_args(argv)
    if args.launch_token is None:
        args.launch_token = args.run_token
    return args


async def _main_async(argv: list[str] | None = None) -> int:
    task = asyncio.create_task(run_graph_process(parse_args(argv)))
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, task.cancel)
        except (NotImplementedError, RuntimeError):
            pass
    try:
        return await task
    except asyncio.CancelledError:
        return 130


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main_async(argv))


if __name__ == '__main__':
    raise SystemExit(main())
