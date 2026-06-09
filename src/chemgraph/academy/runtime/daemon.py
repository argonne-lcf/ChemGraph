from __future__ import annotations

import argparse
import asyncio
import pathlib

from academy.exchange.redis import RedisExchangeFactory
from academy.handle import Handle
from academy.runtime import Runtime
from academy.runtime import RuntimeConfig

from chemgraph.academy.core.peer_protocol import build_message
from chemgraph.academy.runtime.registration import load_academy_registrations
from chemgraph.academy.runtime.registration import wait_academy_registrations
from chemgraph.academy.runtime.registration import write_academy_registrations
from chemgraph.academy.observability.run_artifacts import initialize_run_files
from chemgraph.academy.observability.run_artifacts import (
    wait_for_agent_statuses_finished,
)
from chemgraph.academy.observability.run_artifacts import write_status_snapshot
from chemgraph.academy.core.campaign import campaign_bootstrap_text
from chemgraph.academy.core.campaign import ChemGraphDaemonConfig
from chemgraph.academy.core.campaign import ExecutionSpec
from chemgraph.academy.core.campaign import load_campaign
from chemgraph.academy.core.campaign import namespace_for_run
from chemgraph.academy.core.campaign import resolve_campaign_resources
from chemgraph.academy.core.campaign import selected_agent
from chemgraph.academy.core.campaign import validate_campaign
from chemgraph.academy.examples import resolve_builtin_campaign
from chemgraph.academy.runtime.mpi import append_system_trace
from chemgraph.academy.runtime.mpi import local_rank_from_env
from chemgraph.academy.runtime.mpi import placement_payload
from chemgraph.academy.runtime.mpi import rank_from_env
from chemgraph.academy.core.agent import ChemGraphLogicalAgent
from chemgraph.academy.core.lm import load_lm_config
from chemgraph.academy.core.prompt import load_prompt_profile
from chemgraph.academy.core.fastmcp import (
    build_campaign_fastmcp_tool_invoker,
)


async def run_daemon(config: ChemGraphDaemonConfig) -> int:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    llm_settings = load_lm_config(config.lm_config)
    campaign = resolve_campaign_resources(
        load_campaign(config.campaign_config),
        config.run_dir,
    )
    prompt_profile = load_prompt_profile(campaign.prompt_profile)
    validate_campaign(campaign, config.agent_count)
    agent_spec = selected_agent(campaign, config.rank)
    placement = placement_payload(config, agent_spec.name)

    academy_factory = RedisExchangeFactory(
        hostname=config.redis_host,
        port=config.redis_port,
    )
    if config.rank == 0:
        initialize_run_files(
            run_dir=config.run_dir,
            campaign=campaign,
            config=config,
            llm_settings=llm_settings,
        )
        registrar = await academy_factory.create_user_client(
            name=f'{config.run_dir.name}-registrar',
            start_listener=False,
        )
        try:
            registered = await registrar.register_agents(
                [
                    (ChemGraphLogicalAgent, spec.name)
                    for spec in campaign.agents
                ],
            )
        finally:
            await registrar.close()
        registrations = dict(
            zip(
                (spec.name for spec in campaign.agents),
                registered,
                strict=True,
            ),
        )
        write_academy_registrations(
            run_dir=config.run_dir,
            run_token=config.run_token,
            registrations=registrations,
        )
    else:
        registrations = await wait_academy_registrations(
            config.run_dir,
            run_token=config.run_token,
            timeout_s=config.startup_timeout_s,
        )

    if config.rank == 0:
        registrations = load_academy_registrations(
            config.run_dir,
            run_token=config.run_token,
        )
    registration = registrations[agent_spec.name]
    peer_agent_ids = {
        peer: registrations[peer].agent_id
        for peer in agent_spec.allowed_peers
        if peer in registrations
    }

    tool_invoker = await build_campaign_fastmcp_tool_invoker(
        specs=list(agent_spec.tools),
        execution=ExecutionSpec(backend='local', system='local'),
        run_dir=config.run_dir,
        agent_name=agent_spec.name,
    )
    agent = ChemGraphLogicalAgent(
        agent_spec,
        campaign=campaign,
        llm_settings=llm_settings,
        prompt_profile=prompt_profile,
        run_dir=config.run_dir,
        max_decisions=config.max_decisions,
        tool_invoker=tool_invoker,
        peer_agent_ids=peer_agent_ids,
        placement=placement,
        poll_timeout_s=config.poll_timeout_s,
        idle_timeout_s=config.idle_timeout_s,
        status_interval_s=config.status_interval_s,
    )
    runtime_config = RuntimeConfig(
        terminate_on_success=False,
        terminate_on_error=False,
    )
    runtime = Runtime(
        agent,
        exchange_factory=academy_factory,
        registration=registration,
        config=runtime_config,
    )
    async with runtime:
        await agent.write_runtime_status()

        if config.rank == 0:
            bootstrap = build_message(
                sender='campaign',
                recipient=campaign.initial_agent,
                content=campaign_bootstrap_text(campaign),
                kind='message',
                tldr='Campaign bootstrap',
                reason='Initial campaign task dispatch.',
                confidence=1.0,
            )
            initial_handle: Handle[Any] = Handle(
                registrations[campaign.initial_agent].agent_id,
            )
            await initial_handle.action(
                'receive_message',
                bootstrap,
            )
            append_system_trace(
                config.run_dir,
                'bootstrap_message_dispatched',
                {
                    'agent': campaign.initial_agent,
                    'message_id': bootstrap['message_id'],
                    'via': 'academy_action',
                },
            )

        await runtime.wait_shutdown()

    if config.rank == 0:
        all_done = await wait_for_agent_statuses_finished(
            run_dir=config.run_dir,
            campaign=campaign,
            timeout_s=config.completion_timeout_s,
        )
        append_system_trace(
            config.run_dir,
            'campaign_finished',
            {'all_agents_done': all_done},
        )
        write_status_snapshot(
            run_dir=config.run_dir,
            campaign=campaign,
            agent_state=await agent.report_state(),
            placement=placement,
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run one persistent ChemGraph-style agent daemon.',
    )
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--run-token', required=True)
    parser.add_argument('--agent-count', type=int, default=5)
    parser.add_argument('--campaign-config', required=True)
    parser.add_argument('--lm-config', required=True)
    parser.add_argument('--max-decisions', type=int, default=6)
    parser.add_argument('--poll-timeout-s', type=float, default=2)
    parser.add_argument('--idle-timeout-s', type=float, default=600)
    parser.add_argument('--startup-timeout-s', type=float, default=120)
    parser.add_argument('--completion-timeout-s', type=float, default=60)
    parser.add_argument('--status-interval-s', type=float, default=5)
    parser.add_argument('--redis-host', default='127.0.0.1')
    parser.add_argument('--redis-port', type=int, required=True)
    parser.add_argument('--redis-namespace')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--local-rank', type=int)
    parser.add_argument('--no-clean-redis', action='store_true')
    parser.add_argument('--chemgraph-repo-root')
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ChemGraphDaemonConfig:
    run_dir = pathlib.Path(args.run_dir).resolve()
    resolved_campaign = resolve_builtin_campaign(args.campaign_config)
    campaign_config = (
        resolved_campaign.resolve()
        if resolved_campaign.exists()
        else pathlib.Path(args.campaign_config).resolve()
    )
    return ChemGraphDaemonConfig(
        run_dir=run_dir,
        run_token=args.run_token,
        agent_count=args.agent_count,
        campaign_config=campaign_config,
        lm_config=pathlib.Path(args.lm_config).resolve(),
        max_decisions=args.max_decisions,
        poll_timeout_s=args.poll_timeout_s,
        idle_timeout_s=args.idle_timeout_s,
        startup_timeout_s=args.startup_timeout_s,
        completion_timeout_s=args.completion_timeout_s,
        status_interval_s=args.status_interval_s,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_namespace=args.redis_namespace or namespace_for_run(run_dir),
        clean_redis=not args.no_clean_redis,
        rank=args.rank if args.rank is not None else rank_from_env(),
        local_rank=(
            args.local_rank
            if args.local_rank is not None
            else local_rank_from_env()
        ),
        chemgraph_repo_root=(
            pathlib.Path(args.chemgraph_repo_root).resolve()
            if args.chemgraph_repo_root
            else pathlib.Path.cwd().resolve()
        ),
    )


def main() -> int:
    return asyncio.run(run_daemon(config_from_args(parse_args())))


if __name__ == '__main__':
    raise SystemExit(main())
