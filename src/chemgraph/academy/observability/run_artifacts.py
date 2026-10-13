from __future__ import annotations

import asyncio
import json
import pathlib
import shutil
import time
from typing import Any

from chemgraph.academy.observability.run_files import append_jsonl
from chemgraph.academy.observability.run_files import write_json
from chemgraph.academy.observability.run_files import write_json_atomic
from chemgraph.academy.core.campaign import ChemGraphAgentSpec
from chemgraph.academy.core.campaign import ChemGraphCampaign
from chemgraph.academy.core.campaign import ChemGraphDaemonConfig
from chemgraph.academy.runtime.mpi import append_system_trace
from chemgraph.models.settings import LLMSettings


def default_agent_state(spec: ChemGraphAgentSpec) -> dict[str, Any]:
    return {
        'agent_name': spec.name,
        'role': spec.role,
        'status_updated_at': None,
        'round': 0,
        'finished': False,
        'last_error': None,
    }


def write_status_snapshot(
    *,
    run_dir: pathlib.Path,
    campaign: ChemGraphCampaign,
    agent_state: dict[str, Any],
    placement: dict[str, Any],
) -> None:
    state_dir = run_dir / 'agent_status'
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(agent_state)
    payload['placement'] = placement
    write_json_atomic(state_dir / f'{agent_state["agent_name"]}.json', payload)

    states_by_agent: dict[str, dict[str, Any]] = {}
    for path in state_dir.glob('*.json'):
        try:
            item = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and isinstance(item.get('agent_name'), str):
            states_by_agent[item['agent_name']] = item

    agents = []
    placements = {}
    for spec in campaign.agents:
        state = states_by_agent.get(spec.name) or default_agent_state(spec)
        agents.append(state)
        if isinstance(state.get('placement'), dict):
            placements[spec.name] = state['placement']

    distinct_hosts = sorted(
        {
            item.get('short_hostname') or item.get('hostname')
            for item in placements.values()
            if item.get('short_hostname') or item.get('hostname')
        },
    )
    placement_doc = {
        'agents': placements,
        'distinct_hostnames': distinct_hosts,
        'distinct_hostname_count': len(distinct_hosts),
    }
    write_json_atomic(run_dir / 'placement.json', placement_doc)

    converged = bool(agents) and all(
        bool(item.get('finished')) for item in agents
    )
    status = {
        'timestamp': time.time(),
        'mode': 'mpi_daemon',
        'campaign_kind': 'chemgraph_agent_swarm',
        'campaign': campaign.run_id,
        'agents': sorted(agents, key=lambda item: item['agent_name']),
        'placement': placement_doc,
        'converged': converged,
    }
    write_json_atomic(run_dir / 'status.json', status)
    append_jsonl(run_dir / 'status_history.jsonl', status)


async def wait_for_agent_statuses_finished(
    *,
    run_dir: pathlib.Path,
    campaign: ChemGraphCampaign,
    timeout_s: float,
) -> bool:
    deadline = time.monotonic() + timeout_s
    state_dir = run_dir / 'agent_status'
    expected = {spec.name for spec in campaign.agents}
    while True:
        finished = set()
        for path in state_dir.glob('*.json'):
            try:
                item = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            if item.get('finished') is True and item.get('agent_name') in expected:
                finished.add(item['agent_name'])
        if finished == expected:
            return True
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.5)


def clear_run_outputs(run_dir: pathlib.Path) -> None:
    for name in (
        'academy_registrations.json',
        'messages.jsonl',
        'events.jsonl',
        'placement.json',
        'status.json',
        'status_history.jsonl',
        'tool_results.jsonl',
    ):
        path = run_dir / name
        if path.exists():
            path.unlink()
    for dirname in ('agent_status', 'artifacts', 'shared'):
        path = run_dir / dirname
        if path.exists():
            shutil.rmtree(path)


def initialize_run_files(
    *,
    run_dir: pathlib.Path,
    campaign: ChemGraphCampaign,
    config: ChemGraphDaemonConfig,
    llm_settings: LLMSettings,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    clear_run_outputs(run_dir)
    write_json(
        run_dir / 'manifest.json',
        {
            'run_dir': str(run_dir),
            'run_token': config.run_token,
            'mode': 'chemgraph_mpi_daemon',
            'agent_runtime': 'academy_runtime',
            'agent_count': config.agent_count,
            'max_decisions_per_agent': config.max_decisions,
            'campaign_config': (
                str(config.campaign_config)
                if config.campaign_config is not None
                else None
            ),
            'prompt_profile': str(campaign.prompt_profile),
            'chemgraph_repo_root': str(config.chemgraph_repo_root),
            'communication_transport': f'academy_{config.exchange_type}_actions',
            'exchange_type': config.exchange_type,
            'redis_host': config.redis_host,
            'redis_port': config.redis_port,
            'redis_namespace': config.redis_namespace,
            'llm_model': llm_settings.model,
            'llm_base_url': llm_settings.base_url,
            'llm_provider': llm_settings.provider,
            'llm_user': llm_settings.user,
        },
    )
    append_system_trace(
        run_dir,
        'campaign_started',
        {
            'mode': 'chemgraph_mpi_daemon',
            'agent_count': config.agent_count,
            'campaign': campaign.run_id,
        },
    )
    append_system_trace(
        run_dir,
        'campaign_planned',
        {
            'agents': [spec.name for spec in campaign.agents],
            'roles': {spec.name: spec.role for spec in campaign.agents},
            'mcp_servers': {
                spec.name: list(spec.mcp_servers)
                for spec in campaign.agents
            },
        },
    )
