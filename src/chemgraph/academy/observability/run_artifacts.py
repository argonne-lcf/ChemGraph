from __future__ import annotations

import asyncio
import json
import pathlib
import shutil
import time
from collections import Counter
from typing import Any

from chemgraph.academy.observability.communication_proof import (
    build_communication_proof,
)
from chemgraph.academy.observability.event_log import CampaignEvent
from chemgraph.academy.observability.event_log import read_events
from chemgraph.academy.observability.run_files import append_jsonl
from chemgraph.academy.observability.run_files import write_json
from chemgraph.academy.observability.run_files import write_json_atomic
from chemgraph.academy.core.campaign import ChemGraphAgentSpec
from chemgraph.academy.core.campaign import ChemGraphCampaign
from chemgraph.academy.core.campaign import ChemGraphDaemonConfig
from chemgraph.academy.runtime.mpi import append_system_trace
from chemgraph.academy.core.lm import LLMSettings


def write_run_artifacts(run_dir: str | pathlib.Path) -> dict[str, Any]:
    """Write placement, communication proof, and summary artifacts."""
    root = pathlib.Path(run_dir)
    events = read_events(root / "events.jsonl")
    placement = build_placement(events, root / "status.json")
    proof = build_communication_proof(events, placement)
    summary = summarize_events(events)

    write_json(root / "placement.json", placement)
    write_json(root / "communication_proof.json", proof)
    write_json(root / "summary.json", summary)
    return {
        "placement": placement,
        "communication_proof": proof,
        "summary": summary,
    }


def build_placement(
    events: list[CampaignEvent],
    status_path: str | pathlib.Path | None = None,
) -> dict[str, Any]:
    """Build agent placement proof from events and latest status."""
    agents: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.event != "agent_started" or not event.agent_id:
            continue
        placement = event.payload.get("placement")
        if isinstance(placement, dict):
            agents[event.agent_id] = {
                "agent_id": event.agent_id,
                "role": event.role,
                **placement,
            }

    if status_path is not None:
        path = pathlib.Path(status_path)
        if path.exists():
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
                states = status.get("agent_states", {})
                if isinstance(states, dict):
                    for agent_id, state in states.items():
                        if not isinstance(state, dict):
                            continue
                        placement = state.get("placement")
                        if isinstance(placement, dict):
                            agents.setdefault(
                                agent_id,
                                {
                                    "agent_id": agent_id,
                                    "role": state.get("role"),
                                    **placement,
                                },
                            )
            except json.JSONDecodeError:
                pass

    hostnames = sorted(
        {
            str(record.get("hostname"))
            for record in agents.values()
            if record.get("hostname")
        },
    )
    return {
        "agent_count": len(agents),
        "hostnames": hostnames,
        "distinct_hostname_count": len(hostnames),
        "agents": dict(sorted(agents.items())),
    }


def summarize_events(events: list[CampaignEvent]) -> dict[str, Any]:
    """Return compact run summary from campaign events."""
    counts = Counter(event.event for event in events)
    final_reports = _final_reports(events)
    return {
        "event_count": len(events),
        "event_counts": dict(sorted(counts.items())),
        "finish": _last_payload(
            events,
            {"campaign_finished", "workflow_finished", "run_finished"},
        ),
        "agent_errors": _payloads_of(events, "agent_error"),
        "message_count": counts.get("message_sent", 0),
        "final_reports": final_reports,
        "tool_results": _tool_result_summaries(events),
    }


def _last_payload(
    events: list[CampaignEvent],
    kinds: set[str],
) -> dict[str, Any] | None:
    payloads = [event.payload for event in events if event.event in kinds]
    return payloads[-1] if payloads else None


def _payloads_of(events: list[CampaignEvent], kind: str) -> list[dict[str, Any]]:
    return [
        {
            "agent_id": event.agent_id,
            "role": event.role,
            **event.payload,
        }
        for event in events
        if event.event == kind
    ]


def _final_reports(events: list[CampaignEvent]) -> list[dict[str, Any]]:
    reports = []
    for event in events:
        payload = event.payload
        if event.event == "belief_updated":
            reports.append(
                {
                    "agent_id": event.agent_id,
                    "summary": payload.get("summary") or payload.get("hypothesis"),
                    "confidence": payload.get("confidence"),
                    "supporting_message_ids": payload.get("supporting_message_ids", []),
                    "supporting_tool_result_ids": payload.get(
                        "supporting_tool_result_ids",
                        [],
                    ),
                },
            )
    return reports[-10:]


def _tool_result_summaries(events: list[CampaignEvent]) -> list[dict[str, Any]]:
    results = []
    for event in events:
        if event.event != "tool_call_finished":
            continue
        payload = event.payload
        results.append(
            {
                "timestamp": event.timestamp,
                "agent_id": event.agent_id,
                "tool_name": payload.get("tool_name"),
                "tool_result_id": payload.get("tool_result_id"),
                "status": payload.get("status"),
                "content_preview": payload.get("content_preview"),
            },
        )
    return results


def default_agent_state(spec: ChemGraphAgentSpec) -> dict[str, Any]:
    return {
        'agent_name': spec.name,
        'role': spec.role,
        'status_updated_at': None,
        'round': 0,
        'finished': False,
        'last_error': None,
        'current_activity': None,
        'received_message_count': 0,
        'outbox_count': 0,
        'recent_received_messages': [],
        'recent_outbox': [],
        'tool_names': list(spec.tool_names),
        'tool_result_count': 0,
        'recent_tool_results': [],
        'belief': {
            'hypothesis': None,
            'confidence': 0.0,
            'supporting_message_ids': [],
            'supporting_tool_result_ids': [],
            'reason': None,
        },
        'belief_history': [],
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

    proof = build_communication_proof(
        read_events(run_dir / "events.jsonl"),
        placement_doc,
    )
    status = {
        'timestamp': time.time(),
        'mode': 'mpi_daemon',
        'campaign_kind': 'chemgraph_agent_swarm',
        'campaign': campaign.run_id,
        'agents': sorted(agents, key=lambda item: item['agent_name']),
        'communication_proof': proof,
        'placement': placement_doc,
        'converged': bool(proof.get('passes', {}).get('final_report')),
    }
    write_json_atomic(run_dir / 'status.json', status)
    append_jsonl(run_dir / 'status_history.jsonl', status)
    write_json_atomic(run_dir / 'communication_proof.json', proof)


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
        'communication_proof.json',
        'launch_plan.json',
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
        run_dir / 'campaign_private.json',
        {
            'run_id': campaign.run_id,
            'user_task': campaign.user_task,
            'initial_agent': campaign.initial_agent,
            'prompt_profile': str(campaign.prompt_profile),
            'resources': {
                name: spec.model_dump(exclude_none=True)
                for name, spec in campaign.resources.items()
            },
            'agents': [
                {
                    'name': spec.name,
                    'role': spec.role,
                    'mission': spec.mission,
                    'allowed_peers': list(spec.allowed_peers),
                    'tool_names': list(spec.tool_names),
                    'resources': list(spec.resources),
                }
                for spec in campaign.agents
            ],
        },
    )
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
            'communication_transport': 'academy_redis_actions',
            'redis_host': config.redis_host,
            'redis_port': config.redis_port,
            'redis_namespace': config.redis_namespace,
            'llm_model': llm_settings.model,
            'llm_base_url': llm_settings.base_url,
            'llm_provider': llm_settings.provider,
            'llm_user': llm_settings.user,
        },
    )
    write_json(
        run_dir / 'launch_plan.json',
        {
            'agent_class': 'ChemGraphLogicalAgent',
            'exchange': {
                'backend': 'academy_redis',
                'host': config.redis_host,
                'port': config.redis_port,
            },
            'placement': {
                'launcher': 'mpiexec',
                'agent_count': config.agent_count,
            },
            'agents': [
                {
                    'name': spec.name,
                    'role': spec.role,
                    'agent_class': 'ChemGraphLogicalAgent',
                    'allowed_peers': list(spec.allowed_peers),
                    'tool_names': list(spec.tool_names),
                }
                for spec in campaign.agents
            ],
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
            'tool_names': {
                spec.name: list(spec.tool_names)
                for spec in campaign.agents
            },
        },
    )
