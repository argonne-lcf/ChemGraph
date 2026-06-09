from __future__ import annotations

import asyncio
import json
import pathlib
import time
from collections.abc import Mapping
from typing import Any

from academy.exchange.redis import RedisAgentRegistration
from academy.identifier import AgentId

from chemgraph.academy.observability.run_files import write_json_atomic


def academy_registration_path(run_dir: pathlib.Path) -> pathlib.Path:
    return run_dir / 'academy_registrations.json'


def registration_payload(
    *,
    run_token: str,
    registrations: Mapping[str, RedisAgentRegistration[Any]],
) -> dict[str, Any]:
    return {
        'run_token': run_token,
        'exchange_type': 'redis',
        'agents': {
            name: registration.agent_id.model_dump(mode='json')
            for name, registration in registrations.items()
        },
    }


def write_academy_registrations(
    *,
    run_dir: pathlib.Path,
    run_token: str,
    registrations: Mapping[str, RedisAgentRegistration[Any]],
) -> None:
    write_json_atomic(
        academy_registration_path(run_dir),
        registration_payload(run_token=run_token, registrations=registrations),
    )


def load_academy_registrations(
    run_dir: pathlib.Path,
    *,
    run_token: str,
) -> dict[str, RedisAgentRegistration[Any]]:
    path = academy_registration_path(run_dir)
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('run_token') != run_token:
        raise RuntimeError(
            f'Academy registration file {path} belongs to a different run',
        )
    agents = data.get('agents')
    if not isinstance(agents, dict):
        raise RuntimeError(f'Academy registration file is malformed: {path}')
    return {
        name: RedisAgentRegistration(
            agent_id=AgentId[Any].model_validate(agent_id),
        )
        for name, agent_id in agents.items()
    }


async def wait_academy_registrations(
    run_dir: pathlib.Path,
    *,
    run_token: str,
    timeout_s: float,
) -> dict[str, RedisAgentRegistration[Any]]:
    path = academy_registration_path(run_dir)
    deadline = time.monotonic() + timeout_s
    while True:
        if path.exists():
            return load_academy_registrations(
                run_dir,
                run_token=run_token,
            )
        if time.monotonic() > deadline:
            raise TimeoutError(
                f'Timed out waiting for Academy registrations at {path}',
            )
        await asyncio.sleep(0.25)
