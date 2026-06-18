from __future__ import annotations

import asyncio
import json
import pathlib
import time
from collections.abc import Mapping
from typing import Any

from academy.exchange.hybrid import HybridAgentRegistration
from academy.exchange.local import LocalAgentRegistration
from academy.exchange.redis import RedisAgentRegistration
from academy.exchange.transport import AgentRegistration
from academy.identifier import AgentId
from pydantic import BaseModel

from chemgraph.academy.observability.run_files import write_json_atomic


_REGISTRATION_TYPES: dict[str, type[BaseModel]] = {
    'local': LocalAgentRegistration,
    'hybrid': HybridAgentRegistration,
    'redis': RedisAgentRegistration,
}


def academy_registration_path(run_dir: pathlib.Path) -> pathlib.Path:
    return run_dir / 'academy_registrations.json'


def _exchange_type_of(registration: AgentRegistration[Any]) -> str:
    value = getattr(registration, 'exchange_type', None)
    if not isinstance(value, str):
        raise TypeError(
            f'Registration {type(registration).__name__} has no string '
            '`exchange_type` field; cannot persist.',
        )
    return value


def registration_payload(
    *,
    run_token: str,
    registrations: Mapping[str, AgentRegistration[Any]],
) -> dict[str, Any]:
    if not registrations:
        raise ValueError('at least one registration is required')
    exchange_types = {_exchange_type_of(r) for r in registrations.values()}
    if len(exchange_types) > 1:
        raise ValueError(
            f'mixed exchange types in one campaign: {sorted(exchange_types)}',
        )
    (exchange_type,) = exchange_types
    return {
        'run_token': run_token,
        'exchange_type': exchange_type,
        'agents': {
            name: registration.agent_id.model_dump(mode='json')
            for name, registration in registrations.items()
        },
    }


def write_academy_registrations(
    *,
    run_dir: pathlib.Path,
    run_token: str,
    registrations: Mapping[str, AgentRegistration[Any]],
) -> None:
    write_json_atomic(
        academy_registration_path(run_dir),
        registration_payload(run_token=run_token, registrations=registrations),
    )


def load_academy_registrations(
    run_dir: pathlib.Path,
    *,
    run_token: str,
) -> dict[str, AgentRegistration[Any]]:
    path = academy_registration_path(run_dir)
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('run_token') != run_token:
        raise RuntimeError(
            f'Academy registration file {path} belongs to a different run',
        )
    exchange_type = data.get('exchange_type')
    if exchange_type not in _REGISTRATION_TYPES:
        raise RuntimeError(
            f'Academy registration file has unsupported exchange_type '
            f'{exchange_type!r}; expected one of '
            f'{sorted(_REGISTRATION_TYPES)}',
        )
    cls = _REGISTRATION_TYPES[exchange_type]
    agents = data.get('agents')
    if not isinstance(agents, dict):
        raise RuntimeError(f'Academy registration file is malformed: {path}')
    return {
        name: cls(agent_id=AgentId[Any].model_validate(agent_id))
        for name, agent_id in agents.items()
    }


async def wait_academy_registrations(
    run_dir: pathlib.Path,
    *,
    run_token: str,
    timeout_s: float,
) -> dict[str, AgentRegistration[Any]]:
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
