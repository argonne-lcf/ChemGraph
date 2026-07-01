from __future__ import annotations

import os
import pathlib
import socket
import sys
from collections.abc import Mapping
from typing import Any

from chemgraph.academy.observability.event_log import EventLog
from chemgraph.academy.observability.run_files import write_json_atomic

MPI_RANK_ENV = (
    'PMI_RANK',
    'PMIX_RANK',
    'OMPI_COMM_WORLD_RANK',
    'PALS_RANK',
    'SLURM_PROCID',
)

MPI_LOCAL_RANK_ENV = (
    'MPI_LOCALRANKID',
    'PMI_LOCAL_RANK',
    'PMIX_LOCAL_RANK',
    'OMPI_COMM_WORLD_LOCAL_RANK',
    'PALS_LOCAL_RANK',
    'SLURM_LOCALID',
)


def rank_from_env(env: Mapping[str, str] | None = None) -> int:
    env = os.environ if env is None else env
    for name in MPI_RANK_ENV:
        value = env.get(name)
        if value is not None:
            return int(value)
    raise RuntimeError(
        'Could not determine MPI rank from environment. Expected one of '
        f'{", ".join(MPI_RANK_ENV)}. Run this through mpiexec.',
    )


def local_rank_from_env(env: Mapping[str, str] | None = None) -> int | None:
    env = os.environ if env is None else env
    for name in MPI_LOCAL_RANK_ENV:
        value = env.get(name)
        if value is not None:
            return int(value)
    return None


def append_system_trace(
    run_dir: pathlib.Path,
    event: str,
    payload: dict[str, Any],
) -> None:
    EventLog(run_dir / 'events.jsonl').emit(
        event,  # type: ignore[arg-type]
        run_id=run_dir.name,
        agent_id='system',
        payload=payload,
    )


def placement_payload(config: Any, agent_name: str) -> dict[str, Any]:
    host = socket.gethostname()
    pbs_keys = (
        'PBS_JOBID',
        'PBS_NODEFILE',
        'PBS_O_WORKDIR',
        'PBS_NCPUS',
        'PBS_NUM_NODES',
        'PBS_TASKNUM',
    )
    mpi_keys = (*MPI_RANK_ENV, *MPI_LOCAL_RANK_ENV)
    env = {
        key: os.environ[key]
        for key in (*pbs_keys, *mpi_keys)
        if key in os.environ
    }
    nodefile = os.environ.get('PBS_NODEFILE')
    nodes: list[str] = []
    if nodefile and pathlib.Path(nodefile).exists():
        nodes = [
            line.strip()
            for line in pathlib.Path(nodefile).read_text().splitlines()
            if line.strip()
        ]
    return {
        'agent_name': agent_name,
        'hostname': host,
        'short_hostname': host.split('.')[0],
        'pid': os.getpid(),
        'cwd': os.getcwd(),
        'python_executable': sys.executable,
        'rank': config.rank,
        'local_rank': config.local_rank,
        'exchange_type': config.exchange_type,
        'redis_host': config.redis_host,
        'redis_port': config.redis_port,
        'redis_namespace': config.redis_namespace,
        'env': env,
        'pbs_nodefile_nodes': nodes,
    }
