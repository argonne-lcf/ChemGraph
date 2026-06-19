"""Local Parsl configuration for development and single-node runs.

Uses ``HighThroughputExecutor`` with a ``LocalProvider`` (no MPI
launcher, no PBS/Slurm dependency).  Suitable for laptops, CI runners,
and single-node workstations where the Parsl backend is desired but no
HPC scheduler is available.
"""

from __future__ import annotations

import logging
import os

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider

from chemgraph.hpc_configs.loader import resolve_worker_init

logger = logging.getLogger(__name__)

_DEFAULT_MAX_WORKERS = 4


def get_local_config(
    run_dir: str | None = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    worker_init: str | None = None,
) -> Config:
    """Generate a Parsl configuration for local execution.

    Parameters
    ----------
    run_dir : str, optional
        Parsl run directory.  Defaults to the current working directory.
    max_workers : int, optional
        Maximum number of concurrent workers.  Default: 4.
    worker_init : str, optional
        Explicit shell snippet for worker init. When ``None`` (default),
        :func:`resolve_worker_init` picks ``CHEMGRAPH_WORKER_INIT`` /
        ``VIRTUAL_ENV`` / ``CONDA_PREFIX`` over a noop fallback.
    """
    if run_dir is None:
        run_dir = os.getcwd()

    if worker_init is None:
        worker_init = resolve_worker_init(run_dir, fallback="true")

    logger.info("Creating local Parsl config with %d workers", max_workers)

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="local_htex",
                max_workers_per_node=max_workers,
                provider=LocalProvider(
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1,
                    worker_init=worker_init,
                ),
            ),
        ],
        run_dir=run_dir,
    )

    return config
