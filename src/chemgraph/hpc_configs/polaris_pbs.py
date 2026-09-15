"""Acquire Polaris allocations from a login node using Parsl's PBS provider."""

import os
from pathlib import Path
import shlex


def get_polaris_pbs_config(
    run_dir,
    *,
    account,
    worker_init,
    queue="debug",
    walltime="00:30:00",
    filesystems="home:eagle",
    nodes_per_block=1,
    max_blocks=1,
    max_workers_per_node=1,
    address=None,
):
    """Return a lazy, bounded PBS worker pool; no allocation is needed to start.

    ``worker_init`` must initialize the shared compute environment explicitly.
    Workers connect to ``address``, or the login node's ``bond0`` interface.
    Each worker gets one GPU and executes one complete calculation at a time.
    """
    for name, value in {
        "account": account,
        "queue": queue,
        "filesystems": filesystems,
        "walltime": walltime,
    }.items():
        if (
            not isinstance(value, str)
            or not value.strip()
            or any(c.isspace() for c in value)
        ):
            raise ValueError(f"{name} must be a nonempty PBS value without whitespace.")
    if not isinstance(worker_init, str) or not worker_init.strip():
        raise ValueError("PBS workers require an explicit worker_init script.")
    for name, value in {
        "nodes_per_block": nodes_per_block,
        "max_blocks": max_blocks,
        "max_workers_per_node": max_workers_per_node,
    }.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if max_workers_per_node > 4:
        raise ValueError("Polaris supports at most four GPU workers per node.")
    if os.environ.get("PBS_JOBID"):
        raise ValueError(
            "PBS allocation mode runs on a login node; use existing mode inside PBS."
        )

    from parsl import Config
    from parsl.addresses import address_by_interface
    from parsl.executors import HighThroughputExecutor
    from parsl.launchers import MpiExecLauncher
    from parsl.providers import PBSProProvider

    root = str(Path(run_dir).expanduser().resolve())
    return Config(
        run_dir=root,
        retries=0,
        max_idletime=120,
        executors=[
            HighThroughputExecutor(
                label="htex",
                address=address or address_by_interface("bond0"),
                available_accelerators=4,
                max_workers_per_node=max_workers_per_node,
                cpu_affinity="list:24-31,56-63:16-23,48-55:8-15,40-47:0-7,32-39",
                prefetch_capacity=0,
                provider=PBSProProvider(
                    account=account,
                    queue=queue,
                    walltime=walltime,
                    select_options="system=polaris:ngpus=4",
                    scheduler_options=f"#PBS -l filesystems={filesystems}\n#PBS -r n",
                    worker_init=(
                        f"set -e\nexport TMPDIR=/tmp\n{worker_init}\n"
                        f"cd {shlex.quote(root)}\nexport OMP_NUM_THREADS=8"
                    ),
                    nodes_per_block=nodes_per_block,
                    cpus_per_node=64,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=max_blocks,
                    launcher=MpiExecLauncher(
                        bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"
                    ),
                ),
            )
        ],
    )
