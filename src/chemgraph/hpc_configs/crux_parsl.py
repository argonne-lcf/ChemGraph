import os
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher


def get_crux_config(
    run_dir=None,
    max_workers_per_node: int = 16,
):
    """Create a Parsl configuration for ALCF Crux PBS jobs.

    Crux is a CPU-only AMD EPYC system (no accelerators).

    Parameters
    ----------
    run_dir : str, optional
        Directory used as Parsl's run directory and worker working directory.
    max_workers_per_node : int, optional
        Number of concurrent workers per node. Defaults to 16
        (≈8 cores per worker on a 128-core node).

    Returns
    -------
    parsl.config.Config
        Configured Parsl ``Config`` for Crux.
    """
    if run_dir is None:
        run_dir = os.getcwd()

    worker_init = (
        f"export TMPDIR=/tmp; cd {run_dir}; "
        "module load conda; conda activate base"
    )

    node_file = os.getenv("PBS_NODEFILE")
    if node_file and os.path.exists(node_file):
        with open(node_file, "r", encoding="utf-8") as f:
            node_list = f.readlines()
            num_nodes = len(node_list)
    else:
        raise ValueError(
            "PBS_NODEFILE not found. Cannot determine node count for Crux."
        )

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex",
                heartbeat_period=30,
                heartbeat_threshold=240,
                max_workers_per_node=max_workers_per_node,
                provider=LocalProvider(
                    nodes_per_block=num_nodes,
                    launcher=MpiExecLauncher(
                        bind_cmd="--cpu-bind", overrides="--ppn 1"
                    ),
                    init_blocks=1,
                    worker_init=worker_init,
                    max_blocks=1,
                    min_blocks=0,
                ),
            )
        ],
        run_dir=run_dir,
    )

    return config
