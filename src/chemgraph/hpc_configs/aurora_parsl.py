import os
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.addresses import address_by_interface


def get_aurora_config(
    run_dir=None,
    worker_init: str | None = None,
    max_workers_per_node: int | None = None,
):
    if run_dir is None:
        run_dir = os.getcwd()

    if worker_init is None:
        worker_init = f"export TMPDIR=/tmp; cd {run_dir}; module load frameworks"

    if max_workers_per_node is None:
        max_workers_per_node = int(
            os.getenv("CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE", "9")
        )

    # Get the number of nodes:
    node_file = os.getenv("PBS_NODEFILE")
    if node_file and os.path.exists(node_file):
        with open(node_file, "r") as f:
            node_list = f.readlines()
            num_nodes = len(node_list)
    else:
        raise ValueError(
            "PBS_NODEFILE not found. Cannot determine node count for Aurora."
        )

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex",
                heartbeat_period=30,
                heartbeat_threshold=240,
                available_accelerators=12,
                max_workers_per_node=max_workers_per_node,
                address=address_by_interface('bond0'),
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
