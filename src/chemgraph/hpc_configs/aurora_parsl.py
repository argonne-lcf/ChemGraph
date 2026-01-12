import os
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.addresses import address_by_interface


def get_aurora_config(
    run_dir=None,
):
    if run_dir is None:
        run_dir = os.getcwd()

    # Hard-wired worker_init for aurora
    worker_init = f"export TMPDIR=/tmp; cd {run_dir}; module load frameworks"

    # Get the number of nodes:
    node_file = os.getenv("PBS_NODEFILE")
    if node_file and os.path.exists(node_file):
        with open(node_file, "r") as f:
            node_list = f.readlines()
            num_nodes = len(node_list)
    else:
        # Fallback for testing/local runs without PBS
        raise ValueError("Warning: PBS_NODEFILE not found. Defaulting to 1 node.")
        num_nodes = 1

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex",
                heartbeat_period=30,
                heartbeat_threshold=240,
                available_accelerators=12,
                max_workers_per_node=9,
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
