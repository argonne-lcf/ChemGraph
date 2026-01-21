import os
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher


def get_polaris_config(
    run_dir=None,
    worker_init: str = "export TMPDIR=/tmp",
):
    """
    Generates the Parsl configuration for the Polaris supercomputer.
    """
    if run_dir is None:
        run_dir = os.getcwd()

    # Get the number of nodes from the PBS environment
    node_file = os.getenv("PBS_NODEFILE")
    if node_file and os.path.exists(node_file):
        with open(node_file, "r") as f:
            node_list = f.readlines()
            num_nodes = len(node_list)
    else:
        # Fallback for testing/local runs without PBS
        print("Warning: PBS_NODEFILE not found. Defaulting to 1 node.")
        num_nodes = 1

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex",
                heartbeat_period=30,
                heartbeat_threshold=360,
                worker_debug=True,
                available_accelerators=4,
                cpu_affinity="list:24-31,56-63:16-23,48-55:8-15,40-47:0-7,32-39",
                prefetch_capacity=0,
                provider=LocalProvider(
                    launcher=MpiExecLauncher(
                        bind_cmd="--cpu-bind", overrides="--depth=1 --ppn 1"
                    ),
                    worker_init=worker_init,
                    nodes_per_block=num_nodes,
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1,
                ),
            ),
        ],
        run_dir=run_dir,
    )

    return config
