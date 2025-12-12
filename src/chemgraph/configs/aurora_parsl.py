import os
from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.utils import get_all_checkpoints


def get_polaris_config(
    run_dir=None,
    worker_init: str = "export TMPDIR=/tmp",
):
    tile_names = [f'{gid}.{tid}' for gid in range(6) for tid in range(2)]

    if run_dir is None:
        run_dir = os.getcwd()

    # Load previous checkpoints if they exist
    checkpoints = get_all_checkpoints(run_dir)

    # Get the number of nodes:
    node_file = os.getenv("PBS_NODEFILE")
    if node_file and os.path.exists(node_file):
        with open(node_file, "r") as f:
            node_list = f.readlines()
            num_nodes = len(node_list)
    else:
        # Fallback for testing/local runs without PBS
        print("Warning: PBS_NODEFILE not found. Defaulting to 1 node.")
        num_nodes = 1

    aurora_single_tile_config = Config(
        executors=[
            HighThroughputExecutor(
                available_accelerators=tile_names,
                max_workers_per_node=12,
                cpu_affinity="list:1-8,105-112:9-16,113-120:17-24,121-128:25-32,129-136:33-40,137-144:41-48,145-152:53-60,157-164:61-68,165-172:69-76,173-180:77-84,181-188:85-92,189-196:93-100,197-204",
                prefetch_capacity=0,
                provider=LocalProvider(
                    nodes_per_block=num_nodes,
                    launcher=MpiExecLauncher(
                        bind_cmd="--cpu-bind", overrides="--ppn 1"
                    ),
                    init_blocks=1,
                    worker_init=worker_init,
                    max_blocks=1,
                ),
            ),
        ],
        checkpoint_files=checkpoints,
        run_dir=run_dir,
        checkpoint_mode="task_exit",
        app_cache=True,
    )

    return aurora_single_tile_config
