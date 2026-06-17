"""EnsembleLauncher execution backend.

Wraps `EnsembleLauncher <https://github.com/argonne-lcf/ensemble_launcher>`_
to conform to the :class:`ExecutionBackend` interface.  Uses the
cluster-mode API (``EnsembleLauncher.start()`` + ``ClusterClient``) so
that tasks can be submitted dynamically.

EnsembleLauncher must be installed separately
(``pip install chemgraphagent[ensemble_launcher]``).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import Future
from typing import List, Literal, Optional, Union

from chemgraph.execution.base import ExecutionBackend, TaskSpec

try:
    from ensemble_launcher import EnsembleLauncher
    from ensemble_launcher.config import (
        LauncherConfig,
        MPIConfig,
        PolicyConfig,
        SystemConfig,
    )
    from ensemble_launcher.helper_functions import get_nodes
    from ensemble_launcher.orchestrator import ClusterClient

    _ENSEMBLE_LAUNCHER_AVAILABLE = True
except ImportError:
    EnsembleLauncher = None
    LauncherConfig = None
    MPIConfig = None
    PolicyConfig = None
    SystemConfig = None
    get_nodes = None
    ClusterClient = None
    _ENSEMBLE_LAUNCHER_AVAILABLE = False

logger = logging.getLogger(__name__)


def _require_ensemble_launcher() -> None:
    if not _ENSEMBLE_LAUNCHER_AVAILABLE:
        raise ImportError(
            "EnsembleLauncher is required for the EnsembleLauncherBackend. "
            "Install it with: pip install ensemble-launcher"
        )


def get_local_system_config():
    _require_ensemble_launcher()
    system_config = SystemConfig(
        name="local",
        ncpus=os.cpu_count(),
        cpus=list(range(os.cpu_count())),
    )
    return system_config


def get_polaris_system_config():
    _require_ensemble_launcher()
    system_config = SystemConfig(
        name="polaris",
        ncpus=32,
        cpus=list(range(32)),
        ngpus=4,
        gpus=list(range(4)),
    )
    return system_config


def get_aurora_system_config():
    _require_ensemble_launcher()
    system_config = SystemConfig(
        name="aurora",
        ncpus=102,
        cpus=list(range(1, 52)) + list(range(53, 104)),
        ngpus=12,
        gpus=list(range(12)),
    )
    return system_config


def get_crux_system_config():
    _require_ensemble_launcher()
    system_config = SystemConfig(
        name="crux",
        ncpus=128,
        cpus=list(range(128)),
    )
    return system_config


def get_launcher_config(
    task_executor_name: Union[str, List] = "async_processpool",
    child_executor_policy: str = "fixed_leafs_children_policy",
    policy_config=None,
    checkpoint_dir=None,
    mpi_flavour: Literal[
        "test", "mpich", "intel", "cray-pals", "openmpi", "srun", "aprun", "jsrun"
    ] = "mpich",
):
    """Build a LauncherConfig.

    ``mpi_flavour`` defaults to ``"mpich"`` (hydra ``mpiexec``) which is the
    multi-node-safe choice for Aurora/Polaris/Crux. Use ``"test"`` only for
    single-node local runs — its ``write_file_to_nodes`` does not actually
    distribute child-spec JSON to remote ``/tmp``.
    """
    _require_ensemble_launcher()
    if policy_config is None:
        policy_config = PolicyConfig(nlevels=2, leaf_nodes=len(get_nodes()))
    if checkpoint_dir is None:
        checkpoint_dir = f"{os.getcwd()}/.ckpt_{uuid.uuid4().hex[:6]}"
    return LauncherConfig(
        child_executor_name="async_mpi",
        task_executor_name=task_executor_name,
        return_stdout=True,
        worker_logs=True,
        master_logs=True,
        children_scheduler_policy=child_executor_policy,
        policy_config=policy_config,
        cluster=True,
        checkpoint_dir=checkpoint_dir,
        mpi_config=MPIConfig(flavor=mpi_flavour),
    )


class EnsembleLauncherBackend(ExecutionBackend):
    """Execution backend that submits tasks through a :class:`ClusterClient`.

    Supports two initialization modes:

    **Client-only** — connect to a running EnsembleLauncher orchestrator::

        backend.initialize(checkpoint_dir="/path/to/running/el")

    **Managed** — start a local orchestrator, then connect::

        backend.initialize(system_config=..., launcher_config=...)

    In both modes the backend submits work through :class:`ClusterClient`.
    ``shutdown()`` tears down the client and, in managed mode, stops the
    orchestrator.
    """

    def __init__(self) -> None:
        _require_ensemble_launcher()
        super().__init__()
        self._orchestrator = None
        self._client = None

    def initialize(
        self,
        system: str = "local",
        *,
        client_only: bool = False,
        checkpoint_dir: Optional[str] = None,
        node_id: str = "global",
        system_config: Optional[SystemConfig] = None,
        launcher_config: Optional[LauncherConfig] = None,
        startup_delay: float = 10.0,
        **kwargs,
    ) -> None:
        """Prepare the backend for accepting work.

        Parameters
        ----------
        client_only : bool
            When ``True``, connect to a running orchestrator via
            *checkpoint_dir* — no orchestrator is started.
        checkpoint_dir : str
            Path to the orchestrator's checkpoint directory.  Required
            when *client_only* is ``True``.
        node_id : str
            Orchestrator node to connect to (default ``"global"``).
        system_config, launcher_config
            Required for **managed** mode (``client_only=False``).
            The backend starts its own orchestrator with these.
        startup_delay : float
            Seconds to wait for the orchestrator to become ready
            (managed mode only).
        """
        if client_only:
            # -- client-only mode ----------------------------------------------
            if checkpoint_dir is None:
                raise ValueError(
                    "client_only=True requires a checkpoint_dir pointing "
                    "to a running orchestrator."
                )
            self._client = ClusterClient(checkpoint_dir=checkpoint_dir, node_id=node_id)
            self._client.start()
            self._initialized = True
            logger.info(
                "EnsembleLauncherBackend initialized in client-only mode "
                "(checkpoint_dir='%s', node_id='%s')",
                checkpoint_dir,
                node_id,
            )
        else:
            # -- managed mode: start orchestrator first ------------------------
            if system_config is None or launcher_config is None:
                raise ValueError(
                    "Managed mode requires system_config and launcher_config "
                    "(or set client_only=True with a checkpoint_dir)."
                )
            os.makedirs(launcher_config.checkpoint_dir, exist_ok=True)
            with tempfile.TemporaryDirectory() as tmp_dir:
                launcher_config_fname = os.path.join(tmp_dir, "launcher_config.json")
                with open(launcher_config_fname, "w") as f:
                    f.write(launcher_config.model_dump_json())
                system_config_fname = os.path.join(tmp_dir, "system_config.json")
                with open(system_config_fname, "w") as f:
                    f.write(system_config.model_dump_json())
                cmd = [
                    "el",
                    "start",
                    "--system-config-file",
                    f"{system_config_fname}",
                    "--launcher-config-file",
                    f"{launcher_config_fname}",
                ]
                self._orchestrator = subprocess.Popen(
                    cmd,
                    stderr=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
                time.sleep(startup_delay)

                self._client = ClusterClient(
                    checkpoint_dir=launcher_config.checkpoint_dir,
                    node_id=node_id,
                )
                self._client.start()
                self._initialized = True
            logger.info(
                "EnsembleLauncherBackend initialized in managed mode "
                "(system='%s', comm='%s', executor='%s', nodes=%s)",
                system_config.name,
                launcher_config.comm_name,
                launcher_config.task_executor_name,
            )

    def submit(self, task: TaskSpec) -> Future:
        if not self._initialized or self._client is None:
            raise RuntimeError(
                "EnsembleLauncherBackend is not initialized. Call initialize() first."
            )

        from ensemble_launcher.ensemble import Task as ELTask

        if task.task_type == "python":
            if task.callable is None:
                raise ValueError(
                    f"Task '{task.task_id}': task_type='python' requires a callable."
                )
            el_task = ELTask(
                task_id=task.task_id,
                nnodes=task.num_nodes,
                ppn=task.processes_per_node,
                executable=task.callable,
                args=task.args or (),
                kwargs=task.kwargs or {},
                env=task.env,
            )
            return self._client.submit(el_task)

        elif task.task_type == "shell":
            if task.command is None:
                raise ValueError(
                    f"Task '{task.task_id}': task_type='shell' requires a command."
                )
            el_task = ELTask(
                task_id=task.task_id,
                nnodes=task.num_nodes,
                ppn=task.processes_per_node,
                executable=task.command,
                env=task.env,
            )
            return self._client.submit(el_task)

        else:
            raise ValueError(
                f"Task '{task.task_id}': unsupported task_type '{task.task_type}'."
            )

    def shutdown(self) -> None:
        self._initialized = False
        client_ok = True
        if self._client is not None:
            try:
                self._client.teardown()
                self._client = None
            except Exception:
                client_ok = False
                logger.warning(
                    "Error tearing down EnsembleLauncher client.", exc_info=True
                )

        orchestrator_ok = True
        if self._orchestrator is not None:
            try:
                self._orchestrator.terminate()
                self._orchestrator.wait(timeout=10.0)
            finally:
                if self._orchestrator.poll() is None:
                    self._orchestrator.kill()

        if client_ok and orchestrator_ok:
            logger.info("EnsembleLauncherBackend shut down.")
        else:
            logger.warning(
                "EnsembleLauncherBackend partially shut down. "
                "Call shutdown() again to retry failed teardown."
            )


_SYSTEM_CONFIG_BUILDERS = {
    "local": get_local_system_config,
    "aurora": get_aurora_system_config,
    "polaris": get_polaris_system_config,
    "crux": get_crux_system_config,
}


class _LazyRegistry:
    """Built-on-first-access mapping of system name -> SystemConfig.

    Avoids importing ``ensemble_launcher`` at module load time.
    """

    def __contains__(self, key: str) -> bool:
        return key in _SYSTEM_CONFIG_BUILDERS

    def __getitem__(self, key: str):
        if key not in _SYSTEM_CONFIG_BUILDERS:
            raise KeyError(key)
        return _SYSTEM_CONFIG_BUILDERS[key]()

    def keys(self):
        return _SYSTEM_CONFIG_BUILDERS.keys()


SYSTEM_CONFIG_REGISTRY = _LazyRegistry()
