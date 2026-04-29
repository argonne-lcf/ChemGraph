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
import socket
import time
import uuid
from concurrent.futures import Future
from typing import Any

from chemgraph.execution.base import ExecutionBackend, TaskSpec

logger = logging.getLogger(__name__)


class EnsembleLauncherBackend(ExecutionBackend):
    """Execution backend that delegates work to EnsembleLauncher.

    The backend starts an EnsembleLauncher orchestrator in cluster mode
    and submits tasks through a :class:`ClusterClient`.

    Configuration
    -------------
    The following ``kwargs`` are accepted by :meth:`initialize`:

    ``comm_name`` : str
        Communication backend (``"zmq"``, ``"async_zmq"``, ``"multiprocessing"``).
        Default: ``"async_zmq"``.
    ``task_executor_name`` : str
        Task executor (``"multiprocessing"``, ``"mpi"``,
        ``"async_processpool"``).  Default: ``"async_processpool"``.
    ``nlevels`` : int
        Hierarchy depth.  Default: ``0`` (single-node).
    ``max_workers`` : int
        Number of CPUs to expose.  Default: ``os.cpu_count()``.
    ``checkpoint_dir`` : str
        Directory for orchestrator checkpoint files.  Auto-generated
        when omitted.
    ``nodes`` : list[str]
        List of compute node hostnames.  Defaults to ``[hostname]``.
    ``startup_delay`` : float
        Seconds to wait after ``el.start()`` for the orchestrator to be
        ready.  Default: ``2.0``.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._el = None
        self._client = None
        self._checkpoint_dir: str | None = None

    def initialize(self, system: str = "local", **kwargs: Any) -> None:
        try:
            from ensemble_launcher import EnsembleLauncher
            from ensemble_launcher.config import LauncherConfig, SystemConfig
            from ensemble_launcher.orchestrator import ClusterClient
        except ImportError as exc:
            raise ImportError(
                "EnsembleLauncher is required for the EnsembleLauncherBackend. "
                "Install it with: pip install ensemble-launcher"
            ) from exc

        # -- extract parameters ------------------------------------------------
        comm_name = kwargs.get("comm_name", "async_zmq")
        task_executor = kwargs.get("task_executor_name", "async_processpool")
        nlevels = kwargs.get("nlevels", 0)
        ncpus = kwargs.get("max_workers", os.cpu_count() or 4)
        checkpoint_dir = kwargs.get(
            "checkpoint_dir",
            os.path.join(os.getcwd(), f".el_ckpt_{uuid.uuid4().hex[:8]}"),
        )
        nodes = kwargs.get("nodes", [socket.gethostname()])
        startup_delay = kwargs.get("startup_delay", 2.0)

        self._checkpoint_dir = checkpoint_dir

        # -- configure ---------------------------------------------------------
        system_config = SystemConfig(
            name=system,
            ncpus=ncpus,
            cpus=list(range(ncpus)),
        )

        launcher_config = LauncherConfig(
            task_executor_name=task_executor,
            comm_name=comm_name,
            nlevels=nlevels,
            cluster=True,
            checkpoint_dir=checkpoint_dir,
        )

        # -- start orchestrator ------------------------------------------------
        self._el = EnsembleLauncher(
            ensemble_file={},
            system_config=system_config,
            launcher_config=launcher_config,
            Nodes=nodes,
        )
        self._el.start()
        time.sleep(startup_delay)

        # -- connect client ----------------------------------------------------
        self._client = ClusterClient(checkpoint_dir=checkpoint_dir)
        self._client.start()

        self._initialized = True
        logger.info(
            "EnsembleLauncherBackend initialized (system='%s', "
            "comm='%s', executor='%s', nodes=%s)",
            system,
            comm_name,
            task_executor,
            nodes,
        )

    def submit(self, task: TaskSpec) -> Future:
        if not self._initialized or self._client is None:
            raise RuntimeError(
                "EnsembleLauncherBackend is not initialized. "
                "Call initialize() first."
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
                cmd_template=task.command,
            )
            return self._client.submit(el_task)

        else:
            raise ValueError(
                f"Task '{task.task_id}': unsupported task_type '{task.task_type}'."
            )

    def shutdown(self) -> None:
        if self._client is not None:
            try:
                self._client.teardown()
            except Exception:
                logger.warning(
                    "Error tearing down EnsembleLauncher client.", exc_info=True
                )
            self._client = None

        if self._el is not None:
            try:
                self._el.stop()
            except Exception:
                logger.warning(
                    "Error stopping EnsembleLauncher orchestrator.", exc_info=True
                )
            self._el = None

        self._initialized = False
        logger.info("EnsembleLauncherBackend shut down.")
