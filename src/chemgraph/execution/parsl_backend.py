"""Parsl execution backend.

Wraps `Parsl <https://parsl-project.org/>`_ to conform to the
:class:`ExecutionBackend` interface.  Python tasks are dispatched via
``@python_app`` and shell tasks via ``@bash_app``.

Parsl must be installed separately (``pip install chemgraph[parsl]``).
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import Any

from chemgraph.execution.base import ExecutionBackend, TaskSpec

logger = logging.getLogger(__name__)


class ParslBackend(ExecutionBackend):
    """Execution backend that delegates work to Parsl.

    Configuration
    -------------
    The ``system`` argument passed to :meth:`initialize` is forwarded to
    :func:`chemgraph.hpc_configs.loader.load_parsl_config` which returns
    the appropriate ``parsl.config.Config``.

    Extra ``kwargs`` are forwarded to the config loader (e.g.
    ``worker_init``).
    """

    def __init__(self) -> None:
        super().__init__()
        self._python_app = None
        self._bash_app = None
        self._is_async_remote = False
        self._executors = []

    @property
    def is_async_remote(self) -> bool:
        return self._is_async_remote

    def initialize(self, system: str = "polaris", **kwargs: Any) -> None:
        try:
            import parsl
            from parsl import bash_app, python_app
        except ImportError as exc:
            raise ImportError(
                "Parsl is required for the ParslBackend. "
                "Install it with: pip install chemgraph[parsl]"
            ) from exc

        from chemgraph.hpc_configs.loader import load_parsl_config

        config = load_parsl_config(system, **kwargs)
        parsl.load(config)
        self._is_async_remote = kwargs.get("allocation_mode") == "pbs"
        self._executors = config.executors

        # Create generic app wrappers ------------------------------------------
        # These are created once and reused for all submitted tasks.

        @python_app
        def _generic_python_app(fn, args, kwargs):
            """Execute an arbitrary callable on a Parsl worker."""
            return fn(*args, **kwargs)

        @bash_app
        def _generic_bash_app(command, stdout=None, stderr=None):
            """Execute a shell command string on a Parsl worker."""
            return command

        self._python_app = _generic_python_app
        self._bash_app = _generic_bash_app

        self._initialized = True
        logger.info("ParslBackend initialized for system '%s'", system)

    def submit(self, task: TaskSpec) -> Future:
        if not self._initialized:
            raise RuntimeError(
                "ParslBackend is not initialized. Call initialize() first."
            )

        if task.task_type == "python":
            if task.callable is None:
                raise ValueError(
                    f"Task '{task.task_id}': task_type='python' requires a callable."
                )
            from chemgraph.execution.utils import to_picklable

            return self._python_app(
                task.callable, to_picklable(task.args), to_picklable(task.kwargs)
            )

        elif task.task_type == "shell":
            if task.command is None:
                raise ValueError(
                    f"Task '{task.task_id}': task_type='shell' requires a command."
                )
            bash_kwargs: dict[str, Any] = {"command": task.command}
            if task.stdout:
                bash_kwargs["stdout"] = task.stdout
            if task.stderr:
                bash_kwargs["stderr"] = task.stderr
            return self._bash_app(**bash_kwargs)

        else:
            raise ValueError(
                f"Task '{task.task_id}': unsupported task_type '{task.task_type}'."
            )

    def get_execution_status(self) -> dict:
        """Report allocation IDs separately from ChemGraph calculation batches."""
        allocations = []
        for executor in self._executors:
            if not hasattr(executor, "blocks_to_job_id"):
                continue
            statuses = executor.status_facade
            for block_id, job_id in list(executor.blocks_to_job_id.items()):
                status = statuses.get(block_id)
                allocations.append({
                    "executor": executor.label, "block_id": block_id,
                    "scheduler_job_id": job_id,
                    "state": status.state.name if status else "UNKNOWN",
                    "message": status.message if status else None,
                })
        return {
            "backend": "parsl",
            "allocation_mode": "pbs" if self.is_async_remote else "existing",
            "status_source": "Parsl cached provider status",
            "allocations": allocations,
        }

    def shutdown(self) -> None:
        if self._initialized:
            try:
                import parsl

                # cleanup() stops executors and releases resources;
                # clear() only removes the DFK from the global registry.
                # Without cleanup(), Parsl logs
                # "Python is exiting with a DFK still running" at interpreter exit.
                try:
                    parsl.dfk().cleanup()
                except Exception:
                    logger.warning("Error during Parsl DFK cleanup.", exc_info=True)
                parsl.clear()
                logger.info("ParslBackend shut down.")
            except Exception:
                logger.warning("Error during Parsl shutdown.", exc_info=True)
        self._initialized = False
        self._executors = []
