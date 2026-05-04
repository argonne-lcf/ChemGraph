"""Parsl execution backend.

Wraps `Parsl <https://parsl-project.org/>`_ to conform to the
:class:`ExecutionBackend` interface.  Python tasks are dispatched via
``@python_app`` and shell tasks via ``@bash_app``.

Parsl must be installed separately (``pip install chemgraphagent[parsl]``).
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

    def initialize(self, system: str = "polaris", **kwargs: Any) -> None:
        try:
            import parsl
            from parsl import bash_app, python_app
        except ImportError as exc:
            raise ImportError(
                "Parsl is required for the ParslBackend. "
                "Install it with: pip install chemgraphagent[parsl]"
            ) from exc

        from chemgraph.hpc_configs.loader import load_parsl_config

        run_dir = kwargs.pop("run_dir", None)
        worker_init = kwargs.pop("worker_init", None)

        # Build kwargs for the config loader
        loader_kwargs: dict[str, Any] = {}
        if run_dir is not None:
            loader_kwargs["run_dir"] = run_dir
        if worker_init is not None:
            loader_kwargs["worker_init"] = worker_init

        config = load_parsl_config(system, **loader_kwargs)
        parsl.load(config)

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
            return self._python_app(task.callable, task.args, task.kwargs)

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

    def shutdown(self) -> None:
        if self._initialized:
            try:
                import parsl

                parsl.clear()
                logger.info("ParslBackend shut down.")
            except Exception:
                logger.warning("Error during Parsl shutdown.", exc_info=True)
        self._initialized = False
