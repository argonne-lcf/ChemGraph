"""Local execution backend using ``concurrent.futures.ProcessPoolExecutor``.

Ideal for development, testing, and single-node runs where no HPC
workflow manager is needed.  Requires zero external dependencies beyond
the Python standard library.
"""

from __future__ import annotations

import logging
import subprocess
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

from chemgraph.execution.base import ExecutionBackend, TaskSpec

logger = logging.getLogger(__name__)

# Default number of worker processes (can be overridden via config).
_DEFAULT_MAX_WORKERS = 4


def _run_shell_task(
    command: str,
    working_dir: str | None,
    stdout_path: str | None,
    stderr_path: str | None,
) -> int:
    """Execute a shell command in a child process.

    Returns the process exit code.  stdout/stderr are captured to
    files when paths are provided.
    """
    stdout_fh = open(stdout_path, "w") if stdout_path else None  # noqa: SIM115
    stderr_fh = open(stderr_path, "w") if stderr_path else None  # noqa: SIM115
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            stdout=stdout_fh,
            stderr=stderr_fh,
            check=True,
        )
        return result.returncode
    finally:
        if stdout_fh:
            stdout_fh.close()
        if stderr_fh:
            stderr_fh.close()


def _run_python_task(
    fn: Any,  # Callable -- typed as Any for pickling
    args: tuple,
    kwargs: dict,
) -> Any:
    """Execute a Python callable in a child process."""
    return fn(*args, **kwargs)


class LocalBackend(ExecutionBackend):
    """Execution backend backed by :class:`ProcessPoolExecutor`.

    Configuration
    -------------
    ``max_workers`` : int
        Maximum number of concurrent worker processes (default: 4).
    """

    def __init__(self) -> None:
        self._pool: ProcessPoolExecutor | None = None
        self._initialized = False

    def initialize(self, system: str = "local", **kwargs: Any) -> None:
        max_workers = kwargs.get("max_workers", _DEFAULT_MAX_WORKERS)
        self._pool = ProcessPoolExecutor(max_workers=max_workers)
        self._initialized = True
        logger.info(
            "LocalBackend initialized with %d workers", max_workers
        )

    def submit(self, task: TaskSpec) -> Future:
        if not self._initialized or self._pool is None:
            raise RuntimeError(
                "LocalBackend is not initialized. Call initialize() first."
            )

        if task.task_type == "python":
            if task.callable is None:
                raise ValueError(
                    f"Task '{task.task_id}': task_type='python' requires a callable."
                )
            return self._pool.submit(
                _run_python_task, task.callable, task.args, task.kwargs
            )

        elif task.task_type == "shell":
            if task.command is None:
                raise ValueError(
                    f"Task '{task.task_id}': task_type='shell' requires a command."
                )
            return self._pool.submit(
                _run_shell_task,
                task.command,
                task.working_dir,
                task.stdout,
                task.stderr,
            )

        else:
            raise ValueError(
                f"Task '{task.task_id}': unsupported task_type '{task.task_type}'."
            )

    def shutdown(self) -> None:
        if self._pool is not None:
            logger.info("Shutting down LocalBackend process pool.")
            self._pool.shutdown(wait=True)
            self._pool = None
        self._initialized = False
