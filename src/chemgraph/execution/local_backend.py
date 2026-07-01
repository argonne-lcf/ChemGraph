"""Local execution backend using ``concurrent.futures.ProcessPoolExecutor``.

Ideal for development, testing, and single-node runs where no HPC
workflow manager is needed.  Requires zero external dependencies beyond
the Python standard library.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Any

from chemgraph.execution.base import ExecutionBackend, TaskSpec

logger = logging.getLogger(__name__)

# Default number of worker processes (can be overridden via config).
_DEFAULT_MAX_WORKERS = 4


def _silence_worker_stdout() -> None:
    """ProcessPoolExecutor *initializer*: redirect this worker's stdout fd to stderr.

    Used when ``LocalBackend`` runs inside a stdio MCP server, where the
    parent process's stdout is the JSON-RPC channel. Worker children inherit
    that fd by default, so any unguarded print (e.g. ``mace/tools/cg.py``'s
    "cuequivariance ... will be disabled" notice) corrupts the protocol
    stream. dup2 redirects this child's stdout fd to its stderr fd so prints
    are logged but never reach the client.
    """
    try:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):
        # Best-effort: skip silently if the fds aren't real (e.g. in some
        # test or notebook contexts where stderr is captured).
        pass


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
    import contextlib

    with (
        open(stdout_path, "w") if stdout_path else contextlib.nullcontext() as stdout_fh,
        open(stderr_path, "w") if stderr_path else contextlib.nullcontext() as stderr_fh,
    ):
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            stdout=stdout_fh,
            stderr=stderr_fh,
            check=True,
        )
        return result.returncode


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
        super().__init__()
        self._pool: ProcessPoolExecutor | None = None

    def initialize(self, system: str = "local", **kwargs: Any) -> None:
        max_workers = kwargs.get("max_workers", _DEFAULT_MAX_WORKERS)

        # Opt-in: silence worker stdout (redirect fd to stderr) so prints
        # from worker callables don't pollute a parent's stdout. Required
        # when LocalBackend runs under stdio MCP, where the parent's stdout
        # IS the JSON-RPC channel. Off by default so notebook/CLI users
        # still see prints. Explicit kwarg wins; otherwise env var.
        silence = kwargs.get("silence_worker_stdout")
        if silence is None:
            silence = os.environ.get("CHEMGRAPH_LOCAL_SILENCE_STDOUT") == "1"

        pool_kwargs: dict[str, Any] = {"max_workers": max_workers}
        if silence:
            pool_kwargs["initializer"] = _silence_worker_stdout

        self._pool = ProcessPoolExecutor(**pool_kwargs)
        self._initialized = True
        logger.info(
            "LocalBackend initialized with %d workers (silence_worker_stdout=%s)",
            max_workers,
            bool(silence),
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
