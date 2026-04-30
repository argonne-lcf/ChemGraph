"""Abstract base classes for execution backends.

This module defines the ``ExecutionBackend`` protocol and the ``TaskSpec``
data model that all workflow managers (Parsl, EnsembleLauncher, local
process pool, etc.) must implement.  Downstream code (MCP servers, tools)
only depends on these abstractions -- never on a concrete backend.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class TaskSpec(BaseModel):
    """Specification for a single unit of work to submit to a backend.

    Supports two execution modes:

    * **python** -- run a Python callable (``callable(*args, **kwargs)``)
    * **shell**  -- run a shell command string

    Resource hints (``num_nodes``, ``processes_per_node``, ``gpus_per_task``)
    are advisory; backends may ignore hints they do not support.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_id: str = Field(
        description="Unique identifier for this task within the batch."
    )
    task_type: Literal["python", "shell"] = Field(
        default="python",
        description="Execution mode: 'python' for a callable, 'shell' for a command.",
    )

    # ── Python task fields ──────────────────────────────────────────────
    callable: Optional[Callable[..., Any]] = Field(
        default=None,
        description="Python callable to execute (required when task_type='python').",
    )
    args: tuple = Field(
        default=(),
        description="Positional arguments forwarded to the callable.",
    )
    kwargs: dict = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the callable.",
    )

    # ── Shell task fields ───────────────────────────────────────────────
    command: Optional[str] = Field(
        default=None,
        description="Shell command to execute (required when task_type='shell').",
    )
    working_dir: Optional[str] = Field(
        default=None,
        description="Working directory for the shell command.",
    )
    stdout: Optional[str] = Field(
        default=None,
        description="Path to capture stdout (shell tasks).",
    )
    stderr: Optional[str] = Field(
        default=None,
        description="Path to capture stderr (shell tasks).",
    )

    # ── Resource hints ──────────────────────────────────────────────────
    num_nodes: int = Field(
        default=1,
        description="Number of compute nodes requested.",
    )
    processes_per_node: int = Field(
        default=1,
        description="Number of processes (ranks) per node.",
    )
    gpus_per_task: int = Field(
        default=0,
        description="Number of GPUs requested per task.",
    )


class ExecutionBackend(ABC):
    """Abstract interface that every workflow-manager adapter must implement.

    Lifecycle
    ---------
    1. ``initialize(system, **kwargs)``  -- start the backend
    2. ``submit(task)`` / ``submit_batch(tasks)``  -- dispatch work
    3. ``shutdown()``  -- release resources

    The class also supports the context-manager protocol (``with`` statement).
    """

    def __init__(self) -> None:
        self._initialized: bool = False

    @abstractmethod
    def initialize(self, system: str = "local", **kwargs: Any) -> None:
        """Prepare the backend for accepting work.

        Parameters
        ----------
        system : str
            Target HPC system name (e.g. ``"polaris"``, ``"aurora"``,
            ``"local"``).  Backends may use this to load system-specific
            configurations.
        **kwargs
            Backend-specific options (worker_init, run_dir, etc.).
        """

    @abstractmethod
    def submit(self, task: TaskSpec) -> Future:
        """Submit a single task and return a ``concurrent.futures.Future``.

        The future resolves to whatever the callable/command returns.
        """

    def submit_batch(self, tasks: list[TaskSpec]) -> list[Future]:
        """Submit multiple tasks, returning futures in submission order.

        The default implementation simply loops over ``submit()``.
        Backends may override this for optimized batch submission.
        """
        return [self.submit(t) for t in tasks]

    @abstractmethod
    def shutdown(self) -> None:
        """Release all resources held by the backend."""

    # ── Context-manager protocol ────────────────────────────────────────

    def __enter__(self) -> ExecutionBackend:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.shutdown()
