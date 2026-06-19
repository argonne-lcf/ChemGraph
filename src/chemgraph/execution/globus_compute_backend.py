"""Globus Compute execution backend.

Wraps the `Globus Compute SDK <https://globus-compute.readthedocs.io/>`_
to conform to the :class:`ExecutionBackend` interface.  Python tasks are
dispatched via :meth:`Executor.submit` and shell tasks via
:class:`ShellFunction`.

Unlike the Parsl and EnsembleLauncher backends, Globus Compute does **not**
require an active PBS/Slurm allocation at submit time.  A persistent
Globus Compute *endpoint* daemon running on the HPC login node
automatically provisions and manages batch jobs as tasks arrive.

**Prerequisites**

1. Install the SDK: ``pip install chemgraphagent[globus_compute]``
2. On the HPC system, configure and start an endpoint::

       globus-compute-endpoint configure chemgraph-polaris
       globus-compute-endpoint start chemgraph-polaris
       # -> prints the endpoint UUID

3. Set ``endpoint_id`` in ``config.toml`` or pass it to
   :func:`~chemgraph.execution.config.get_backend`.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import Any

from chemgraph.execution.base import ExecutionBackend, TaskSpec

logger = logging.getLogger(__name__)


class GlobusComputeBackend(ExecutionBackend):
    """Execution backend that delegates work to Globus Compute.

    Configuration
    -------------
    The following ``kwargs`` are accepted by :meth:`initialize`:

    ``endpoint_id`` : str  **required**
        UUID of the Globus Compute endpoint to submit tasks to.
    ``amqp_port`` : int, optional
        Port for the AMQP result-streaming connection.  Defaults to the
        SDK default (5671).  Set to ``443`` if outbound 5671 is blocked.
    """

    def __init__(self) -> None:
        super().__init__()
        self._executor = None
        self._endpoint_id: str | None = None
        self._shares_filesystem = False

    @property
    def is_async_remote(self) -> bool:
        return True

    @property
    def shares_filesystem(self) -> bool:
        return self._shares_filesystem

    # ── lifecycle ────────────────────────────────────────────────────────

    def initialize(self, system: str = "local", **kwargs: Any) -> None:
        try:
            from globus_compute_sdk import Executor
        except ImportError as exc:
            raise ImportError(
                "globus-compute-sdk is required for the GlobusComputeBackend. "
                "Install it with: pip install chemgraphagent[globus_compute]"
            ) from exc

        endpoint_id = kwargs.get("endpoint_id")
        if not endpoint_id:
            raise ValueError(
                "GlobusComputeBackend requires an 'endpoint_id'. "
                "Set it in config.toml under [execution.globus_compute] "
                "or pass it directly to get_backend()."
            )

        executor_kwargs: dict[str, Any] = {"endpoint_id": endpoint_id}

        amqp_port = kwargs.get("amqp_port")
        if amqp_port is not None:
            executor_kwargs["amqp_port"] = int(amqp_port)

        # Opt-in: a Globus Compute endpoint that shares an HPC filesystem with
        # the MCP server can skip inline file embedding and read paths directly.
        self._shares_filesystem = bool(kwargs.get("shares_filesystem", False))

        self._endpoint_id = endpoint_id
        self._executor = Executor(**executor_kwargs)
        self._initialized = True
        logger.info(
            "GlobusComputeBackend initialized (system='%s', endpoint='%s')",
            system,
            endpoint_id,
        )

    # ── task submission ─────────────────────────────────────────────────

    def _ensure_executor(self) -> None:
        """Re-create the Executor if it was shut down (e.g. after a
        task failure)."""
        from globus_compute_sdk import Executor

        if self._executor is None or getattr(self._executor, "_stopped", False):
            logger.info("Re-creating Globus Compute Executor")
            self._executor = Executor(endpoint_id=self._endpoint_id)

    @staticmethod
    def _looks_like_stopped_executor(exc: BaseException) -> bool:
        """Heuristic: did a submit fail because the Executor is shut down?

        The SDK does not expose a stable exception type for this state;
        we match on common substrings observed in practice.
        """
        msg = str(exc).lower()
        return "shut down" in msg or "stopped" in msg or "closed" in msg

    def submit(self, task: TaskSpec) -> Future:
        if not self._initialized or self._executor is None:
            raise RuntimeError(
                "GlobusComputeBackend is not initialized. Call initialize() first."
            )

        self._ensure_executor()

        if task.task_type == "python":
            if task.callable is None:
                raise ValueError(
                    f"Task '{task.task_id}': task_type='python' requires a callable."
                )
            # Executor.submit() returns a ComputeFuture (a
            # concurrent.futures.Future subclass), fully compatible
            # with asyncio.wrap_future() used by gather_futures().
            try:
                return self._executor.submit(task.callable, *task.args, **task.kwargs)
            except Exception as exc:
                if not self._looks_like_stopped_executor(exc):
                    raise
                logger.warning(
                    "Submit raised %s -- rebuilding Globus Compute Executor "
                    "and retrying once.",
                    type(exc).__name__,
                )
                self._executor = None
                self._ensure_executor()
                return self._executor.submit(task.callable, *task.args, **task.kwargs)

        elif task.task_type == "shell":
            if task.command is None:
                raise ValueError(
                    f"Task '{task.task_id}': task_type='shell' requires a command."
                )
            from globus_compute_sdk import ShellFunction

            shell_fn = ShellFunction(task.command)
            try:
                return self._executor.submit(shell_fn)
            except Exception as exc:
                if not self._looks_like_stopped_executor(exc):
                    raise
                logger.warning(
                    "Submit raised %s -- rebuilding Globus Compute Executor "
                    "and retrying once.",
                    type(exc).__name__,
                )
                self._executor = None
                self._ensure_executor()
                return self._executor.submit(shell_fn)

        else:
            raise ValueError(
                f"Task '{task.task_id}': unsupported task_type '{task.task_type}'."
            )

    # ── health check ────────────────────────────────────────────────────

    def check_endpoint_status(self) -> dict:
        """Check the status of the configured Globus Compute endpoint.

        Returns a dict with ``endpoint_id`` and ``status`` fields.
        Useful as a pre-flight check before submitting tasks.
        """
        try:
            from globus_compute_sdk import Client

            client = Client()
            status = client.get_endpoint_status(self._endpoint_id)
            return {
                "endpoint_id": self._endpoint_id,
                "status": status,
            }
        except Exception as e:
            logger.warning(
                "Endpoint status check failed: %s", e, exc_info=True,
            )
            return {
                "endpoint_id": self._endpoint_id,
                "status": "error",
                "error": str(e),
            }

    # ── teardown ────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown()
                logger.info("GlobusComputeBackend shut down.")
            except Exception:
                logger.warning("Error during Globus Compute shutdown.", exc_info=True)
            self._executor = None
        self._initialized = False
