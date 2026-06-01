"""Backend-aware FastMCP subclass for ChemGraph.

:class:`CGFastMCP` extends :class:`FastMCP` with an execution backend.
Tools registered via :meth:`tool` are automatically submitted to the
backend as :class:`~chemgraph.execution.base.TaskSpec` instances —
the tool author writes a plain function and the framework handles
submission, future resolution, and async job tracking.

Tools that do **not** need the backend (e.g. JSON loaders, plotting
utilities) should be registered with :meth:`add_tool` (inherited from
FastMCP) which bypasses the backend wrapper entirely.
"""

import asyncio
import functools
import inspect
import logging
from typing import Any, Callable, Dict, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

logger = logging.getLogger(__name__)


class CGFastMCP(FastMCP):
    """FastMCP with an integrated execution backend.

    Parameters
    ----------
    **kwargs
        Forwarded to :class:`FastMCP` (``name``, ``instructions``, etc.).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backend = None
        self._tracker = None
        self._backend_kwargs: Optional[dict[str, Any]] = None

    # ── Backend lifecycle ───────────────────────────────────────────────

    def init_backend(self, **kwargs: Any) -> None:
        """Register backend configuration for lazy initialisation.

        The backend is not created until the first tool invocation,
        so the MCP server can start accepting connections immediately.
        All keyword arguments are forwarded to
        :func:`~chemgraph.execution.config.get_backend`.
        """
        self._backend_kwargs = kwargs
        self._register_job_tools()
        logger.info("CGFastMCP backend configured (lazy init).")

    def _ensure_backend(self) -> None:
        """Create the backend on first use."""
        if self._backend is not None:
            return
        if self._backend_kwargs is None:
            raise RuntimeError(
                "Backend not configured. Call init_backend() first."
            )
        from chemgraph.execution import JobTracker, get_backend

        self._backend = get_backend(**self._backend_kwargs)
        self._tracker = JobTracker()
        logger.info(
            "CGFastMCP backend initialised: %s", type(self._backend).__name__
        )

    def shutdown_backend(self) -> None:
        """Shut down the execution backend and release resources."""
        if self._backend is not None:
            try:
                self._backend.shutdown()
            except Exception:
                logger.warning("Error during backend shutdown.", exc_info=True)
            self._backend = None
            self._tracker = None
            self._backend_kwargs = None
            logger.info("CGFastMCP backend shut down.")

    # ── Job tracking tools ─────────────────────────────────────────────

    def _register_job_tools(self) -> None:
        """Register job-management tools (status, results, cancel)."""

        @self.add_tool
        def check_job_status(batch_id: str) -> dict:
            """Check the status of a submitted job batch."""
            self._ensure_backend()
            return self._tracker.get_status(batch_id)

        @self.add_tool
        def get_job_results(
            batch_id: str, include_partial: bool = False
        ) -> dict:
            """Retrieve results from a completed job batch."""
            self._ensure_backend()
            return self._tracker.get_results(
                batch_id, include_partial=include_partial
            )

        @self.add_tool
        def list_jobs() -> list[dict]:
            """List all tracked job batches."""
            self._ensure_backend()
            batches = self._tracker.list_batches()
            if not batches:
                return [{"message": "No job batches tracked."}]
            return batches

        @self.add_tool
        def cancel_job(batch_id: str) -> dict:
            """Cancel pending tasks in a job batch."""
            self._ensure_backend()
            return self._tracker.cancel_batch(batch_id)

        @self.add_tool
        def check_endpoint_status() -> dict:
            """Check whether the remote compute endpoint is reachable."""
            self._ensure_backend()
            if hasattr(self._backend, "check_endpoint_status"):
                return self._backend.check_endpoint_status()
            return {"status": "not_applicable",
                    "message": "This backend does not support endpoint status checks."}

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _fix_module_for_pickle(fn: Callable) -> None:
        """Ensure *fn* is picklable when the MCP server runs as ``__main__``."""
        if fn.__module__ == "__main__":
            import sys

            spec = getattr(sys.modules.get("__main__"), "__spec__", None)
            if spec and spec.name:
                fn.__module__ = spec.name
                if spec.name not in sys.modules:
                    sys.modules[spec.name] = sys.modules["__main__"]

    # ── Tool registration ───────────────────────────────────────────────

    def tool(
        self,
        name: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        annotations: Optional[ToolAnnotations] = None,
        structured_output: Optional[bool] = None,
        # ── TaskSpec resource hints ──────────────────────────────────
        num_nodes: int = 1,
        processes_per_node: int = 1,
        gpus_per_task: int = 0,
        env: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None,
    ) -> Callable:
        """Register a tool that runs on the execution backend.

        Same calling convention as :meth:`FastMCP.tool` — **parens are
        required** (``@mcp.tool()``, not ``@mcp.tool``).

        The additional parameters (``num_nodes``, ``processes_per_node``,
        ``gpus_per_task``, ``env``, ``working_dir``) are forwarded to the
        :class:`~chemgraph.execution.base.TaskSpec` that wraps the
        decorated function when it is invoked.

        Parameters
        ----------
        name, title, description, annotations, structured_output
            Passed through to :meth:`FastMCP.add_tool`.
        num_nodes : int
            Number of compute nodes (default ``1``).
        processes_per_node : int
            Processes per node (default ``1``).
        gpus_per_task : int
            GPUs per task (default ``0``).
        env : dict, optional
            Extra environment variables for the worker.
        working_dir : str, optional
            Working directory for the task.
        """
        fastmcp_kwargs: dict[str, Any] = {}
        if name is not None:
            fastmcp_kwargs["name"] = name
        if title is not None:
            fastmcp_kwargs["title"] = title
        if description is not None:
            fastmcp_kwargs["description"] = description
        if annotations is not None:
            fastmcp_kwargs["annotations"] = annotations
        if structured_output is not None:
            fastmcp_kwargs["structured_output"] = structured_output

        task_spec_kwargs: dict[str, Any] = {
            "num_nodes": num_nodes,
            "processes_per_node": processes_per_node,
            "gpus_per_task": gpus_per_task,
            "env": env or {},
        }
        if working_dir is not None:
            task_spec_kwargs["working_dir"] = working_dir

        def decorator(fn: Callable) -> Callable:
            wrapper = self._make_backend_wrapper(fn, task_spec_kwargs)
            self.add_tool(wrapper, **fastmcp_kwargs)
            return fn

        return decorator

    # ── Ensemble tool registration ─────────────────────────────────────

    def ensemble_tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        annotations: Optional[ToolAnnotations] = None,
        # ── TaskSpec resource hints ──────────────────────────────────
        num_nodes: int = 1,
        processes_per_node: int = 1,
        gpus_per_task: int = 0,
        env: Optional[Dict[str, str]] = None,
        working_dir: Optional[str] = None,
    ) -> Callable:
        """Register a fan-out tool that submits ``list[params]`` to the backend.

        Decorates ``fn(params: Schema) -> result``.  The MCP tool schema
        becomes ``list[Schema]`` — the LLM provides a list of jobs and
        the framework submits each as a
        :class:`~chemgraph.execution.base.TaskSpec`, then gathers results
        via :func:`~chemgraph.execution.utils.submit_or_gather`.

        Parameters
        ----------
        name, description, annotations
            Passed through to :meth:`FastMCP.add_tool`.
        num_nodes, processes_per_node, gpus_per_task, env, working_dir
            Forwarded to :class:`~chemgraph.execution.base.TaskSpec`.
        """
        from chemgraph.execution.base import TaskSpec
        from chemgraph.execution.utils import submit_or_gather

        task_spec_kwargs: dict[str, Any] = {
            "num_nodes": num_nodes,
            "processes_per_node": processes_per_node,
            "gpus_per_task": gpus_per_task,
            "env": env or {},
        }
        if working_dir is not None:
            task_spec_kwargs["working_dir"] = working_dir

        fastmcp_kwargs: dict[str, Any] = {}
        if name is not None:
            fastmcp_kwargs["name"] = name
        if description is not None:
            fastmcp_kwargs["description"] = description
        if annotations is not None:
            fastmcp_kwargs["annotations"] = annotations

        def decorator(fn: Callable) -> Callable:
            self._fix_module_for_pickle(fn)
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            if len(params) != 1:
                raise TypeError(
                    f"@ensemble_tool expects a function with exactly one "
                    f"parameter (the per-item schema), got {len(params)} "
                    f"on {fn.__qualname__}."
                )
            param = params[0]
            param_type = param.annotation

            async def wrapper(params):
                self._ensure_backend()
                pending = []
                for i, p in enumerate(params):
                    task = TaskSpec(
                        task_id=f"{fn.__name__}_{i}",
                        task_type="python",
                        callable=fn,
                        kwargs={param.name: p},
                        **task_spec_kwargs,
                    )
                    fut = self._backend.submit(task)
                    pending.append(({"index": i}, fut))

                return await submit_or_gather(
                    self._backend,
                    pending,
                    self._tracker,
                    name or fn.__name__,
                )

            wrapper.__name__ = name or fn.__name__
            wrapper.__doc__ = fn.__doc__
            wrapper.__module__ = fn.__module__
            wrapper.__qualname__ = fn.__qualname__

            new_param = inspect.Parameter(
                "params",
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=list[param_type],
            )
            wrapper.__signature__ = inspect.Signature(
                parameters=[new_param]
            )

            self.add_tool(wrapper, **fastmcp_kwargs)
            return fn

        return decorator

    # ── Internal ────────────────────────────────────────────────────────

    def _make_backend_wrapper(
        self, fn: Callable, task_spec_kwargs: dict[str, Any]
    ) -> Callable:
        """Build an async wrapper that submits *fn* to the backend."""
        from chemgraph.execution.base import TaskSpec
        from chemgraph.execution.utils import submit_or_gather

        self._fix_module_for_pickle(fn)

        @functools.wraps(fn)
        async def wrapper(**kwargs: Any) -> Any:
            self._ensure_backend()
            task = TaskSpec(
                task_id=fn.__name__,
                task_type="python",
                callable=fn,
                kwargs=kwargs,
                **task_spec_kwargs,
            )
            fut = self._backend.submit(task)

            if self._backend.is_async_remote:
                return await submit_or_gather(
                    self._backend,
                    [({"task_id": fn.__name__}, fut)],
                    self._tracker,
                    fn.__name__,
                )

            return await asyncio.wrap_future(fut)

        return wrapper
