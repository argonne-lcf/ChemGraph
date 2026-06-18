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


def _register_fastmcp_dynamic_models() -> None:
    """Make pydantic models built by ``fastmcp.func_metadata`` pickle-by-qualname.

    FastMCP builds per-tool ``<tool>Arguments`` / ``<tool>Output`` classes via
    ``pydantic.create_model(__module__="mcp.server.fastmcp.utilities.func_metadata")``
    but never inserts them into that module's namespace. Dill's by-qualname
    lookup then fails and either raises ``PicklingError`` or falls back to
    pickle-by-value, which walks ``__globals__`` and can hit other surprises.
    Wrapping ``func_metadata`` so the resulting models are inserted into the
    module's ``__dict__`` makes the lookup succeed regardless of how the
    pickle graph reaches the class.
    """
    import sys

    from mcp.server.fastmcp.utilities import func_metadata as _fm

    if getattr(_fm, "_chemgraph_models_registered", False):
        return

    _orig = _fm.func_metadata
    _mod_ns = sys.modules[_fm.__name__].__dict__

    def _register(model):
        if model is None:
            return
        name = getattr(model, "__name__", None)
        if name and name not in _mod_ns:
            _mod_ns[name] = model
            try:
                model.__module__ = _fm.__name__
            except (AttributeError, TypeError):
                pass

    def _patched(*args, **kwargs):
        meta = _orig(*args, **kwargs)
        _register(getattr(meta, "arg_model", None))
        _register(getattr(meta, "output_model", None))
        return meta

    _fm.func_metadata = _patched
    # Several fastmcp modules captured the original via
    # ``from mcp.server.fastmcp.utilities.func_metadata import func_metadata``
    # before this patch ran, so they hold their own bound name. Rebind the
    # name in each known call site so every tool registration goes through
    # the wrapper.
    for _modname in (
        "mcp.server.fastmcp.tools.base",
        "mcp.server.fastmcp.prompts.base",
        "mcp.server.fastmcp.resources.templates",
    ):
        _m = sys.modules.get(_modname)
        if _m is not None and getattr(_m, "func_metadata", None) is _orig:
            _m.func_metadata = _patched

    _fm._chemgraph_models_registered = True


_register_fastmcp_dynamic_models()


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
        self._tracker_kwargs: dict[str, Any] = {}
        self._pre_submit_hook: Optional[Callable] = None
        self._task_counter: int = 0

    # ── Backend lifecycle ───────────────────────────────────────────────

    def init_backend(
        self,
        *,
        tracker_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Register backend configuration for lazy initialisation.

        The backend is not created until the first tool invocation,
        so the MCP server can start accepting connections immediately.

        Parameters
        ----------
        tracker_kwargs : dict, optional
            Forwarded to :class:`~chemgraph.execution.job_tracker.JobTracker`
            on first use. Use this to pass ``persist_file`` for cross-session
            job state recovery.
        **kwargs
            Forwarded to :func:`~chemgraph.execution.config.get_backend`.
        """
        self._backend_kwargs = kwargs
        self._tracker_kwargs = tracker_kwargs or {}
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
        self._tracker = JobTracker(**self._tracker_kwargs)
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
            self._tracker_kwargs = {}
            logger.info("CGFastMCP backend shut down.")

    # ── Pre-submit transport hook ──────────────────────────────────────

    def set_pre_submit_hook(self, hook: Optional[Callable]) -> None:
        """Register a hook that transforms each TaskSpec before submission.

        The hook receives the :class:`~chemgraph.execution.base.TaskSpec`
        and must return one (possibly the same instance). Used for
        transport concerns that should apply to every backend-submitted
        tool on this server -- e.g. embedding a local structure file
        into ``kwargs`` so a remote worker can materialise it, or
        rewriting a local path to a pre-staged remote path.

        Pass ``None`` to clear the hook.
        """
        self._pre_submit_hook = hook

    def _apply_pre_submit_hook(self, task):
        """Run the registered pre-submit hook (no-op when unset).

        Hook exceptions are wrapped in a ``ValueError`` naming the hook
        and the offending task_id, so they surface to the agent as a
        structured error instead of an opaque traceback.
        """
        if self._pre_submit_hook is None:
            return task
        try:
            return self._pre_submit_hook(task)
        except Exception as exc:
            hook_name = getattr(
                self._pre_submit_hook, "__name__", repr(self._pre_submit_hook)
            )
            task_id = getattr(task, "task_id", "<unknown>")
            logger.warning(
                "Pre-submit hook %s failed for task %s",
                hook_name,
                task_id,
                exc_info=True,
            )
            raise ValueError(
                f"Pre-submit hook '{hook_name}' failed for task '{task_id}': {exc}"
            ) from exc

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
        """Ensure *fn* is picklable when the MCP server runs as ``__main__``.

        Under ``python -m pkg.mod`` runpy sets ``__name__ == "__main__"``
        and populates both ``sys.modules["__main__"]`` and
        ``sys.modules["pkg.mod"]`` -- but it does **not** attach
        ``mod`` as an attribute of the parent package ``pkg``. Dill's
        by-qualname pickling resolves ``pkg.mod.fn`` via
        ``__import__("pkg", fromlist=["mod"])`` followed by
        ``getattr(pkg, "mod")``, which fails for that reason and silently
        falls back to pickle-by-value -- dragging the entire module's
        globals (including the FastMCP dynamic ``arg_model`` classes)
        into the byte stream.

        Three things must be true for dill to pickle ``fn`` by reference:

        1. ``fn.__module__`` points at the real dotted name (not ``__main__``).
        2. ``sys.modules[fn.__module__]`` exists and contains ``fn`` as
           an attribute.
        3. The parent package has the leaf module attached as an attribute
           (so ``getattr(pkg, leaf)`` resolves to the same module object).
        """
        if fn.__module__ == "__main__":
            import sys

            spec = getattr(sys.modules.get("__main__"), "__spec__", None)
            if spec and spec.name:
                fn.__module__ = spec.name
                target = sys.modules.get(spec.name)
                if target is None:
                    target = sys.modules["__main__"]
                    sys.modules[spec.name] = target
                elif getattr(target, fn.__name__, None) is not fn:
                    setattr(target, fn.__name__, fn)
                # Attach the leaf module to its parent package so dill's
                # ``__import__(parent, fromlist=[leaf])`` lookup succeeds.
                if "." in spec.name:
                    parent_name, _, leaf = spec.name.rpartition(".")
                    parent = sys.modules.get(parent_name)
                    if parent is not None and getattr(parent, leaf, None) is not target:
                        setattr(parent, leaf, target)

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
                from chemgraph.execution.utils import to_picklable

                self._ensure_backend()
                self._task_counter += 1
                batch_counter = self._task_counter
                pending = []
                for i, p in enumerate(params):
                    task = TaskSpec(
                        task_id=f"{fn.__name__}_{batch_counter}_{i}",
                        task_type="python",
                        callable=fn,
                        kwargs={param.name: to_picklable(p)},
                        **task_spec_kwargs,
                    )
                    task = self._apply_pre_submit_hook(task)
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

    # ── Schema-driven fanout tool ──────────────────────────────────────

    def schema_fanout_tool(
        self,
        *,
        worker: Callable,
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
        """Register a fan-out tool driven by a single *ensemble* schema.

        The decorated function is an **expander**: it receives the
        ensemble schema and returns a list of per-item arguments. The
        framework calls ``worker(item)`` on the backend for each item,
        gathers the results, and returns a batch summary -- same shape
        as :meth:`ensemble_tool`.

        Unlike :meth:`ensemble_tool` (whose tool signature is
        ``list[Schema]``), this preserves the ensemble schema as the
        agent-facing API, so the LLM makes a single tool call against
        e.g. ``input_structure_directory`` and server-side expansion
        produces the per-file jobs.

        Parameters
        ----------
        worker : Callable
            The per-item function executed on the backend. Must take
            a single positional argument (the item produced by the
            expander).
        name, description, annotations
            Passed through to :meth:`FastMCP.add_tool`.
        num_nodes, processes_per_node, gpus_per_task, env, working_dir
            Forwarded to each :class:`~chemgraph.execution.base.TaskSpec`.
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

        # Worker is what actually runs on the backend, so it must be
        # picklable from the MCP server's __main__ module.
        self._fix_module_for_pickle(worker)

        worker_sig = inspect.signature(worker)
        worker_params = list(worker_sig.parameters.values())
        if len(worker_params) != 1:
            raise TypeError(
                f"schema_fanout_tool worker must take exactly one "
                f"parameter, got {len(worker_params)} on "
                f"{worker.__qualname__}."
            )
        worker_param_name = worker_params[0].name

        def decorator(expander: Callable) -> Callable:
            sig = inspect.signature(expander)
            params = list(sig.parameters.values())
            if len(params) != 1:
                raise TypeError(
                    f"@schema_fanout_tool expander must take exactly one "
                    f"parameter (the ensemble schema), got {len(params)} "
                    f"on {expander.__qualname__}."
                )
            param = params[0]
            tool_name = name or expander.__name__

            async def wrapper(**kwargs):
                from chemgraph.execution.utils import to_picklable

                self._ensure_backend()
                self._task_counter += 1
                batch_counter = self._task_counter
                ensemble_params = kwargs[param.name]
                items = expander(ensemble_params)
                pending = []
                for i, item in enumerate(items):
                    task = TaskSpec(
                        task_id=f"{tool_name}_{batch_counter}_{i}",
                        task_type="python",
                        callable=worker,
                        kwargs={worker_param_name: to_picklable(item)},
                        **task_spec_kwargs,
                    )
                    task = self._apply_pre_submit_hook(task)
                    fut = self._backend.submit(task)
                    pending.append(({"index": i}, fut))

                return await submit_or_gather(
                    self._backend,
                    pending,
                    self._tracker,
                    tool_name,
                )

            wrapper.__name__ = tool_name
            wrapper.__doc__ = expander.__doc__
            wrapper.__module__ = expander.__module__
            wrapper.__qualname__ = expander.__qualname__
            # Preserve the expander's input signature so FastMCP advertises
            # the ensemble schema to the LLM, not the worker's per-item one.
            # The wrapper returns a submit_or_gather batch summary, though,
            # so it must not inherit the expander's list-of-jobs annotation.
            wrapper.__signature__ = sig.replace(
                return_annotation=dict[str, Any]
            )

            self.add_tool(wrapper, **fastmcp_kwargs)
            return expander

        return decorator

    # ── Internal ────────────────────────────────────────────────────────

    def _make_backend_wrapper(
        self, fn: Callable, task_spec_kwargs: dict[str, Any]
    ) -> Callable:
        """Build an async wrapper that submits *fn* to the backend."""
        from chemgraph.execution.base import TaskSpec
        from chemgraph.execution.utils import submit_or_gather, to_picklable

        self._fix_module_for_pickle(fn)

        @functools.wraps(fn)
        async def wrapper(**kwargs: Any) -> Any:
            self._ensure_backend()
            self._task_counter += 1
            task_id = f"{fn.__name__}_{self._task_counter}"
            task = TaskSpec(
                task_id=task_id,
                task_type="python",
                callable=fn,
                kwargs=to_picklable(kwargs),
                **task_spec_kwargs,
            )
            task = self._apply_pre_submit_hook(task)
            fut = self._backend.submit(task)

            if self._backend.is_async_remote:
                return await submit_or_gather(
                    self._backend,
                    [({"task_id": task_id}, fut)],
                    self._tracker,
                    fn.__name__,
                )

            return await asyncio.wrap_future(fut)

        return wrapper
