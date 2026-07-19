"""Shared utilities for ensemble execution in MCP servers.

Consolidates patterns that were previously duplicated across
``graspa_mcp_parsl.py``, ``xanes_mcp_parsl.py``, and
``mace_mcp_parsl.py``:

* Structure file resolution from directory or file list
* Async future gathering with error handling
* JSONL result writing
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    from chemgraph.execution.base import ExecutionBackend
    from chemgraph.execution.job_tracker import JobTracker

logger = logging.getLogger(__name__)


def to_picklable(value: Any) -> Any:
    """Recursively convert Pydantic instances to plain dicts.

    FastMCP's ``func_metadata`` builds tool-argument models with
    ``pydantic.create_model`` and a ``__module__`` that does not actually
    contain the class, so cloudpickle cannot serialize instances of those
    classes to a Parsl/Globus-Compute worker. Converting every Pydantic
    instance to a dict at the framework boundary side-steps the problem
    without patching the third-party library.

    Containers (``dict``, ``list``, ``tuple``) are walked recursively and
    rebuilt with the same shape; everything else passes through unchanged.
    """
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: to_picklable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_picklable(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_picklable(v) for v in value)
    return value


def resolve_structure_files(
    input_source: str | list[str],
    extensions: set[str] | None = None,
) -> tuple[list[Path], Path]:
    """Resolve a directory path or file list into a list of structure files.

    Parameters
    ----------
    input_source : str or list[str]
        Either a directory path (all matching files will be collected)
        or an explicit list of file paths.
    extensions : set[str], optional
        File extensions to include when scanning a directory (e.g.
        ``{".cif", ".xyz"}``).  If *None*, all files are included.

    Returns
    -------
    structure_files : list[Path]
        Sorted list of resolved file paths.
    output_dir : Path
        The parent directory (useful for placing output files).

    Raises
    ------
    ValueError
        If no files are found or if listed files do not exist.
    """
    # A bare relative filename (e.g. "water.cif") from a small model refers to
    # a file a sibling tool wrote into CHEMGRAPH_LOG_DIR, not the cwd. Resolve
    # each listed name against the log dir before checking existence so those
    # inputs still resolve; absolute/cwd paths are returned unchanged. The
    # directory branch is intentionally left untouched: writer tools emit
    # files (not directories) into the log dir, so there is no sibling-written
    # directory to fall back to.
    from chemgraph.tools.ase_core import _resolve_existing_path

    structure_files: list[Path] = []
    output_dir: Path = Path.cwd()

    if isinstance(input_source, list):
        structure_files = [Path(_resolve_existing_path(str(p))) for p in input_source]
        missing = [p for p in structure_files if not p.exists()]
        if missing:
            raise ValueError(f"The following input files are missing: {missing}")
        if structure_files:
            output_dir = structure_files[0].parent
    else:
        input_dir = Path(input_source)
        if not input_dir.is_dir():
            raise ValueError(f"'{input_dir}' is not a valid directory.")

        if extensions:
            structure_files = sorted(
                p for p in input_dir.iterdir() if p.suffix in extensions
            )
        else:
            structure_files = sorted(p for p in input_dir.iterdir() if p.is_file())

        output_dir = input_dir

    if not structure_files:
        raise ValueError("No structure files found to simulate.")

    return structure_files, output_dir


async def gather_futures(
    pending: list[tuple[dict, Future]],
    post_fn: Optional[Callable[[dict, Any], dict]] = None,
    timeout: Optional[float] = None,
) -> list[dict]:
    """Await a list of ``(metadata, future)`` pairs concurrently.

    Each future is converted to an asyncio-awaitable via
    :func:`asyncio.wrap_future` and gathered concurrently.

    Parameters
    ----------
    pending : list[tuple[dict, Future]]
        Each element is ``(task_metadata_dict, concurrent_futures_Future)``.
    post_fn : callable, optional
        If provided, called as ``post_fn(metadata, result)`` after a
        successful future resolution.  Must return a ``dict`` to include
        in the results list.  When *None*, the raw result is merged with
        metadata.
    timeout : float, optional
        Maximum seconds to wait for all futures to resolve.  If the
        timeout expires, an :class:`asyncio.TimeoutError` is raised.
        When *None* (default), wait indefinitely.

    Returns
    -------
    list[dict]
        One result dict per task (successful or failed).

    Raises
    ------
    asyncio.TimeoutError
        If *timeout* is set and exceeded before all futures complete.
    """

    async def _wait(meta: dict, fut: Future) -> dict:
        try:
            result = await asyncio.wrap_future(fut)
            if post_fn is not None:
                return post_fn(meta, result)
            # Default: merge metadata with result (if result is a dict)
            if isinstance(result, dict):
                merged = {**meta, **result}
                merged.setdefault("status", "success")
                return merged
            return {**meta, "result": result, "status": "success"}
        except Exception as e:
            return {
                **meta,
                "status": "failure",
                "error_type": type(e).__name__,
                "message": str(e),
            }

    coro = asyncio.gather(*(_wait(meta, fut) for meta, fut in pending))
    if timeout is not None:
        return list(await asyncio.wait_for(coro, timeout=timeout))
    return list(await coro)


async def submit_or_gather(
    backend: ExecutionBackend,
    pending: list[tuple[dict, Future]],
    tracker: JobTracker,
    tool_name: str,
    post_fn: Optional[Callable[[dict, Any], dict]] = None,
) -> dict:
    """Gather results or register for async tracking, depending on the backend.

    When ``backend.is_async_remote`` is ``True``, the pending futures are
    registered with the *tracker* and a submission confirmation is
    returned immediately (non-blocking).  Otherwise, results are gathered
    synchronously via :func:`gather_futures`.

    Parameters
    ----------
    backend : ExecutionBackend
        The active execution backend.
    pending : list[tuple[dict, Future]]
        Each element is ``(metadata_dict, future)``.
    tracker : JobTracker
        The job tracker instance to register batches with.
    tool_name : str
        Name of the MCP tool submitting the batch.
    post_fn : callable, optional
        Post-processing function for results.

    Returns
    -------
    dict
        Either ``{"status": "submitted", "batch_id": ..., ...}`` for
        async backends, or ``{"status": "completed", "results": ...}``
        for synchronous backends.
    """
    if backend.is_async_remote:
        batch_id = tracker.register_batch(tool_name, pending, post_fn=post_fn)
        return {
            "status": "submitted",
            "batch_id": batch_id,
            "n_tasks": len(pending),
            "message": (
                f"Submitted {len(pending)} task(s) to remote HPC endpoint. "
                f"Use check_job_status(batch_id='{batch_id}') to monitor "
                f"progress, and get_job_results(batch_id='{batch_id}') to "
                f"retrieve results once complete."
            ),
        }

    results = await gather_futures(pending, post_fn=post_fn)
    return {"status": "completed", "results": results}


def write_results_jsonl(
    results: list[dict],
    output_path: Path,
    append: bool = True,
) -> tuple[int, int]:
    """Write results to a JSONL file and return (success_count, total_count).

    Parameters
    ----------
    results : list[dict]
        Each dict should contain a ``"status"`` key.
    output_path : Path
        Path to the JSONL file.
    append : bool
        If *True* (default), append to an existing file.

    Returns
    -------
    success_count : int
    total_count : int
    """
    mode = "a" if append else "w"
    success_count = 0

    with open(output_path, mode, encoding="utf-8") as f:
        for res in results:
            if res.get("status") == "success":
                success_count += 1
            f.write(json.dumps(res) + "\n")

    return success_count, len(results)


def make_per_structure_output(
    struct_path: Path,
    base_output: Path,
) -> Path:
    """Generate a per-structure output filename.

    Given ``struct_path = "/data/MOF-5.cif"`` and
    ``base_output = "/results/output.json"``, returns
    ``"/results/MOF-5_output.json"``.
    """
    base_suffix = base_output.suffix or ".json"
    base_stem = base_output.stem
    return base_output.with_name(f"{struct_path.stem}_{base_stem}{base_suffix}")
