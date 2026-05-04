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
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


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
    structure_files: list[Path] = []
    output_dir: Path = Path.cwd()

    if isinstance(input_source, list):
        structure_files = [Path(p) for p in input_source]
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

    Returns
    -------
    list[dict]
        One result dict per task (successful or failed).
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

    return list(
        await asyncio.gather(*(_wait(meta, fut) for meta, fut in pending))
    )


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
