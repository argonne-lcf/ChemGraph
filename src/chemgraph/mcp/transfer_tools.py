"""Shared MCP tools for Globus Transfer file staging.

Call :func:`register_transfer_tools` to add ``transfer_files``,
``check_transfer_status``, and ``list_remote_files`` to any
:class:`~mcp.server.fastmcp.FastMCP` (or
:class:`~chemgraph.mcp.cg_fastmcp.CGFastMCP`) server instance.

These tools allow an LLM agent to stage input files on a remote HPC
filesystem *before* submitting compute jobs, avoiding the overhead of
encoding large files inside Globus Compute function payloads.

Note
----
Transfer tools are orchestration tools (they call the Globus Transfer
API directly from the MCP server process), not compute tools, so they
are registered via :meth:`FastMCP.add_tool` rather than CGFastMCP's
backend-submitting ``@tool()`` decorator.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from chemgraph.execution.globus_transfer import GlobusTransferManager

logger = logging.getLogger(__name__)


def register_transfer_tools(
    mcp: FastMCP,
    transfer_manager: GlobusTransferManager,
) -> None:
    """Register file-transfer MCP tools on *mcp*.

    Parameters
    ----------
    mcp : FastMCP
        The MCP server to register tools on. May be a plain ``FastMCP``
        or a :class:`~chemgraph.mcp.cg_fastmcp.CGFastMCP`; ``add_tool``
        is inherited so the same registration works either way.
    transfer_manager : GlobusTransferManager
        The configured transfer manager instance.
    """

    def transfer_files(
        source_paths: Union[str, list[str]],
        extensions: Optional[list[str]] = None,
        remote_subdir: Optional[str] = None,
        wait: bool = True,
        label: Optional[str] = None,
    ) -> dict:
        """Transfer files to the remote HPC endpoint via Globus Transfer.

        Parameters
        ----------
        source_paths : str or list[str]
            A directory path (all matching files transferred) or a list
            of individual file paths.
        extensions : list[str], optional
            When *source_paths* is a directory, only transfer files with
            these extensions (e.g. ``[".cif", ".xyz"]``).  Ignored when
            *source_paths* is a list.
        remote_subdir : str, optional
            Subdirectory name on the remote endpoint.  Auto-generated if
            omitted.
        wait : bool
            If True (default), block until the transfer completes.
        label : str, optional
            Human-readable label for the transfer task.
        """
        if isinstance(source_paths, str):
            src = Path(source_paths)
            if src.is_dir():
                if extensions:
                    ext_set = {
                        e if e.startswith(".") else f".{e}" for e in extensions
                    }
                    files = sorted(
                        str(f)
                        for f in src.iterdir()
                        if f.is_file() and f.suffix.lower() in ext_set
                    )
                else:
                    files = sorted(
                        str(f) for f in src.iterdir() if f.is_file()
                    )
                if not files:
                    return {
                        "status": "error",
                        "message": f"No files found in {source_paths}"
                        + (
                            f" with extensions {extensions}"
                            if extensions
                            else ""
                        ),
                    }
            elif src.is_file():
                files = [str(src.resolve())]
            else:
                return {
                    "status": "error",
                    "message": f"Path not found: {source_paths}",
                }
        else:
            files = [str(Path(p).resolve()) for p in source_paths]

        transfer_result = transfer_manager.transfer_files(
            local_paths=files,
            remote_subdir=remote_subdir,
            label=label,
        )

        response = {
            "task_id": transfer_result.task_id,
            "remote_directory": transfer_result.remote_directory,
            "file_count": len(files),
            "file_mapping": transfer_result.file_mapping,
        }

        if wait:
            status = transfer_manager.wait_for_transfer(transfer_result.task_id)
            response["status"] = (
                "completed"
                if status["status"] == "SUCCEEDED"
                else status["status"]
            )
            response.update(
                {
                    k: status[k]
                    for k in ("bytes_transferred", "files_transferred")
                    if k in status
                }
            )
        else:
            response["status"] = "submitted"

        return response

    def check_transfer_status(task_id: str) -> dict:
        """Check the status of a Globus Transfer task.

        Use to poll a non-blocking transfer submitted with ``wait=False``.
        """
        return transfer_manager.check_transfer_status(task_id)

    def list_remote_files(remote_path: str) -> list[dict]:
        """List files in a directory on the remote HPC endpoint.

        Useful to verify that files were staged correctly before
        running ensemble calculations.
        """
        return transfer_manager.list_remote_directory(remote_path)

    mcp.add_tool(
        transfer_files,
        name="transfer_files",
        description=(
            "Transfer local files to the remote HPC filesystem via "
            "Globus Transfer. Use this to pre-stage structure files "
            "before running ensemble calculations with "
            "remote_structure_directory. Returns the remote directory "
            "path and a mapping of local-to-remote file paths."
        ),
    )
    mcp.add_tool(
        check_transfer_status,
        name="check_transfer_status",
        description=(
            "Check the status of a Globus Transfer task. Use this to "
            "poll a non-blocking transfer submitted with wait=False."
        ),
    )
    mcp.add_tool(
        list_remote_files,
        name="list_remote_files",
        description=(
            "List files in a directory on the remote HPC endpoint. "
            "Useful to verify that files were staged correctly before "
            "running ensemble calculations."
        ),
    )
