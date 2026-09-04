"""Shared MCP tools for Globus Transfer file staging.

Call :func:`register_transfer_tools` to add ``list_transfer_facilities`` and,
when Transfer is configured, ``transfer_files``, ``check_transfer_status``,
and ``list_remote_files`` to any
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
    transfer_manager: GlobusTransferManager | None,
) -> None:
    """Register Transfer discovery and configured operation tools on *mcp*.

    Parameters
    ----------
    mcp : FastMCP
        The MCP server to register tools on. May be a plain ``FastMCP``
        or a :class:`~chemgraph.mcp.cg_fastmcp.CGFastMCP`; ``add_tool``
        is inherited so the same registration works either way.
    transfer_manager : GlobusTransferManager, optional
        The configured transfer manager instance. When omitted, facility
        discovery is still registered but operational tools are not.
    """
    from chemgraph.hpc_configs import list_facility_transfer_profiles

    profiles = list_facility_transfer_profiles()
    active_system = None
    if transfer_manager is not None:
        configured_system = getattr(transfer_manager, "system", None)
        if isinstance(configured_system, str):
            active_system = configured_system
        else:
            destination_id = getattr(
                transfer_manager,
                "destination_endpoint_id",
                None,
            )
            active_profile = next(
                (
                    profile
                    for profile in profiles
                    if profile.collection_id == destination_id
                ),
                None,
            )
            if active_profile is not None:
                active_system = active_profile.system

    def list_transfer_facilities() -> dict:
        """List supported Transfer facilities and the active server target.

        Facility selection is fixed when the MCP server starts. Reconfigure
        and restart the server to change the active Transfer destination.
        """
        facilities = []
        for profile in profiles:
            active = profile.system == active_system
            facilities.append(
                {
                    "system": profile.system,
                    "collection_name": profile.collection_name,
                    "collection_id": profile.collection_id,
                    "transfer_root": profile.transfer_root,
                    "compute_root": profile.compute_root,
                    "documentation_url": profile.documentation_url,
                    "verified_on": (
                        profile.verified_on.isoformat()
                        if profile.verified_on is not None
                        else None
                    ),
                    "active": active,
                    "uses_bundled_collection": bool(
                        active
                        and transfer_manager is not None
                        and transfer_manager.destination_endpoint_id
                        == profile.collection_id
                    ),
                }
            )
        return {
            "selection_mode": "server_configured",
            "transfer_configured": transfer_manager is not None,
            "active_system": active_system,
            "facilities": facilities,
        }

    mcp.add_tool(
        list_transfer_facilities,
        name="list_transfer_facilities",
        description=(
            "List supported Globus Transfer facilities (Polaris and Aurora), "
            "their public collection/path metadata, and the active "
            "server-configured target."
        ),
    )

    if transfer_manager is None:
        return

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
            # Compute tools historically consume ``remote_directory``. Keep
            # that contract while exposing the collection path separately.
            "remote_directory": transfer_result.compute_directory,
            "transfer_directory": transfer_result.remote_directory,
            "file_count": len(files),
            "file_mapping": transfer_result.file_mapping,
            "compute_file_mapping": transfer_result.compute_file_mapping,
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
        """List files using a collection-visible destination path.

        Useful to verify that files were staged correctly before
        running ensemble calculations. Pass ``transfer_directory`` from
        ``transfer_files`` when Transfer and compute path namespaces differ.
        """
        return transfer_manager.list_remote_directory(remote_path)

    mcp.add_tool(
        transfer_files,
        name="transfer_files",
        description=(
            "Transfer local files to the server-configured "
            f"{active_system or 'remote'} HPC filesystem via Globus Transfer. "
            "Use this to pre-stage structure files "
            "before running ensemble calculations with "
            "remote_structure_directory. Returns remote_directory for "
            "compute tools and transfer_directory for Transfer API calls."
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
            "List files in a destination collection directory. Pass the "
            "transfer_directory returned by transfer_files."
        ),
    )
