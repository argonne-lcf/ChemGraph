"""Shared MCP tools for Globus Transfer file staging.

Call :func:`register_transfer_tools` to add ``list_transfer_facilities``,
``transfer_files``, ``check_transfer_status``, and ``list_remote_files`` to any
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
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from chemgraph.hpc_configs import list_facility_transfer_profiles

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from chemgraph.execution.globus_transfer import GlobusTransferManager

logger = logging.getLogger(__name__)

# One registry supplies both discovery metadata and validated tool choices.
TransferComputeSystem = Enum(
    "TransferComputeSystem",
    {profile.system: profile.system for profile in list_facility_transfer_profiles()},
    type=str,
)


def register_transfer_tools(
    mcp: FastMCP,
    transfer_manager: GlobusTransferManager | None,
) -> None:
    """Register Transfer discovery and operation tools on *mcp*.

    Parameters
    ----------
    mcp : FastMCP
        The MCP server to register tools on. May be a plain ``FastMCP``
        or a :class:`~chemgraph.mcp.cg_fastmcp.CGFastMCP`; ``add_tool``
        is inherited so the same registration works either way.
    transfer_manager : GlobusTransferManager, optional
        The configured defaults. Missing settings can be supplied in tool
        calls; operational tools validate configuration when called.
    """
    from chemgraph.execution.config import get_transfer_manager

    profiles = list_facility_transfer_profiles()
    task_managers: dict[str, GlobusTransferManager] = {}

    def resolve_manager(
        compute_system: TransferComputeSystem | None = None,
        destination_endpoint_id: str | None = None,
        source_endpoint_id: str | None = None,
    ) -> GlobusTransferManager:
        overrides = {}
        if compute_system is not None:
            overrides["system"] = TransferComputeSystem(compute_system).value
        for key, value in (
            ("destination_endpoint_id", destination_endpoint_id),
            ("source_endpoint_id", source_endpoint_id),
        ):
            if value is not None:
                if not value.strip():
                    raise ValueError(f"{key} must not be empty.")
                overrides[key] = value.strip()
        if transfer_manager is not None and not overrides:
            return transfer_manager
        manager = get_transfer_manager(
            default_manager=transfer_manager,
            allow_interactive_auth=False,
            **overrides,
        )
        if manager is None:
            raise ValueError(
                "Globus Transfer requires a source_endpoint_id, a destination "
                "(compute_system or destination_endpoint_id), and a configured "
                "destination_base_path (GLOBUS_TRANSFER_DESTINATION_BASE_PATH "
                "or [execution.globus_transfer] in config.toml). "
                "Supply missing endpoint selectors in the call or server defaults."
            )
        return manager

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
        """List supported Transfer facilities and the server's default target.

        Select a system per call with compute_system, or override its
        collection with destination_endpoint_id. Active flags describe defaults.
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
            "selection_mode": "per_call",
            "transfer_configured": transfer_manager is not None,
            "active_system": active_system,
            "facilities": facilities,
        }

    mcp.add_tool(
        list_transfer_facilities,
        name="list_transfer_facilities",
        description=(
            "List supported compute_system choices for Globus Transfer, "
            "their public collection/path metadata, and the server's default target."
        ),
    )

    def transfer_files(
        source_paths: Union[str, list[str]],
        extensions: Optional[list[str]] = None,
        remote_subdir: Optional[str] = None,
        wait: bool = True,
        label: Optional[str] = None,
        compute_system: Optional[TransferComputeSystem] = None,
        destination_endpoint_id: Optional[str] = None,
        source_endpoint_id: Optional[str] = None,
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
        compute_system : TransferComputeSystem, optional
            Destination from the supported system choices. Resolves its
            Transfer collection and paths; does not change the Compute endpoint.
        destination_endpoint_id : str, optional
            Destination collection UUID. Takes priority over compute_system.
        source_endpoint_id : str, optional
            Source collection UUID, overriding the server default. Source files
            must still be accessible locally to the MCP server.
        """
        manager = resolve_manager(
            compute_system, destination_endpoint_id, source_endpoint_id
        )
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

        transfer_result = manager.transfer_files(
            local_paths=files,
            remote_subdir=remote_subdir,
            label=label,
        )
        task_managers[transfer_result.task_id] = manager

        response = {
            "task_id": transfer_result.task_id,
            "source_endpoint_id": transfer_result.source_endpoint_id,
            "destination_endpoint_id": transfer_result.destination_endpoint_id,
            # Compute tools historically consume ``remote_directory``. Keep
            # that contract while exposing the collection path separately.
            "remote_directory": transfer_result.compute_directory,
            "transfer_directory": transfer_result.remote_directory,
            "file_count": len(files),
            "file_mapping": transfer_result.file_mapping,
            "compute_file_mapping": transfer_result.compute_file_mapping,
        }

        if wait:
            status = manager.wait_for_transfer(transfer_result.task_id)
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
        manager = task_managers.get(task_id) or resolve_manager()
        return manager.check_transfer_status(task_id)

    def list_remote_files(
        remote_path: str,
        compute_system: Optional[TransferComputeSystem] = None,
        destination_endpoint_id: Optional[str] = None,
    ) -> list[dict]:
        """List files using a collection-visible destination path.

        Useful to verify that files were staged correctly before
        running ensemble calculations. Pass ``transfer_directory`` from
        ``transfer_files`` when Transfer and compute path namespaces differ.
        Select a supported compute_system or supply destination_endpoint_id;
        an explicit ID takes priority. Omitted selectors use server defaults.
        """
        manager = resolve_manager(compute_system, destination_endpoint_id)
        return manager.list_remote_directory(remote_path)

    mcp.add_tool(
        transfer_files,
        name="transfer_files",
        description=(
            "Stage local files via Globus Transfer. Select a supported "
            "compute_system or supply destination_endpoint_id; an explicit ID "
            "takes priority. source_endpoint_id overrides the source collection. "
            "Omitted selectors use server defaults. "
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
            "transfer_directory and destination_endpoint_id returned by "
            "transfer_files, or select a supported compute_system."
        ),
    )
