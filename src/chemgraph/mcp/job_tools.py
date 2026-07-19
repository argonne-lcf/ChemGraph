"""Shared MCP tools for job status tracking and result retrieval.

Call :func:`register_job_tools` to add ``check_job_status``,
``get_job_results``, ``list_jobs``, ``cancel_job``, and (optionally)
``check_endpoint_status`` to any :class:`~mcp.server.fastmcp.FastMCP`
server instance.

These tools are only useful when the execution backend is async-remote
(e.g. Globus Compute), but are registered unconditionally so the LLM
agent always has a consistent tool surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from chemgraph.execution.base import ExecutionBackend
    from chemgraph.execution.job_tracker import JobTracker


def register_job_tools(
    mcp: FastMCP,
    tracker: JobTracker,
    backend: ExecutionBackend,
) -> None:
    """Register job-management MCP tools on *mcp*.

    Parameters
    ----------
    mcp : FastMCP
        The MCP server to register tools on.
    tracker : JobTracker
        The job tracker for this server process.
    backend : ExecutionBackend
        The active execution backend (used for endpoint health checks).
    """

    @mcp.tool(
        name="check_job_status",
        description=(
            "Check the status of a previously submitted HPC job batch. "
            "Returns progress information including how many tasks are "
            "complete, failed, or still pending. Use this to poll "
            "long-running remote compute jobs."
        ),
    )
    def check_job_status(batch_id: str) -> dict:
        """Check the status of a submitted job batch."""
        return tracker.get_status(batch_id)

    @mcp.tool(
        name="get_job_results",
        description=(
            "Retrieve results from a completed (or partially completed) "
            "HPC job batch. By default, returns results only when all "
            "tasks are done. Set include_partial=True to get results "
            "for tasks that have finished so far."
        ),
    )
    def get_job_results(
        batch_id: str,
        include_partial: bool = False,
    ) -> dict:
        """Retrieve results from a job batch."""
        return tracker.get_results(batch_id, include_partial=include_partial)

    @mcp.tool(
        name="list_jobs",
        description=(
            "List all tracked job batches with their current status. "
            "Shows batch IDs, tool names, submission times, and progress."
        ),
    )
    def list_jobs() -> list[dict]:
        """List all tracked job batches."""
        batches = tracker.list_batches()
        if not batches:
            return [{"message": "No job batches tracked."}]
        return batches

    @mcp.tool(
        name="cancel_job",
        description=(
            "Cancel pending tasks in a job batch. Only tasks that have "
            "not yet started executing can be cancelled."
        ),
    )
    def cancel_job(batch_id: str) -> dict:
        """Cancel pending tasks in a job batch."""
        return tracker.cancel_batch(batch_id)

    if backend.is_async_remote and hasattr(backend, "check_endpoint_status"):

        @mcp.tool(
            name="check_endpoint_status",
            description=(
                "Check whether the remote HPC compute endpoint is "
                "reachable and accepting tasks. Use this as a pre-flight "
                "check before submitting jobs."
            ),
        )
        def check_endpoint_status() -> dict:
            """Check the remote compute endpoint status."""
            return backend.check_endpoint_status()
