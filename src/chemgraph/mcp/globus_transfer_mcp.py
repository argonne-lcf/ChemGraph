"""Dedicated MCP server for Globus Transfer file staging.

The server deliberately exposes transfer orchestration only.  Chemistry and
simulation tools remain on their existing MCP servers, allowing an agent to
combine the two services explicitly when it needs to create and then move an
artifact between sites.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from chemgraph.execution.config import get_transfer_manager
from chemgraph.execution.globus_transfer import GlobusTransferManager
from chemgraph.mcp.transfer_tools import register_transfer_tools


_REQUIRED_ENV_VARS = (
    "GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID",
    "GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID",
    "GLOBUS_TRANSFER_DESTINATION_BASE_PATH",
)


def create_globus_transfer_mcp(
    transfer_manager: GlobusTransferManager | None = None,
) -> FastMCP:
    """Create a transfer-only MCP server from explicit or configured settings."""

    manager = (
        transfer_manager
        if transfer_manager is not None
        else get_transfer_manager()
    )
    if manager is None:
        variables = ", ".join(_REQUIRED_ENV_VARS)
        raise RuntimeError(
            "Globus Transfer is not configured. Set the corresponding "
            "[execution.globus_transfer] values in config.toml or export: "
            f"{variables}."
        )

    mcp = FastMCP(
        name="ChemGraph Globus Transfer Tools",
        instructions="""
            You stage files between configured Globus collections.
            Transfer only paths returned by upstream tools; do not invent paths.
            Wait for successful completion before handing a destination path to
            another agent. Report transfer failures exactly as returned.
        """,
    )
    register_transfer_tools(mcp, manager)
    return mcp


def main() -> None:
    """Run the dedicated transfer server."""

    from chemgraph.mcp.server_utils import run_mcp_server

    run_mcp_server(create_globus_transfer_mcp(), default_port=9006)


if __name__ == "__main__":
    main()
