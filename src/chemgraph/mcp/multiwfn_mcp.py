"""Dedicated FastMCP server for scripted Multiwfn analyses."""

from __future__ import annotations

import asyncio

from mcp.server.fastmcp import FastMCP

from chemgraph.schemas.multiwfn_schema import MultiwfnInputSchema, MultiwfnResult
from chemgraph.tools.multiwfn_core import run_multiwfn_core


mcp = FastMCP(
    name="ChemGraph Multiwfn Tools",
    instructions="""
        You run Multiwfn analyses from exact, pre-validated menu-response
        sequences. Use an empty string for pressing Enter and include all
        responses needed to return from submenus and exit Multiwfn cleanly.

        Multiwfn is configured by the server through MULTIWFN_EXE and optional
        MULTIWFN_HOME environment variables. Full console output and generated
        artifacts are written to a unique directory under CHEMGRAPH_LOG_DIR.
        Do not invent menu numbers or claim that a failed or timed-out run
        succeeded.
    """,
)


@mcp.tool(
    name="run_multiwfn",
    description=(
        "Run a Multiwfn analysis from an exact sequence of menu responses. "
        "Use empty strings for Enter and include the responses needed to exit."
    ),
    structured_output=True,
)
async def run_multiwfn(params: MultiwfnInputSchema) -> MultiwfnResult:
    """Run Multiwfn without blocking the MCP server's event loop."""
    return await asyncio.to_thread(run_multiwfn_core, params)


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    run_mcp_server(mcp, default_port=9006)
