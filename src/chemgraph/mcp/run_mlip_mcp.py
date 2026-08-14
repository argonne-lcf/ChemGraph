"""Standalone MCP server for runtime-selectable MLIP calculations."""

from mcp.server.fastmcp import FastMCP

from chemgraph.mcp.server_utils import run_mcp_server
from chemgraph.schemas.mlip_input import MLIPBatchInputSchema, MLIPInputSchema
from chemgraph.tools.run_mlip_core import run_mlip_batch_core, run_mlip_core

mcp = FastMCP(
    name="ChemGraph MLIP Tools",
    instructions=(
        "Run machine-learned interatomic-potential calculations through an "
        "explicit ASE or NVIDIA ALCHEMI execution runtime."
    ),
)


@mcp.tool()
def run_mlip(params: MLIPInputSchema) -> dict:
    """Run one MLIP energy calculation or fixed-cell geometry optimization."""
    return run_mlip_core(params)


@mcp.tool()
def run_mlip_batch(params: MLIPBatchInputSchema) -> dict:
    """Run an ordered batch of MLIP calculations and write a JSON manifest."""
    return run_mlip_batch_core(params)


if __name__ == "__main__":
    run_mcp_server(mcp, default_port=9006)
