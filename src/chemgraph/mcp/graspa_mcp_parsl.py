"""Backward-compatibility alias for :mod:`chemgraph.mcp.graspa_mcp_hpc`.

.. deprecated::
    This module has been renamed to ``chemgraph.mcp.graspa_mcp_hpc``.
    Import from there instead.  This shim will be removed in a future
    release.

To use the Parsl backend specifically, set ``CHEMGRAPH_EXECUTION_BACKEND=parsl``
or configure ``[execution] backend = "parsl"`` in ``config.toml``.
"""

from chemgraph.mcp.graspa_mcp_hpc import mcp, run_graspa_ensemble  # noqa: F401
from chemgraph.mcp.server_utils import run_mcp_server  # noqa: F401

if __name__ == "__main__":
    run_mcp_server(mcp, default_port=9001)
