"""Backward-compatibility alias for :mod:`chemgraph.mcp.xanes_mcp_hpc`.

.. deprecated::
    This module has been renamed to ``chemgraph.mcp.xanes_mcp_hpc``.
    Import from there instead.  This shim will be removed in a future
    release.

To use the Parsl backend specifically, set ``CHEMGRAPH_EXECUTION_BACKEND=parsl``
or configure ``[execution] backend = "parsl"`` in ``config.toml``.
"""

from chemgraph.mcp.xanes_mcp_hpc import (  # noqa: F401
    fetch_mp_structures,
    mcp,
    plot_xanes,
    run_xanes_ensemble,
    run_xanes_single,
)
from chemgraph.mcp.server_utils import run_mcp_server  # noqa: F401

if __name__ == "__main__":
    run_mcp_server(mcp, default_port=9007)
