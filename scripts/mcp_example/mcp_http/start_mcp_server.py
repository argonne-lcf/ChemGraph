"""Start ChemGraph's MCP server via HTTP (streamable_http).

This is a thin convenience wrapper around ChemGraph's built-in MCP
server.  It is equivalent to running:

    python -m chemgraph.mcp.mcp_tools --transport streamable_http --port 9003

Usage
-----
    python start_mcp_server.py                # default port 9003
    python start_mcp_server.py --port 9005    # custom port
"""

import sys

from chemgraph.mcp.mcp_tools import mcp
from chemgraph.mcp.server_utils import run_mcp_server

# Override default argv to use streamable_http if no --transport flag provided
if "--transport" not in sys.argv:
    sys.argv.extend(["--transport", "streamable_http"])

run_mcp_server(mcp, default_port=9003)
