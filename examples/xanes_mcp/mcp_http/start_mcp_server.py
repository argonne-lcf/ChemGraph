"""
Start the ChemGraph XANES MCP server via HTTP.

This is a thin wrapper that launches the XANES MCP server
(chemgraph.mcp.xanes_mcp) with streamable HTTP transport.

Prerequisites:
  - FDMNES_EXE set in environment
  - MP_API_KEY set in environment (for fetch_mp_structures)

Usage:
  python start_mcp_server.py

  # Custom host/port:
  python start_mcp_server.py --host 0.0.0.0 --port 9007
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Start the ChemGraph XANES MCP server (HTTP transport).",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to. Default: 0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9007,
        help="Port to listen on. Default: 9007",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "chemgraph.mcp.xanes_mcp",
        "--transport",
        "streamable_http",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    print(f"Starting XANES MCP server on {args.host}:{args.port} ...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Connect at: http://localhost:{args.port}/mcp/")
    print()

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
