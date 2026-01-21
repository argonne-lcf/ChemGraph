import argparse
import logging
import sys
import uvicorn
from mcp.server.fastmcp import FastMCP


def run_mcp_server(
    mcp: FastMCP, default_port: int = 8000, default_host: str = "127.0.0.1"
):
    """
    Standardizes the startup process for ChemGraph MCP servers.
    Supports 'stdio' (default) and 'sse' (HTTP) transports.
    Ensures logging is correctly routed to stderr to avoid corrupting stdio transport.
    """
    parser = argparse.ArgumentParser(description=f"Run {mcp.name} MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable_http"],
        default="stdio",
        help="Transport protocol to use: 'stdio' (default) or 'streamable_http' (HTTP/SSE).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Port for streamable_http transport (default: {default_port})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=default_host,
        help=f"Host for streamable_http transport (default: {default_host})",
    )

    args = parser.parse_args()

    # Configure logging to write to stderr.
    # This is CRITICAL for stdio transport mode, as stdout is used for communication.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        stream=sys.stderr,
    )

    if args.transport == "streamable_http":
        logging.info(
            "Starting %s on %s:%s via streamable_http transport...",
            mcp.name,
            args.host,
            args.port,
        )
        # FastMCP.streamable_http_app() returns a Starlette/FastAPI-compatible app
        app = mcp.streamable_http_app()
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        logging.info("Starting %s via stdio transport...", mcp.name)
        # FastMCP.run(transport='stdio') handles the stdio loop
        mcp.run(transport="stdio")
