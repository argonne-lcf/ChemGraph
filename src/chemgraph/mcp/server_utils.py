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
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )

    # Always log to stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # If CHEMGRAPH_LOG_DIR is set, also log to a file
    import os

    log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"mcp_server_{mcp.name}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logging.info(f"Logging to file: {log_file}")
        except Exception as e:
            logging.error(f"Failed to setup file logging to {log_dir}: {e}")

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
