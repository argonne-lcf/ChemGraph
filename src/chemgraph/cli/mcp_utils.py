"""MCP client utilities for the ChemGraph CLI.

Handles connecting to MCP servers and loading tools for use with
MCP-enabled workflows (single_agent, multi_agent, etc.).
"""

from __future__ import annotations

import shlex
import time
from typing import List, Optional

from rich.progress import Progress, SpinnerColumn, TextColumn

from chemgraph.cli.formatting import console
from chemgraph.mcp.client import load_mcp_tools
from chemgraph.mcp.config import MCPToolSourceConfig
from chemgraph.utils.async_utils import run_async_callable



def load_mcp_tools_from_config(
    url: Optional[str] = None,
    command: Optional[str] = None,
    server_name: str = "ChemGraph General Tools",
    verbose: bool = False,
) -> Optional[List]:
    """Connect to an MCP server and return loaded tools.

    Supports two transports:

    - **streamable_http**: specify *url* (e.g. ``http://localhost:9003/mcp/``)
    - **stdio**: specify *command* as a shell command string

    Parameters
    ----------
    url : str, optional
        MCP server URL for streamable_http transport.
    command : str, optional
        Shell command to launch an MCP server via stdio transport.
        The first token is the executable; the rest are arguments.
    server_name : str
        Display name for the MCP server connection.
    verbose : bool
        Print extra diagnostic information.

    Returns
    -------
    list or None
        List of loaded MCP tools, or ``None`` on failure.
    """
    if url and command:
        console.print(
            "[yellow]Both --mcp-url and --mcp-command specified; "
            "using --mcp-url (streamable_http).[/yellow]"
        )

    if url:
        source = MCPToolSourceConfig(name=server_name, url=url)
        transport_label = f"streamable_http @ {url}"
    elif command:
        parts = shlex.split(command)
        if not parts:
            console.print("[red]MCP command must not be empty.[/red]")
            return None
        source = MCPToolSourceConfig(
            name=server_name, transport="stdio", command=parts[0], args=tuple(parts[1:]),
        )
        transport_label = f"stdio: {command}"
    else:
        console.print("[red]No MCP server URL or command provided.[/red]")
        return None

    if verbose:
        console.print(f"[blue]Connecting to MCP server: {transport_label}[/blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Loading MCP tools...", total=None)
        try:
            tools = run_async_callable(lambda: load_mcp_tools([source]))
            progress.update(
                task,
                description=f"[green]Loaded {len(tools)} MCP tools!",
            )
            time.sleep(0.3)

            if verbose:
                tool_names = [t.name for t in tools]
                console.print(f"[blue]MCP tools: {tool_names}[/blue]")

            if not tools:
                console.print(
                    "[yellow]Warning: MCP server returned zero tools.[/yellow]"
                )

            return tools

        except Exception as e:
            progress.update(task, description="[red]MCP connection failed!")
            console.print(f"[red]Failed to load MCP tools: {e}[/red]")

            err_str = str(e).lower()
            if "connection" in err_str or "refused" in err_str:
                console.print(
                    "[dim]Check that the MCP server is running and reachable.[/dim]"
                )
            elif "timeout" in err_str:
                console.print(
                    "[dim]The MCP server did not respond in time.[/dim]"
                )
            return None
