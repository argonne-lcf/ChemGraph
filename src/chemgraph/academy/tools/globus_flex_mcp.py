"""Flexible Globus Transfer MCP server -- LLM picks source + dest per call.

Contrast with globus_transfer_mcp.py, which pins one (source, dest,
base_path) triple per server instance (matches chemgraph's upstream
GlobusTransferManager shape). This module exposes the raw primitive:
the LLM names both endpoints on each call, so one agent can push AND
pull, and the same server serves multiple endpoint pairs.

Tools:
  transfer_file(source_endpoint, source_path, dest_endpoint, dest_path,
                wait=true, label=null) -> {task_id, status, ...}
  check_transfer_status(task_id) -> {status, bytes_transferred, ...}
  list_files(endpoint, path) -> [{name, type, size}, ...]

Auth: reuses the token cache
``~/.globus/chemgraph_transfer_tokens.json`` written by
chemgraph.execution.globus_transfer.GlobusTransferManager. If a token
isn't there, do the one-time OAuth flow via chemgraph's manager once
(any endpoint pair, doesn't matter) -- the cached refresh token works
for every subsequent invocation, from every endpoint pair.

CLI matches the other swarm MCP servers:

    python -m swarm.tools.globus_flex_mcp \\
        --transport streamable_http --host 127.0.0.1 --port <PORT>
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("chemgraph.academy.tools.globus_flex")

_TOKEN_FILE = Path.home() / ".globus" / "chemgraph_transfer_tokens.json"
_TRANSFER_SCOPE = "urn:globus:auth:scope:transfer.api.globus.org:all"
# ponytail: reuse chemgraph's default native-app client so the cached
# tokens minted by GlobusTransferManager work here unchanged.
_CLIENT_ID = "61338d24-54d5-408f-a10d-66c06b59f6d2"


mcp = FastMCP(
    name="globus_flex",
    instructions=(
        "Globus Transfer with per-call source and destination. The LLM "
        "supplies source_endpoint / source_path / dest_endpoint / "
        "dest_path on every transfer_file call, so one server handles "
        "any direction and any endpoint pair. Endpoint UUIDs come from "
        "the agent's mission prompt (do not guess). For long transfers "
        "pass wait=false and poll with check_transfer_status(task_id)."
    ),
)


_TC = None


def _get_transfer_client():
    """Return an authenticated globus_sdk.TransferClient.

    Loads a cached refresh token from ``~/.globus/chemgraph_transfer_tokens.json``
    (the file chemgraph.execution.globus_transfer already writes). If
    missing, raises with instructions to mint one via the chemgraph
    manager one time; we do NOT run the interactive flow here because
    an MCP server has no tty.
    """
    global _TC
    if _TC is not None:
        return _TC

    import globus_sdk

    if not _TOKEN_FILE.is_file():
        raise RuntimeError(
            f"Globus token cache not found at {_TOKEN_FILE}. Run the "
            "chemgraph GlobusTransferManager once from an interactive "
            "shell to mint tokens, then restart this MCP server."
        )
    tokens = json.loads(_TOKEN_FILE.read_text())

    client = globus_sdk.NativeAppAuthClient(_CLIENT_ID)
    # Refresh if expired. Best-effort: fall back to the cached access
    # token if refresh fails (the transfer client will surface any
    # auth error on the actual call).
    if tokens.get("expires_at_seconds", 0) < time.time():
        try:
            resp = client.oauth2_refresh_tokens(
                globus_sdk.RefreshTokenAuthorizer(tokens["refresh_token"], client)
            )
            tokens = resp.by_resource_server["transfer.api.globus.org"]
            _TOKEN_FILE.write_text(json.dumps(dict(tokens), indent=2))
            _TOKEN_FILE.chmod(0o600)
        except Exception as exc:
            logger.warning("token refresh failed, using cached token: %s", exc)

    authorizer = globus_sdk.AccessTokenAuthorizer(tokens["access_token"])
    _TC = globus_sdk.TransferClient(authorizer=authorizer)
    return _TC


def transfer_file(
    source_endpoint: str,
    source_path: str,
    dest_endpoint: str,
    dest_path: str,
    wait: bool = True,
    label: str | None = None,
    timeout_s: float = 3600.0,
    poll_interval_s: float = 5.0,
) -> dict[str, Any]:
    """Submit a Globus Transfer of one file (or one directory tree).

    Parameters
    ----------
    source_endpoint, dest_endpoint : str
        Endpoint UUIDs. Direction is entirely per-call.
    source_path : str
        Absolute path on the source endpoint. A trailing '/' means
        recursive directory transfer.
    dest_path : str
        Absolute destination path (file) or dir path (with trailing
        '/' when source is a directory).
    wait : bool
        Block until SUCCEEDED / FAILED / timeout. Default true.
    timeout_s : float
        Max seconds to wait when wait=true. Default 3600.
    """
    import globus_sdk
    tc = _get_transfer_client()

    recursive = source_path.endswith("/")
    # globus_sdk >=4 dropped the transfer_client kwarg; TransferData is
    # now a pure request body. submit_transfer on the client remains.
    tdata = globus_sdk.TransferData(
        source_endpoint=source_endpoint,
        destination_endpoint=dest_endpoint,
        label=label or "chemgraph.academy.globus_flex",
        sync_level="checksum",
    )
    tdata.add_item(source_path, dest_path, recursive=recursive)
    result = tc.submit_transfer(tdata)
    task_id = result["task_id"]

    response: dict[str, Any] = {
        "task_id": task_id,
        "source_endpoint": source_endpoint,
        "source_path": source_path,
        "dest_endpoint": dest_endpoint,
        "dest_path": dest_path,
        "recursive": recursive,
    }
    if not wait:
        response["status"] = "submitted"
        return response

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        task = tc.get_task(task_id)
        if task["status"] in ("SUCCEEDED", "FAILED"):
            response.update({
                "status": task["status"],
                "nice_status": task.get("nice_status", ""),
                "bytes_transferred": task.get("bytes_transferred", 0),
                "files": task.get("files", 0),
                "files_transferred": task.get("files_transferred", 0),
            })
            return response
        time.sleep(poll_interval_s)

    task = tc.get_task(task_id)
    response.update({
        "status": task["status"],
        "nice_status": task.get("nice_status", ""),
        "timed_out": True,
    })
    return response


def check_transfer_status(task_id: str) -> dict[str, Any]:
    """Poll a transfer's status. Use with transfer_file(wait=false)."""
    tc = _get_transfer_client()
    task = tc.get_task(task_id)
    return {
        "task_id": task_id,
        "status": task["status"],
        "nice_status": task.get("nice_status", ""),
        "bytes_transferred": task.get("bytes_transferred", 0),
        "files": task.get("files", 0),
        "files_transferred": task.get("files_transferred", 0),
    }


def list_files(endpoint: str, path: str) -> list[dict[str, Any]]:
    """List one directory on any endpoint. Verify files landed after a transfer."""
    tc = _get_transfer_client()
    return [
        {"name": e["name"], "type": e["type"], "size": e.get("size", 0)}
        for e in tc.operation_ls(endpoint, path=path)
    ]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Globus Flex MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "streamable_http"],
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9014)
    args = parser.parse_args()

    mcp.add_tool(transfer_file, name="transfer_file", description=(
        "Transfer one file (or a directory when source_path ends with '/') "
        "between two Globus endpoints. Both endpoint UUIDs and both paths "
        "come from the LLM per call, so the same tool handles push AND pull. "
        "wait=true blocks until SUCCEEDED/FAILED; wait=false returns "
        "immediately with a task_id to poll via check_transfer_status."
    ))
    mcp.add_tool(check_transfer_status, name="check_transfer_status", description=(
        "Poll a Globus Transfer task by task_id. Use after transfer_file(wait=false)."
    ))
    mcp.add_tool(list_files, name="list_files", description=(
        "List one directory on any Globus endpoint. Useful to verify files "
        "landed after a transfer, or to discover inputs on a remote endpoint."
    ))

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
