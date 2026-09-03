#!/usr/bin/env python
"""Run a transferred ASE/EMT energy calculation with a Deep Agent.

The laptop process authenticates Globus Transfer, starts the backend-aware ASE
MCP server over stdio, and binds its Transfer and Compute tools to ChemGraph's
generic Deep Agent.  The model stages and submits the calculation; this driver
polls the asynchronous batch deterministically and resumes the same graph to
retrieve and summarize the result.

Required environment variables::

    GLOBUS_COMPUTE_ENDPOINT_ID
    GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID
    GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID
    GLOBUS_TRANSFER_DESTINATION_BASE_PATH

The selected model's credentials are also required.  The first run may prompt
for a Globus authorization code before the MCP subprocess is started.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from chemgraph.graphs.deep_agent import construct_deep_agent_graph
from chemgraph.models.loader import load_chat_model


_HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = _HERE / "structures" / "water.xyz"
REQUIRED_ENV = (
    "GLOBUS_COMPUTE_ENDPOINT_ID",
    "GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID",
    "GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID",
    "GLOBUS_TRANSFER_DESTINATION_BASE_PATH",
)
BOUND_TOOL_NAMES = (
    "check_endpoint_status",
    "transfer_files",
    "check_transfer_status",
    "list_remote_files",
    "run_ase_ensemble",
    "check_job_status",
    "get_job_results",
)
_SERVER_ENV_KEYS = (
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "PYTHONPATH",
    "COMPUTE_SYSTEM",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    *REQUIRED_ENV,
)

GLOBUS_ASE_SYSTEM_PROMPT = """\
You are running one explicit live integration test of Globus Transfer, Globus
Compute, and ASE. Use the supplied MCP tools exactly as directed. Do not use
Deep Agent filesystem tools, subagents, or shell-like behavior, and never call
an ordinary in-process `run_ase` tool.

For a submission request, call `check_endpoint_status`, then `transfer_files`
with `wait=true`. Only after the transfer reports `status="completed"`, pass
its exact `remote_directory` to `run_ase_ensemble`. Stop after submission and
report the returned `batch_id`; do not poll the Compute batch yourself.

For a retrieval request naming a completed batch, call `get_job_results`
exactly once with that batch ID. Report the returned potential energy in eV
without inventing or changing any value.
"""


def select_globus_ase_tools(tools: Sequence[Any]) -> tuple[list[Any], dict[str, Any]]:
    """Return the exact MCP tool subset required by this example."""
    by_name: dict[str, Any] = {}
    duplicates: set[str] = set()
    for tool in tools:
        name = getattr(tool, "name", None)
        if name not in BOUND_TOOL_NAMES:
            continue
        if name in by_name:
            duplicates.add(name)
        else:
            by_name[name] = tool

    if duplicates:
        raise RuntimeError(
            "Duplicate Globus ASE MCP tools: " + ", ".join(sorted(duplicates))
        )
    missing = [name for name in BOUND_TOOL_NAMES if name not in by_name]
    if missing:
        raise RuntimeError(
            "Missing Globus ASE MCP tools: "
            + ", ".join(missing)
            + ". Confirm all GLOBUS_TRANSFER_* variables were forwarded."
        )
    return [by_name[name] for name in BOUND_TOOL_NAMES], by_name


def decode_tool_payload(value: Any) -> dict[str, Any]:
    """Decode dict or MCP/LangChain content into one JSON object."""
    if isinstance(value, dict):
        for key in ("structuredContent", "structured_content", "data"):
            nested = value.get(key)
            if isinstance(nested, dict):
                return nested
        return value

    data = getattr(value, "data", None)
    if isinstance(data, dict):
        return data
    artifact = getattr(value, "artifact", None)
    if artifact is not None:
        try:
            return decode_tool_payload(artifact)
        except ValueError:
            pass
    content = getattr(value, "content", value)

    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool returned non-JSON text: {content}") from exc
        if not isinstance(decoded, dict):
            raise ValueError("Tool JSON result is not an object.")
        return decode_tool_payload(decoded)

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    try:
                        return decode_tool_payload(block["text"])
                    except ValueError:
                        continue
                try:
                    return decode_tool_payload(block)
                except ValueError:
                    continue
            text = getattr(block, "text", None)
            if isinstance(text, str):
                try:
                    return decode_tool_payload(text)
                except ValueError:
                    continue

    raise ValueError(f"Tool returned no JSON object: {value!r}")


def find_tool_payload(state: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the most recent payload for a named graph tool call."""
    for message in reversed(state.get("messages", [])):
        message_name = (
            getattr(message, "name", None)
            if not isinstance(message, dict)
            else message.get("name")
        )
        message_type = (
            getattr(message, "type", None)
            if not isinstance(message, dict)
            else message.get("type")
        )
        if message_name == name and (
            isinstance(message, ToolMessage) or message_type == "tool"
        ):
            return decode_tool_payload(message)
    raise RuntimeError(f"Deep Agent did not call required tool {name!r}.")


async def invoke_json_tool(tool: Any, arguments: dict[str, Any]) -> dict[str, Any]:
    """Invoke a LangChain MCP tool and require a JSON-object response."""
    return decode_tool_payload(await tool.ainvoke(arguments))


async def wait_for_batch(
    status_tool: Any,
    batch_id: str,
    *,
    timeout: float,
    poll_interval: float,
) -> dict[str, Any]:
    """Poll a Compute batch outside the LLM until it reaches a terminal state."""
    deadline = time.monotonic() + timeout
    while True:
        status = await invoke_json_tool(status_tool, {"batch_id": batch_id})
        state = str(status.get("status", "")).lower()
        print(
            "Compute status: "
            f"{state or 'unknown'} "
            f"({status.get('completed_tasks', 0)}/{status.get('total_tasks', '?')})"
        )
        if state == "completed":
            return status
        if state in {"failed", "partial"} or "error" in status:
            raise RuntimeError(f"Compute batch {batch_id} failed: {status}")
        if state not in {"pending", "running"}:
            raise RuntimeError(
                f"Compute batch {batch_id} returned unknown status: {status}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Compute batch {batch_id} did not finish within {timeout:g}s."
            )
        await asyncio.sleep(min(poll_interval, remaining))


def validate_energy_result(payload: dict[str, Any], batch_id: str) -> float:
    """Require one successful result with a finite potential energy."""
    if payload.get("batch_id") != batch_id:
        raise RuntimeError(
            f"Result batch ID does not match {batch_id!r}: {payload!r}"
        )
    if payload.get("status") != "completed":
        raise RuntimeError(f"Compute results are not completed: {payload!r}")
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise RuntimeError(f"Expected exactly one Compute result: {payload!r}")
    result = results[0]
    if not isinstance(result, dict) or result.get("status") != "success":
        raise RuntimeError(f"ASE calculation failed: {result!r}")
    energy = result.get("potential_energy")
    if isinstance(energy, bool) or not isinstance(energy, (int, float)):
        raise RuntimeError(f"ASE result has no numeric potential_energy: {result!r}")
    energy = float(energy)
    if not math.isfinite(energy):
        raise RuntimeError(f"ASE potential_energy is not finite: {energy!r}")
    return energy


def _server_environment(amqp_port: int | None) -> dict[str, str]:
    env = {key: os.environ[key] for key in _SERVER_ENV_KEYS if key in os.environ}
    env["CHEMGRAPH_EXECUTION_BACKEND"] = "globus_compute"
    if amqp_port is not None:
        env["GLOBUS_COMPUTE_AMQP_PORT"] = str(amqp_port)
    return env


def _endpoint_is_online(payload: dict[str, Any]) -> bool:
    status = payload.get("status")
    if isinstance(status, dict):
        status = status.get("status")
    return str(status).lower() in {"online", "ok", "running"}


async def run_example(args: argparse.Namespace) -> float:
    server_name = "ChemGraph ASE (Globus)"
    client = MultiServerMCPClient(
        {
            server_name: {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-u", "-m", "chemgraph.mcp.ase_mcp_hpc"],
                "env": _server_environment(args.amqp_port),
            }
        }
    )

    async with contextlib.AsyncExitStack() as stack:
        session = await stack.enter_async_context(client.session(server_name))
        loaded_tools = await load_mcp_tools(session)
        bound_tools, tools = select_globus_ase_tools(loaded_tools)
        print(f"Bound MCP tools: {[tool.name for tool in bound_tools]}")

        model = load_chat_model(model_name=args.model, temperature=0.0)
        graph = construct_deep_agent_graph(
            model,
            tools=bound_tools,
            system_prompt=GLOBUS_ASE_SYSTEM_PROMPT,
            interrupt_on={},
            name="globus_ase_demo",
        )
        config = {
            "configurable": {"thread_id": f"globus-ase-{uuid.uuid4().hex}"}
        }
        input_path = str(args.input.resolve())
        submission_prompt = f"""\
Run the live integration test now.

1. Check the configured Globus Compute endpoint.
2. Transfer this exact local file with wait=true: {input_path}
3. Submit run_ase_ensemble with this exact params object, replacing only
   REMOTE_DIRECTORY with the transfer result's remote_directory:
   {{
     "remote_structure_directory": "REMOTE_DIRECTORY",
     "output_results_file": "globus_ase_energy.json",
     "driver": "energy",
     "calculator": {{"calculator_type": "emt"}}
   }}
4. Stop after submission and report the batch_id. Do not poll it.
"""
        submission = await graph.ainvoke(
            {"messages": [HumanMessage(content=submission_prompt)]},
            config=config,
        )

        endpoint = find_tool_payload(submission, "check_endpoint_status")
        if not _endpoint_is_online(endpoint):
            raise RuntimeError(f"Globus Compute endpoint is not online: {endpoint}")
        transfer = find_tool_payload(submission, "transfer_files")
        if transfer.get("status") != "completed":
            raise RuntimeError(f"Globus Transfer did not complete: {transfer}")
        remote_directory = transfer.get("remote_directory")
        if not isinstance(remote_directory, str) or not remote_directory:
            raise RuntimeError(f"Transfer returned no remote directory: {transfer}")
        submitted = find_tool_payload(submission, "run_ase_ensemble")
        if submitted.get("status") != "submitted" or not submitted.get("batch_id"):
            raise RuntimeError(f"ASE batch was not submitted: {submitted}")
        batch_id = str(submitted["batch_id"])
        print(f"Remote directory: {remote_directory}")
        print(f"Compute batch: {batch_id}")

        await wait_for_batch(
            tools["check_job_status"],
            batch_id,
            timeout=args.compute_timeout,
            poll_interval=args.poll_interval,
        )
        retrieval = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            f"Compute batch {batch_id} is now completed. Call "
                            "get_job_results exactly once for that batch and "
                            "report its potential energy."
                        )
                    )
                ]
            },
            config=config,
        )
        result_payload = find_tool_payload(retrieval, "get_job_results")
        energy = validate_energy_result(result_payload, batch_id)

        print(f"PASS: water EMT potential energy = {energy:.12g} eV")
        print(
            "Remote inputs and JSON results were left in place at "
            f"{remote_directory}; remove them manually when no longer needed."
        )
        return energy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--amqp-port",
        type=int,
        choices=(443, 5671, 5672),
        default=(
            int(os.environ["GLOBUS_COMPUTE_AMQP_PORT"])
            if os.environ.get("GLOBUS_COMPUTE_AMQP_PORT")
            else None
        ),
        help="Result-streaming port; use 443 when outbound 5671 is blocked.",
    )
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--compute-timeout", type=float, default=1800.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"ERROR: missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2
    if not args.input.is_file():
        print(f"ERROR: input file does not exist: {args.input}", file=sys.stderr)
        return 2
    if args.poll_interval <= 0 or args.compute_timeout <= 0:
        print("ERROR: polling interval and timeout must be positive.", file=sys.stderr)
        return 2

    try:
        from chemgraph.execution.config import get_transfer_manager

        transfer_manager = get_transfer_manager(allow_interactive_auth=True)
        if transfer_manager is None:
            raise RuntimeError("Globus Transfer configuration is incomplete.")
        print("Authenticating Globus Transfer in the parent process...")
        transfer_manager.authenticate()
        asyncio.run(run_example(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
