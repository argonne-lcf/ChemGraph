#!/usr/bin/env python
"""Run transferred ASE/MACE energy calculations with a Deep Agent.

The laptop process authenticates Globus Transfer, starts the backend-aware ASE
MCP server over stdio, and binds its Transfer and Compute tools to ChemGraph's
generic Deep Agent. The model performs the complete workflow in one graph turn:
facility discovery, transfer, ASE submission, blocking job wait, and result
retrieval. The Python wrapper only validates and prints the final result.

Required environment variables::

    GLOBUS_COMPUTE_ENDPOINT_ID
    GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID
    GLOBUS_TRANSFER_DESTINATION_BASE_PATH

Set ``COMPUTE_SYSTEM=polaris`` or ``COMPUTE_SYSTEM=aurora`` to select the
matching bundled destination collection and path mapping.

``--input`` accepts one structure file or a directory of ``.xyz`` files. The
selected model's credentials are also required. The first run may prompt
for a Globus authorization code before the MCP subprocess is started.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import re
import sys
import traceback
import uuid
from collections.abc import Mapping, Sequence
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
    "GLOBUS_TRANSFER_DESTINATION_BASE_PATH",
)
BOUND_TOOL_NAMES = (
    "list_transfer_facilities",
    "check_endpoint_status",
    "transfer_files",
    "check_transfer_status",
    "list_remote_files",
    "run_ase_ensemble",
    "wait_for_job",
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
    "GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID",
    "GLOBUS_TRANSFER_DESTINATION_COMPUTE_BASE_PATH",
    "CHEMGRAPH_REMOTE_DIRECTORY_TIMEOUT",
    *REQUIRED_ENV,
)

GLOBUS_ASE_SYSTEM_PROMPT = """\
You are a Globus and ASE orchestration agent. Use the available tools
autonomously to complete the request. Read offloaded tool results when
necessary, and never invent simulation results.
"""

_OFFLOADED_TOOL_RESULT_RE = re.compile(
    r"saved in the filesystem at this path:\s*(?P<path>/\S+)"
)
_SENSITIVE_TRACE_TEXT_RE = re.compile(
    r"(?i)\b(access[_-]?token|refresh[_-]?token|api[_-]?key|authorization|"
    r"password|secret)\b([\"']?\s*[:=]\s*[\"']?)([^\s,;\"']+)"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")


def build_user_request(
    *,
    input_path: str,
    is_directory: bool,
    input_count: int,
    timeout: float,
    poll_interval: float,
) -> str:
    """Build one high-level request without prescribing a tool sequence."""
    if is_directory:
        action = f"Stage every .xyz file in {input_path}"
        structures = "staged structures"
    else:
        action = f"Stage the input structure at {input_path}"
        structures = "staged structure"
    return (
        f"{action} to the configured HPC facility and run a MACE-MP small "
        f"CUDA energy simulation over the {structures}. Complete the workflow "
        f"and report the success/failure summary. Expect {input_count} input "
        f"structure(s); wait up to {timeout:g} seconds and use a "
        f"{poll_interval:g}-second polling interval."
    )


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


def decode_tool_payload(
    value: Any,
    *,
    files: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode dict or MCP/LangChain content into one JSON object."""
    if isinstance(value, dict):
        for key in ("structuredContent", "structured_content", "data"):
            nested = value.get(key)
            if isinstance(nested, dict):
                return decode_tool_payload(nested, files=files)
        return value

    data = getattr(value, "data", None)
    if isinstance(data, dict):
        return data
    artifact = getattr(value, "artifact", None)
    if artifact is not None:
        try:
            return decode_tool_payload(artifact, files=files)
        except ValueError:
            pass
    content = getattr(value, "content", value)

    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError as exc:
            match = _OFFLOADED_TOOL_RESULT_RE.search(content)
            if match is None:
                raise ValueError(f"Tool returned non-JSON text: {content}") from exc
            path = match.group("path").rstrip(".,;")
            if files is None or path not in files:
                raise ValueError(
                    f"Tool result was offloaded to {path!r}, but the graph "
                    "state contains no matching file."
                ) from exc
            file_data = files[path]
            if not isinstance(file_data, Mapping):
                raise ValueError(
                    f"Offloaded tool result {path!r} has invalid file metadata."
                ) from exc
            encoding = file_data.get("encoding", "utf-8")
            if encoding != "utf-8":
                raise ValueError(
                    f"Offloaded tool result {path!r} uses unsupported "
                    f"encoding {encoding!r}; expected 'utf-8'."
                ) from exc
            stored_content = file_data.get("content")
            if not isinstance(stored_content, str):
                raise ValueError(
                    f"Offloaded tool result {path!r} has no text content."
                ) from exc
            try:
                return decode_tool_payload(stored_content, files=files)
            except ValueError as stored_exc:
                raise ValueError(
                    f"Offloaded tool result {path!r} does not contain valid "
                    "JSON."
                ) from stored_exc
        if not isinstance(decoded, dict):
            raise ValueError("Tool JSON result is not an object.")
        return decode_tool_payload(decoded, files=files)

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    try:
                        return decode_tool_payload(block["text"], files=files)
                    except ValueError:
                        if _OFFLOADED_TOOL_RESULT_RE.search(block["text"]):
                            raise
                        continue
                try:
                    return decode_tool_payload(block, files=files)
                except ValueError:
                    continue
            text = getattr(block, "text", None)
            if isinstance(text, str):
                try:
                    return decode_tool_payload(text, files=files)
                except ValueError:
                    if _OFFLOADED_TOOL_RESULT_RE.search(text):
                        raise
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
            return decode_tool_payload(message, files=state.get("files"))
    raise RuntimeError(f"Deep Agent did not call required tool {name!r}.")


def find_tool_arguments(state: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the most recent arguments for a named graph tool call."""
    for message in reversed(state.get("messages", [])):
        tool_calls = (
            message.get("tool_calls", [])
            if isinstance(message, dict)
            else getattr(message, "tool_calls", [])
        )
        for tool_call in reversed(tool_calls or []):
            if isinstance(tool_call, dict):
                call_name = tool_call.get("name")
                arguments = tool_call.get("args")
            else:
                call_name = getattr(tool_call, "name", None)
                arguments = getattr(tool_call, "args", None)
            if call_name == name and isinstance(arguments, dict):
                return arguments
    raise RuntimeError(f"Deep Agent did not record tool arguments for {name!r}.")


def validate_mace_tool_call(state: dict[str, Any]) -> None:
    """Prevent this MACE demo from silently falling back to another calculator."""
    submitted = find_tool_arguments(state, "run_ase_ensemble")
    params = submitted.get("params")
    calculator = params.get("calculator") if isinstance(params, dict) else None
    calculator_type = (
        calculator.get("calculator_type")
        if isinstance(calculator, dict)
        else None
    )
    if not isinstance(calculator_type, str) or not calculator_type.startswith(
        "mace_"
    ):
        raise RuntimeError(
            "Deep Agent must use a MACE calculator for run_ase_ensemble; "
            f"got {calculator_type!r}."
        )


def _redact_trace_value(value: Any) -> Any:
    """Redact credential-like values before printing an agent trace."""
    if isinstance(value, Mapping):
        redacted = {}
        for key, nested in value.items():
            key_text = str(key)
            normalized = key_text.lower()
            if any(
                marker in normalized
                for marker in (
                    "token",
                    "secret",
                    "password",
                    "api_key",
                    "apikey",
                    "authorization",
                )
            ):
                redacted[key_text] = "[REDACTED]"
            else:
                redacted[key_text] = _redact_trace_value(nested)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact_trace_value(item) for item in value]
    if isinstance(value, str):
        value = _SENSITIVE_TRACE_TEXT_RE.sub(
            lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
            value,
        )
        return _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", value)
    return value


def _trace_json(value: Any) -> str:
    return json.dumps(
        _redact_trace_value(value),
        default=str,
        sort_keys=True,
    )


def _print_trace_message(label: str, content: Any) -> None:
    """Print one human- or model-authored message with redaction."""
    print(label)
    if isinstance(content, str):
        print(_redact_trace_value(content))
    else:
        print(f"  content: {_trace_json(content)}")


def print_deep_agent_trace(
    state: dict[str, Any],
    seen: set[tuple[str, str]] | None = None,
    *,
    include_offloaded_payloads: bool = False,
) -> set[tuple[str, str]]:
    """Print newly surfaced Deep Agent messages and tool events in order."""
    seen = seen if seen is not None else set()
    for message_index, message in enumerate(state.get("messages", [])):
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls", [])
            message_type = message.get("type")
            message_role = message.get("role")
            message_id = message.get("id")
            message_name = message.get("name")
            tool_call_id = message.get("tool_call_id")
            message_status = message.get("status")
            content = message.get("content")
        else:
            tool_calls = getattr(message, "tool_calls", [])
            message_type = getattr(message, "type", None)
            message_role = getattr(message, "role", None)
            message_id = getattr(message, "id", None)
            message_name = getattr(message, "name", None)
            tool_call_id = getattr(message, "tool_call_id", None)
            message_status = getattr(message, "status", None)
            content = getattr(message, "content", None)

        event_id = str(message_id or f"{message_index}:{message_type or message_role}")
        message_key = ("message", event_id)
        has_content = content not in (None, "", [])
        if message_key not in seen and has_content:
            if message_type == "human" or message_role in {"human", "user"}:
                seen.add(message_key)
                _print_trace_message("Deep Agent input [human]:", content)
            elif message_type == "ai" or message_role == "assistant":
                seen.add(message_key)
                _print_trace_message("Deep Agent output [assistant]:", content)

        for call_index, tool_call in enumerate(tool_calls or []):
            if isinstance(tool_call, dict):
                name = tool_call.get("name", "unknown")
                arguments = tool_call.get("args", {})
                call_id = tool_call.get("id")
            else:
                name = getattr(tool_call, "name", "unknown")
                arguments = getattr(tool_call, "args", {})
                call_id = getattr(tool_call, "id", None)
            event_id = str(call_id or f"{message_index}:{call_index}:{name}")
            event_key = ("call", event_id)
            if event_key in seen:
                continue
            seen.add(event_key)
            print(f"Deep Agent tool call [{event_id}]: {name}")
            print(f"  arguments: {_trace_json(arguments)}")

        if isinstance(message, ToolMessage) or message_type == "tool":
            event_id = str(tool_call_id or f"{message_index}:{message_name}")
            event_key = ("result", event_id)
            if event_key in seen:
                continue
            seen.add(event_key)
            try:
                result = decode_tool_payload(
                    message,
                    files=(
                        state.get("files")
                        if include_offloaded_payloads
                        else None
                    ),
                )
            except ValueError:
                result = content
            print(f"Deep Agent tool result [{event_id}]: {message_name}")
            if message_status:
                print(f"  status: {message_status}")
            if isinstance(result, str):
                print("  payload:")
                print(_redact_trace_value(result))
            else:
                print(f"  payload: {_trace_json(result)}")

    sys.stdout.flush()
    return seen


def discover_input_files(input_path: Path) -> list[Path]:
    """Resolve one structure or the direct ``.xyz`` children of a directory."""
    resolved = input_path.resolve()
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")

    files = sorted(
        (
            path.resolve()
            for path in resolved.iterdir()
            if path.is_file() and path.suffix.lower() == ".xyz"
        ),
        key=lambda path: path.name,
    )
    if not files:
        raise ValueError(f"Input directory contains no .xyz files: {input_path}")
    return files


def summarize_energy_results(
    payload: dict[str, Any],
    batch_id: str,
    input_files: Sequence[Path],
) -> dict[str, Any]:
    """Validate an ensemble payload and associate every result with its input."""
    if payload.get("batch_id") != batch_id:
        raise RuntimeError(
            f"Result batch ID does not match {batch_id!r}: {payload!r}"
        )
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"Compute results are not a list: {payload!r}")

    expected_count = len(input_files)
    by_index: dict[int, dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict):
            raise RuntimeError(f"Compute result is not an object: {result!r}")
        index = result.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise RuntimeError(f"Compute result has no integer index: {result!r}")
        if not 0 <= index < expected_count:
            raise RuntimeError(
                f"Compute result index {index} is outside 0..{expected_count - 1}."
            )
        if index in by_index:
            raise RuntimeError(f"Duplicate Compute result index: {index}")
        by_index[index] = result

    energies: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, input_file in enumerate(input_files):
        result = by_index.get(index)
        if result is None:
            failures.append(
                {
                    "index": index,
                    "structure": input_file.name,
                    "error_type": "MissingResult",
                    "message": "No result was returned for this structure.",
                }
            )
            continue

        energy = result.get("potential_energy")
        valid_energy = (
            not isinstance(energy, bool)
            and isinstance(energy, (int, float))
            and math.isfinite(float(energy))
        )
        if result.get("status") == "success" and valid_energy:
            energies.append(
                {
                    "index": index,
                    "structure": input_file.name,
                    "potential_energy": float(energy),
                }
            )
            continue

        if result.get("status") == "success":
            error_type = "InvalidEnergy"
            message = f"Invalid potential_energy: {energy!r}"
        else:
            error_type = str(result.get("error_type") or "CalculationFailed")
            message = str(result.get("message") or result)
        failures.append(
            {
                "index": index,
                "structure": input_file.name,
                "error_type": error_type,
                "message": message,
            }
        )

    energy_values = [entry["potential_energy"] for entry in energies]
    return {
        "batch_id": batch_id,
        "batch_status": str(payload.get("status", "unknown")),
        "expected_count": expected_count,
        "results_received": len(results),
        "succeeded": len(energies),
        "failed": len(failures),
        "energies": energies,
        "failures": failures,
        "energy_min": min(energy_values) if energy_values else None,
        "energy_max": max(energy_values) if energy_values else None,
        "energy_mean": (
            sum(energy_values) / len(energy_values) if energy_values else None
        ),
        "all_succeeded": (
            payload.get("status") == "completed"
            and len(energies) == expected_count
            and not failures
        ),
    }


def print_ensemble_summary(summary: Mapping[str, Any]) -> None:
    """Print aggregate energies and concise per-structure failures."""
    print(
        "MACE ensemble: "
        f"{summary['succeeded']}/{summary['expected_count']} succeeded, "
        f"{summary['failed']} failed "
        f"({summary['results_received']} results received)"
    )
    if summary["succeeded"]:
        print(
            "Potential energy (eV): "
            f"min={summary['energy_min']:.12g}, "
            f"max={summary['energy_max']:.12g}, "
            f"mean={summary['energy_mean']:.12g}"
        )
    if summary["failures"]:
        print("Failed structures:")
        for failure in summary["failures"]:
            print(
                f"  - {failure['structure']} [index={failure['index']}]: "
                f"{failure['error_type']}: {failure['message']}"
            )


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


async def run_example(args: argparse.Namespace) -> dict[str, Any]:
    input_files = discover_input_files(args.input)
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
        bound_tools, _ = select_globus_ase_tools(loaded_tools)
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
        request = build_user_request(
            input_path=input_path,
            is_directory=args.input.resolve().is_dir(),
            input_count=len(input_files),
            timeout=args.compute_timeout,
            poll_interval=args.poll_interval,
        )
        print("Deep Agent input [system]:")
        print(_redact_trace_value(GLOBUS_ASE_SYSTEM_PROMPT.rstrip()))
        sys.stdout.flush()
        seen_events: set[tuple[str, str]] = set()
        result: dict[str, Any] | None = None
        async for state in graph.astream(
            {"messages": [HumanMessage(content=request)]},
            config=config,
            stream_mode="values",
        ):
            result = state
            print_deep_agent_trace(
                state,
                seen_events,
                include_offloaded_payloads=args.trace_full_payloads,
            )
        if result is None:
            raise RuntimeError("Deep Agent produced no graph states.")

        facilities = find_tool_payload(result, "list_transfer_facilities")
        if not facilities.get("transfer_configured") or not facilities.get(
            "active_system"
        ):
            raise RuntimeError(
                f"Globus Transfer has no active facility: {facilities}"
            )
        endpoint = find_tool_payload(result, "check_endpoint_status")
        if not _endpoint_is_online(endpoint):
            raise RuntimeError(f"Globus Compute endpoint is not online: {endpoint}")
        transfer = find_tool_payload(result, "transfer_files")
        if transfer.get("status") != "completed":
            raise RuntimeError(f"Globus Transfer did not complete: {transfer}")
        if transfer.get("file_count") != len(input_files):
            raise RuntimeError(
                "Globus Transfer staged an unexpected number of files: "
                f"expected {len(input_files)}, got {transfer.get('file_count')!r}."
            )
        remote_directory = transfer.get("remote_directory")
        if not isinstance(remote_directory, str) or not remote_directory:
            raise RuntimeError(f"Transfer returned no remote directory: {transfer}")
        transfer_directory = transfer.get("transfer_directory", remote_directory)
        submitted = find_tool_payload(result, "run_ase_ensemble")
        if submitted.get("status") != "submitted" or not submitted.get("batch_id"):
            raise RuntimeError(f"ASE batch was not submitted: {submitted}")
        if submitted.get("n_tasks") != len(input_files):
            raise RuntimeError(
                "ASE submitted an unexpected number of tasks: "
                f"expected {len(input_files)}, got {submitted.get('n_tasks')!r}."
            )
        batch_id = str(submitted["batch_id"])
        waited = find_tool_payload(result, "wait_for_job")
        if waited.get("batch_id") != batch_id:
            raise RuntimeError(f"Compute wait returned the wrong batch: {waited}")
        result_payload = find_tool_payload(result, "get_job_results")
        validate_mace_tool_call(result)
        summary = summarize_energy_results(result_payload, batch_id, input_files)

        print(f"Active facility: {facilities['active_system']}")
        print(f"Local inputs: {len(input_files)} from {input_path}")
        print(f"Compute directory: {remote_directory}")
        print(f"Transfer directory: {transfer_directory}")
        print(f"Compute batch: {batch_id}")
        print_ensemble_summary(summary)
        if waited.get("status") != "completed" or not summary["all_succeeded"]:
            raise RuntimeError(
                "MACE ensemble did not complete successfully: "
                f"wait_status={waited.get('status')!r}, "
                f"result_status={summary['batch_status']!r}, "
                f"succeeded={summary['succeeded']}/{summary['expected_count']}."
            )
        print(
            "PASS: all "
            f"{summary['expected_count']} MACE-MP small calculations completed."
        )
        print(
            "Remote inputs and JSON results were left in place at "
            f"{transfer_directory}; remove them manually when no longer needed."
        )
        return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="One structure file or a directory whose direct .xyz files are staged.",
    )
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
    parser.add_argument(
        "--trace-full-payloads",
        action="store_true",
        help="Expand offloaded tool-result JSON instead of printing its preview.",
    )
    return parser


def _report_failure(exc: Exception) -> None:
    """Print a concise failure and expand nested asynchronous errors."""
    print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
    if isinstance(exc, BaseExceptionGroup):
        traceback.print_exception(exc, file=sys.stderr)


def main() -> int:
    args = _parser().parse_args()
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"ERROR: missing environment variables: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        discover_input_files(args.input)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
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
        _report_failure(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
