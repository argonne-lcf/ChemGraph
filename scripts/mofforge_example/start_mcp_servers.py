#!/usr/bin/env python
"""Start the four MOF workflow MCP servers over streamable HTTP.

The launcher owns the server processes and their runtime environments. Keep it
running while ``demo_single_agent_all_mcp.py`` connects from another terminal.
Press Ctrl-C to stop every server.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SERVER_MODULES = {
    "mofforge": "mofforge.mcp.server",
    "fairchem": "chemgraph.mcp.fairchem_mcp_hpc",
    "pacmof2": "chemgraph.mcp.pacmof2_mcp_hpc",
    "graspa": "chemgraph.mcp.graspa_mcp_hpc",
}

DEFAULT_PORTS = {
    "mofforge": 9010,
    "fairchem": 9008,
    "pacmof2": 9009,
    "graspa": 9001,
}

PYTHON_ENV_VARS = {
    name: f"{name.upper()}_PYTHON" for name in SERVER_MODULES
}

_HOST = "127.0.0.1"
_ENV_NAMES = {
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PYTHONPATH",
    "XDG_CACHE_HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}
_ENV_PREFIXES = (
    "CHEMGRAPH_",
    "GLOBUS_",
    "MOFFORGE_",
    "HF_",
    "CUDA_",
    "ZE_",
    "OMP_",
)
_POLL_INTERVAL_SECONDS = 0.2
_SHUTDOWN_TIMEOUT_SECONDS = 5.0


def _resolve_python(value: str | None, env_name: str) -> str:
    candidate = value or os.environ.get(env_name) or sys.executable
    resolved = shutil.which(candidate)
    if resolved is None:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            resolved = str(path)
    if resolved is None:
        raise ValueError(f"Python executable not found: {candidate!r}")
    # Keep a venv's Python symlink intact. Resolving the symlink can replace
    # ``venv/bin/python`` with the base interpreter and lose the venv packages.
    return str(Path(resolved).absolute())


def _server_environment(backend: str, compute_system: str) -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if name in _ENV_NAMES or name.startswith(_ENV_PREFIXES)
    }
    env["CHEMGRAPH_EXECUTION_BACKEND"] = backend
    env["COMPUTE_SYSTEM"] = compute_system
    return env


def build_server_commands(args: argparse.Namespace) -> dict[str, list[str]]:
    """Build one streamable-HTTP command per server."""
    commands: dict[str, list[str]] = {}
    for name, module in SERVER_MODULES.items():
        python = _resolve_python(
            getattr(args, f"{name}_python"),
            PYTHON_ENV_VARS[name],
        )
        transport = "streamable-http" if name == "mofforge" else "streamable_http"
        command = [
            python,
            "-u",
            "-m",
            module,
            "--transport",
            transport,
        ]
        if name != "mofforge":
            command.extend(["--host", _HOST])
        command.extend(["--port", str(getattr(args, f"{name}_port"))])
        commands[name] = command
    return commands


def _ports(args: argparse.Namespace) -> dict[str, int]:
    return {
        name: getattr(args, f"{name}_port")
        for name in SERVER_MODULES
    }


def _validate_ports(ports: dict[str, int]) -> None:
    invalid = {name: port for name, port in ports.items() if not 1 <= port <= 65535}
    if invalid:
        raise ValueError(f"MCP ports must be between 1 and 65535: {invalid}")
    if len(set(ports.values())) != len(ports):
        raise ValueError(f"MCP ports must be unique: {ports}")


def _ensure_ports_available(ports: dict[str, int]) -> None:
    for name, port in ports.items():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((_HOST, port))
            except OSError as exc:
                raise RuntimeError(
                    f"{name} MCP port {_HOST}:{port} is unavailable"
                ) from exc


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((_HOST, port)) == 0


def _wait_for_ready(
    processes: dict[str, subprocess.Popen[Any]],
    ports: dict[str, int],
    timeout: float,
) -> None:
    pending = set(processes)
    deadline = time.monotonic() + timeout
    while pending:
        for name in tuple(pending):
            process = processes[name]
            returncode = process.poll()
            if returncode is not None:
                raise RuntimeError(
                    f"{name} MCP server exited before readiness "
                    f"(return code {returncode})"
                )
            if _port_is_open(ports[name]):
                pending.remove(name)
        if pending and time.monotonic() >= deadline:
            raise RuntimeError(
                f"MCP servers did not become ready within {timeout:g}s: "
                f"{sorted(pending)}"
            )
        if pending:
            time.sleep(_POLL_INTERVAL_SECONDS)


def _shutdown_processes(
    processes: dict[str, subprocess.Popen[Any]],
) -> None:
    for process in processes.values():
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()

    deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
    for name, process in processes.items():
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            print(f"{name} did not stop gracefully; killing it.", file=sys.stderr)
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=1.0)


def supervise(args: argparse.Namespace) -> int:
    """Launch all servers, monitor them, and clean them up together."""
    ports = _ports(args)
    _validate_ports(ports)
    _ensure_ports_available(ports)
    commands = build_server_commands(args)
    env = _server_environment(args.backend, args.compute_system)
    processes: dict[str, subprocess.Popen[Any]] = {}

    try:
        for name, command in commands.items():
            print(f"Starting {name}: {' '.join(command)}", flush=True)
            processes[name] = subprocess.Popen(
                command,
                env=dict(env),
                start_new_session=True,
            )

        _wait_for_ready(processes, ports, args.startup_timeout)
        print("\nAll MCP servers are ready:")
        for name, port in ports.items():
            print(f"  {name}: http://{_HOST}:{port}/mcp/")
        print("\nPress Ctrl-C to stop all servers.", flush=True)

        while True:
            for name, process in processes.items():
                returncode = process.poll()
                if returncode is not None:
                    raise RuntimeError(
                        f"{name} MCP server exited unexpectedly "
                        f"(return code {returncode})"
                    )
            time.sleep(_POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopping MCP servers...", flush=True)
        return 0
    finally:
        _shutdown_processes(processes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=["local", "parsl", "ensemble_launcher", "globus_compute"],
        default=os.environ.get("CHEMGRAPH_EXECUTION_BACKEND", "local"),
    )
    parser.add_argument(
        "--compute-system",
        default=os.environ.get("COMPUTE_SYSTEM", "local"),
    )
    for name in SERVER_MODULES:
        parser.add_argument(
            f"--{name}-python",
            default=None,
            help=f"Default: {PYTHON_ENV_VARS[name]} or current Python.",
        )
        parser.add_argument(
            f"--{name}-port",
            type=int,
            default=DEFAULT_PORTS[name],
            help=f"HTTP port (default: {DEFAULT_PORTS[name]}).",
        )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for every server to listen (default: 300).",
    )
    return parser


def _handle_sigterm(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def main() -> None:
    args = build_parser().parse_args()
    if args.startup_timeout <= 0:
        raise SystemExit("--startup-timeout must be greater than zero")

    previous_sigterm = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        try:
            returncode = supervise(args)
        except (RuntimeError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            returncode = 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
