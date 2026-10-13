"""Launch peers and submit messages; approvals use local operator controls."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import time
import uuid
from pathlib import Path

from chemgraph.distributed.config import load_config
from chemgraph.distributed.peer_tools import build_peer_tool
from chemgraph.distributed.protocol import PeerEnvelope
from chemgraph.distributed.runner import ConfiguredRunner
from chemgraph.distributed.runtime import PeerRuntime
from chemgraph.observability.event_log import EventLog
from chemgraph.observability.run_files import read_json_file, write_json_atomic


def _transport(config, name=None):
    try:
        from academy.exchange.cloud import HttpExchangeFactory
        from chemgraph.distributed.academy import AcademyTransport
    except ImportError as exc:
        raise RuntimeError("Distributed communication requires pip install 'chemgraph[academy]'") from exc
    headers = {}
    for header, variable in config.exchange_header_env.items():
        value = os.environ.get(variable)
        if not value:
            raise ValueError(f"Exchange header {header!r} requires environment variable {variable!r}")
        headers["Authorization" if header.lower() == "authorization" else header] = value
    factory = HttpExchangeFactory(url=config.exchange_url, auth_method=config.exchange_auth_method,
                                  additional_headers=headers)
    return AcademyTransport(exchange_factory=factory,
                            run_id=config.run_id, launch_id=config.launch_id,
                            name=name, timeout_s=config.timeout_s)


async def process_controls(directory: Path, runtime: PeerRuntime, stop: asyncio.Event,
                           *, shutdown_only: bool = False) -> None:
    for path in sorted((directory / "control").glob("*.request.json")):
        try:
            contents = path.read_text()
        except FileNotFoundError:
            continue  # A shutdown request may have been consumed by the monitor.
        result = {"status": "completed"}
        request = None
        try:
            request = json.loads(contents)
            if (request.get("action") == "shutdown") != shutdown_only:
                continue
            if request.get("instance_id") != runtime.instance_id:
                raise ValueError("Operator request belongs to a previous peer process")
            if request["action"] == "shutdown":
                stop.set()
            else:
                current = runtime.status()
                token = request.get("control_token")
                if (not token or token != current["control_token"]
                        or request.get("message_id") != current["current_message_id"]):
                    raise ValueError("Peer state changed; inspect status and submit a new control request")
                if request["action"] == "resume":
                    await runtime.resume(request["response"], control_token=token)
                elif request["action"] == "retry":
                    await runtime.retry(control_token=token)
                else:
                    raise ValueError("Unknown operator action")
        except Exception as exc:
            if shutdown_only and not isinstance(request, dict):
                continue  # The ordinary control pass reports malformed requests.
            result = {"status": "failed", "error": str(exc)}
        write_json_atomic(path.with_name(path.name.replace(".request.json", ".result.json")), result)
        path.unlink()


async def run_peer(config, name: str) -> None:
    import fcntl

    directory = config.peer_dir(name)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "peer.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Peer {name!r} is already running in {directory}") from exc
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        signals = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
                signals.append(sig)
            except (NotImplementedError, RuntimeError):
                pass

        def trace(event, payload):
            EventLog(directory / "events.jsonl").emit(event, run_id=config.run_id, agent_id=name, payload=payload)

        try:
            async with _transport(config, name) as transport:
                spec = config.peers[name]
                peer_tools = [build_peer_tool(transport=transport, run_id=config.run_id,
                                             launch_id=config.launch_id, sender=name,
                                             allowed_peers=spec.allowed_peers)] if spec.allowed_peers else []
                async with ConfiguredRunner(config, name, peer_tools, trace) as runner:
                    senders = tuple(peer for peer, settings in config.peers.items() if name in settings.allowed_peers)
                    runtime = PeerRuntime(name=name, run_id=config.run_id, launch_id=config.launch_id,
                                          allowed_senders=senders, runner=runner, trace=trace)
                    await runtime.start()
                    control_task = None
                    try:
                        await transport.start(runtime.receive, runtime.status)
                        trace("agent_started", {"workflow_type": spec.workflow_type})
                        print(f"Peer {name} ready; artifacts and operator controls: {directory}", flush=True)
                        while not stop.is_set():
                            transport.check_health()
                            write_json_atomic(directory / "status.json", runtime.status())
                            await process_controls(directory, runtime, stop, shutdown_only=True)
                            if control_task is None or control_task.done():
                                if control_task is not None:
                                    control_task.result()
                                control_task = asyncio.create_task(process_controls(directory, runtime, stop))
                            if (spec.idle_timeout_s is not None and runtime.state == "idle"
                                    and runtime.current is None and runtime.inbox.empty()
                                    and time.monotonic() - runtime.last_activity >= spec.idle_timeout_s):
                                break
                            try:
                                await asyncio.wait_for(stop.wait(), timeout=0.25)
                            except asyncio.TimeoutError:
                                pass
                    finally:
                        await runtime.close(timeout_s=config.timeout_s)
                        if control_task is not None:
                            control_task.cancel()
                            await asyncio.gather(control_task, return_exceptions=True)
                        write_json_atomic(directory / "status.json", runtime.status())
                        trace("agent_stopped", runtime.status())
        finally:
            for sig in signals:
                loop.remove_signal_handler(sig)


def queue_control(config, name: str, action: str, response=None) -> Path:
    directory = config.peer_dir(name)
    status = read_json_file(directory / "status.json", default={})
    if not status or status.get("state") in {"stopped", "stopping"}:
        raise RuntimeError("No active local peer status; run this command on the peer's host")
    if action == "resume" and status.get("state") != "waiting_for_user":
        raise RuntimeError("Peer is not waiting for operator input")
    if action == "retry" and status.get("state") != "failed":
        raise RuntimeError("Peer has no failed turn")
    if action != "shutdown" and not status.get("control_token"):
        raise RuntimeError("No current operator control token; refresh peer status before submitting")
    path = directory / "control" / f"{uuid.uuid4()}.request.json"
    write_json_atomic(path, {"action": action, "response": response,
                            "instance_id": status.get("instance_id"),
                            "control_token": status.get("control_token"),
                            "message_id": status.get("current_message_id"), "interrupts": status.get("interrupts", [])})
    return path.with_name(path.name.replace(".request.json", ".result.json"))


async def dispatch(args) -> None:
    config = load_config(args.config)
    config.peer_dir(args.peer)  # Validate the name before opening any connections.
    if args.operation == "run":
        await run_peer(config, args.peer)
    elif args.operation in {"send", "status"}:
        async with _transport(config) as transport:
            if args.operation == "send":
                result = await transport.send(PeerEnvelope(
                    run_id=config.run_id, launch_id=config.launch_id, sender="operator",
                    recipient=args.peer, content=args.message,
                ))
            else:
                result = await transport.peer_status(args.peer)
            print(json.dumps(result, indent=2))
    else:
        response = json.loads(args.response) if args.operation == "resume" else None
        path = queue_control(config, args.peer, args.operation, response)
        print(f"Operator request queued; result will be written to {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chemgraph distributed")
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for command in ("run", "send", "status", "resume", "retry", "shutdown"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", required=True)
        sub.add_argument("--peer", required=True)
        if command == "send":
            sub.add_argument("--message", required=True)
        if command == "resume":
            sub.add_argument("--response", required=True, help="JSON response using the normal ChemGraph approval/input format.")
    try:
        asyncio.run(dispatch(parser.parse_args(argv)))
    except (ValueError, RuntimeError, TimeoutError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
