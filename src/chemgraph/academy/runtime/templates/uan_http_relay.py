"""Tiny TCP relay used by the dashboard launcher.

Listens on a UAN-visible port and forwards every accepted connection to a
loopback service on the same UAN host. The dashboard launcher pairs this
with a reverse SSH tunnel (Mac argo-shim -> UAN loopback), so compute
nodes can curl http://<uan>:<port>/argoapi/v1 and reach the developer's
local argo-shim.

This file is materialised onto the remote system at runtime by
``chemgraph.academy.runtime.dashboard_launcher.start_relay``. It was
previously expected to live in a sibling ``academy`` source checkout
under ``examples/09-polaris-lm-swarm/``; bundling it here removes the
need for that second checkout on remote hosts.

The implementation is intentionally stdlib-only so the script runs under
any Python interpreter without pip-installing anything on the remote.
"""

from __future__ import annotations

import argparse
import signal
import socket
import sys
import threading


def pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def handle_client(
    client: socket.socket,
    target_host: str,
    target_port: int,
) -> None:
    with client:
        try:
            upstream = socket.create_connection((target_host, target_port))
        except OSError as e:
            print(f'upstream connection failed: {e}', flush=True)
            return
        with upstream:
            left = threading.Thread(target=pump, args=(client, upstream))
            right = threading.Thread(target=pump, args=(upstream, client))
            left.start()
            right.start()
            left.join()
            right.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Relay a UAN-reachable TCP port to a loopback service.',
    )
    parser.add_argument('--listen-host', default='0.0.0.0')
    parser.add_argument('--listen-port', type=int, required=True)
    parser.add_argument('--target-host', default='127.0.0.1')
    parser.add_argument('--target-port', type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.listen_host, args.listen_port))
    server.listen(128)
    print(
        f'relay listening on {args.listen_host}:{args.listen_port} '
        f'-> {args.target_host}:{args.target_port}',
        flush=True,
    )

    def shutdown(_signo: int, _frame: object) -> None:
        # Closing the listen socket inside the handler interrupts
        # server.accept() with EBADF / OSError, which we catch below
        # to fall through to a clean exit. Without this the relay
        # ignores SIGTERM (default action) and orphans the port.
        try:
            server.close()
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        while True:
            try:
                client, addr = server.accept()
            except OSError:
                break
            print(f'accepted connection from {addr[0]}:{addr[1]}', flush=True)
            thread = threading.Thread(
                target=handle_client,
                args=(client, args.target_host, args.target_port),
                daemon=True,
            )
            thread.start()
    finally:
        try:
            server.close()
        except OSError:
            pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
