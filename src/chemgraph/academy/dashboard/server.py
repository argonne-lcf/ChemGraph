from __future__ import annotations

import argparse
import socket
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

from chemgraph.academy.observability.event_log import read_events
from chemgraph.academy.observability.run_files import read_json_file
from chemgraph.academy.observability.run_artifacts import write_run_artifacts

_STATIC_CACHE: dict[str, bytes] = {}


def _static_file(name: str, content_type: str) -> tuple[bytes, str]:
    if name not in _STATIC_CACHE:
        resource = files('chemgraph.academy.dashboard').joinpath(
            'static',
            name,
        )
        _STATIC_CACHE[name] = resource.read_bytes()
    return _STATIC_CACHE[name], content_type


class DashboardHandler(BaseHTTPRequestHandler):
    run_dir: Path

    def do_GET(self) -> None:
        path = self.path.split('?', 1)[0]
        if path in {'/', '/index.html'}:
            body, content_type = _static_file('index.html', 'text/html; charset=utf-8')
            self._send_bytes(200, body, content_type)
            return
        if path == '/static/app.js':
            body, content_type = _static_file(
                'app.js',
                'application/javascript; charset=utf-8',
            )
            self._send_bytes(200, body, content_type)
            return
        if path == '/api/status':
            self._send_json(200, status_payload(self))
            return
        if path == '/api/events':
            self._send_json(200, events_payload(self.run_dir))
            return
        if path == '/api/snapshot':
            self._send_json(200, snapshot(self))
            return
        self._send_json(404, {'error': 'not found'})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode('utf-8')
        self._send_bytes(status, body, 'application/json')

    def _send(self, status: int, body: str, content_type: str) -> None:
        self._send_bytes(status, body.encode('utf-8'), content_type)

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        try:
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            return


def snapshot(handler: DashboardHandler) -> dict[str, Any]:
    data = status_payload(handler)
    data.update(events_payload(handler.run_dir))
    return data


def status_payload(handler: DashboardHandler) -> dict[str, Any]:
    run_dir = handler.run_dir
    status_path = run_dir / "status.json"
    status: dict[str, Any] = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}
    artifacts = write_run_artifacts(run_dir)
    manifest = read_json_file(run_dir / "manifest.json", default={})
    updated = status.get("updated") or status.get("timestamp")
    schema = (
        status.get("mode")
        or (manifest.get("mode") if isinstance(manifest, dict) else None)
        or "canonical_events"
    )
    return {
        "run_dir": str(run_dir),
        "updated": updated,
        "schema": schema,
        "status": status,
        "placement": artifacts["placement"],
        "summary": artifacts["summary"],
    }


def events_payload(run_dir: Path) -> dict[str, Any]:
    events = [
        event.model_dump(mode="json") for event in read_events(run_dir / "events.jsonl")
    ]
    return {
        "run_dir": str(run_dir),
        "events": events,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return serve_dashboard(
        run_dir=Path(args.run_dir).resolve(),
        host=args.host,
        port=args.port,
    )


def serve_dashboard(*, run_dir: Path, host: str, port: int) -> int:
    DashboardHandler.run_dir = run_dir
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Serving {run_dir} at http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
