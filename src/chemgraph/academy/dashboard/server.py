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


def _iter_site_dirs(run_dir: Path) -> list[tuple[str | None, Path]]:
    """Return ``[(site_name, site_dir)]`` for the dashboard to read from.

    Single-site mode (the legacy layout): ``run_dir/events.jsonl``
    exists at the top level. Returns ``[(None, run_dir)]`` and the
    dashboard behaves exactly as it did pre-federation.

    Multi-site mode (federated, per-site subdirs): ``run_dir`` does
    NOT contain ``events.jsonl`` itself; instead it contains one
    subdir per site, each with its own ``events.jsonl``. Returns
    ``[(name, subdir), ...]`` for every subdir that looks like a
    site mirror. The ``site_name`` is used to tag events and
    namespace per-site status / placement / summary in the merged
    payload.

    Detection heuristic: ``events.jsonl`` at the top level wins
    (single-site, even if subdirs exist for some reason). Otherwise
    every immediate subdir whose own ``events.jsonl`` exists OR
    which carries a ``dashboard_metadata.json`` (written per-site by
    the launcher) counts as a site. The metadata check catches the
    pre-startup window where a site is up but no events have been
    written yet, so federated dashboards don't briefly look like
    "empty single-site" while waiting on the first event.
    """
    if (run_dir / "events.jsonl").exists():
        return [(None, run_dir)]
    sites: list[tuple[str | None, Path]] = []
    if run_dir.is_dir():
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir():
                continue
            if (child / "events.jsonl").exists() or (child / "dashboard_metadata.json").exists():
                sites.append((child.name, child))
    if not sites:
        # Neither single-site events nor any recognizable site subdirs.
        # Fall back to treating the dir as single-site so the empty-run
        # case (just-created dir, no events yet) doesn't break.
        return [(None, run_dir)]
    return sites


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


def _site_status(site_dir: Path) -> dict[str, Any]:
    """Compose one site's ``status`` slice (status.json + placement + summary)."""
    status_path = site_dir / "status.json"
    status: dict[str, Any] = {}
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}
    artifacts = write_run_artifacts(site_dir)
    manifest = read_json_file(site_dir / "manifest.json", default={})
    updated = status.get("updated") or status.get("timestamp")
    schema = (
        status.get("mode")
        or (manifest.get("mode") if isinstance(manifest, dict) else None)
        or "canonical_events"
    )
    return {
        "updated": updated,
        "schema": schema,
        "status": status,
        "placement": artifacts["placement"],
        "summary": artifacts["summary"],
    }


def status_payload(handler: DashboardHandler) -> dict[str, Any]:
    run_dir = handler.run_dir
    sites = _iter_site_dirs(run_dir)

    if len(sites) == 1 and sites[0][0] is None:
        # Single-site / legacy layout -- preserve exact pre-federation
        # payload shape so existing JS clients keep working.
        site_data = _site_status(run_dir)
        return {
            "run_dir": str(run_dir),
            **site_data,
        }

    # Federated layout: nest per-site status under ``sites`` and add a
    # top-level ``updated`` reflecting the most recent per-site update
    # so the dashboard header has something to display.
    sites_data: dict[str, dict[str, Any]] = {}
    latest_updated: float | None = None
    for site_name, site_dir in sites:
        assert site_name is not None
        sites_data[site_name] = _site_status(site_dir)
        site_updated = sites_data[site_name].get("updated")
        if isinstance(site_updated, (int, float)):
            latest_updated = (
                site_updated if latest_updated is None
                else max(latest_updated, float(site_updated))
            )
    return {
        "run_dir": str(run_dir),
        "updated": latest_updated,
        "schema": "canonical_events",
        "sites": sites_data,
    }


def events_payload(run_dir: Path) -> dict[str, Any]:
    sites = _iter_site_dirs(run_dir)

    if len(sites) == 1 and sites[0][0] is None:
        # Single-site / legacy layout -- preserve exact event payload
        # shape (no per-event ``site`` tag).
        events = [
            event.model_dump(mode="json")
            for event in read_events(run_dir / "events.jsonl")
        ]
        return {
            "run_dir": str(run_dir),
            "events": events,
        }

    # Federated: tag each event with its site and merge in timestamp
    # order so the dashboard can render a single interleaved stream.
    merged: list[dict[str, Any]] = []
    for site_name, site_dir in sites:
        for event in read_events(site_dir / "events.jsonl"):
            payload = event.model_dump(mode="json")
            payload["site"] = site_name
            merged.append(payload)
    # Sort by timestamp when available; events lacking a timestamp
    # sink to the bottom rather than throw off the ordering of
    # well-formed ones.
    def _ts(e: dict[str, Any]) -> float:
        v = e.get("timestamp") or e.get("time")
        try:
            return float(v) if v is not None else float("inf")
        except (TypeError, ValueError):
            return float("inf")
    merged.sort(key=_ts)
    return {
        "run_dir": str(run_dir),
        "events": merged,
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
