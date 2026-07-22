from __future__ import annotations

import argparse
import json
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any

from chemgraph.academy.dashboard.launch_handler import LaunchController
from chemgraph.academy.observability.event_log import read_events
from chemgraph.academy.observability.run_artifacts import write_run_artifacts
from chemgraph.academy.observability.run_files import read_json_file

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
            if (
                (child / "events.jsonl").exists()
                or (child / "dashboard_metadata.json").exists()
                or any(child.glob("dashboard_metadata.*.json"))
            ):
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
    # Launch-button feature: opt-in via `--enable-launch-buttons` on the
    # dashboard CLI. When ``launch_controller`` is None, the related
    # endpoints return 404 / 503 and the frontend doesn't render the
    # buttons. Keeps existing behavior unchanged for operators who just
    # want the read-only dashboard.
    launch_controller: LaunchController | None = None
    launch_config: dict[str, Any] | None = None  # snapshot of cli args

    def do_GET(self) -> None:
        path = self.path.split('?', 1)[0]
        if path in {'/', '/index.html'}:
            body, content_type = _static_file('index.html', 'text/html; charset=utf-8')
            self._send_bytes(200, body, content_type)
            return
        if path.startswith('/static/') and path.endswith('.js'):
            # Guard against path traversal: only the basename is used
            # to look up the packaged resource. importlib.resources
            # will raise FileNotFoundError for anything not bundled.
            name = path[len('/static/'):]
            if '/' in name or name.startswith('.'):
                self._send_json(404, {'error': 'not found'})
                return
            try:
                body, content_type = _static_file(
                    name, 'application/javascript; charset=utf-8',
                )
            except FileNotFoundError:
                self._send_json(404, {'error': 'not found'})
                return
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
        if path == '/api/launch-config':
            if self.launch_config is None:
                self._send_json(404, {'error': 'launch buttons not enabled'})
                return
            self._send_json(200, self.launch_config)
            return
        if path == '/api/launch-status':
            if self.launch_controller is None:
                self._send_json(404, {'error': 'launch buttons not enabled'})
                return
            self._send_json(200, self.launch_controller.snapshot())
            return
        if path.startswith('/api/site-log/'):
            site = path[len('/api/site-log/'):]
            self._send_json(200, _site_log_payload(self, site))
            return
        if path == '/api/campaigns':
            self._send_json(200, _campaigns_index_payload())
            return
        if path == '/api/engines':
            from chemgraph.academy.runtime.engines import list_engines, DEFAULT_ENGINE_NAME
            self._send_json(200, {
                'engines': list_engines(),
                'default': DEFAULT_ENGINE_NAME,
            })
            return
        if path.startswith('/api/campaign/'):
            name = path[len('/api/campaign/'):]
            self._send_json(200, _campaign_payload_by_name(name))
            return
        if path.startswith('/api/pbs-preview/'):
            site = path[len('/api/pbs-preview/'):]
            self._send_json(200, _pbs_preview_payload(self, site))
            return
        self._send_json(404, {'error': 'not found'})

    def do_POST(self) -> None:
        path = self.path.split('?', 1)[0]
        if self.launch_controller is None or self.launch_config is None:
            self._send_json(404, {'error': 'launch buttons not enabled'})
            return

        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            length = 0
        raw_body = self.rfile.read(length).decode('utf-8') if length else ''
        body: dict[str, Any] = {}
        if raw_body:
            try:
                body = json.loads(raw_body)
            except json.JSONDecodeError:
                self._send_json(400, {'error': 'body must be JSON'})
                return

        if path == '/api/launch':
            try:
                argv = _build_launch_argv(self.launch_config, body)
            except ValueError as e:
                self._send_json(400, {'ok': False, 'detail': str(e)})
                return
            # Reset the controller to the sites THIS launch actually
            # targets. Per-site button passes `sites: ["aurora"]`; the
            # Launch-all button passes `sites: null` meaning "all
            # placed". Without filtering by `sites`, clicking "Launch
            # aurora" would still light both site badges as
            # "submitting" because the reset keyed on every placed site.
            placed = tuple(body.get('site_agents', {}).keys())
            selected = tuple(body.get('sites') or placed)
            self.launch_controller.set_sites(selected)
            # Wipe stale run_dir artifacts BEFORE spawning the launcher
            # so a re-launch of the same campaign name (or a fresh
            # canvas edit that clears agent_status/placement) doesn't
            # inherit prior lm_config/agent_status/placement files.
            # Same-name = override, per user's design intent (2026-07-06).
            wipe_results = _wipe_run_dirs(self, selected)
            ok, msg = self.launch_controller.launch(argv, selected)
            resp: dict[str, Any] = {'ok': ok, 'detail': msg}
            if wipe_results:
                resp['wipe'] = wipe_results
            self._send_json(202 if ok else 409, resp)
            return
        if path == '/api/inject-message':
            # Operator-originated message into an agent mailbox.
            # Kickoff and mid-run nudges both go through this single
            # endpoint (no separate bootstrap concept as of 2026-07-07).
            try:
                argv = _build_inject_argv(self.launch_config, body)
            except ValueError as e:
                self._send_json(400, {'ok': False, 'detail': str(e)})
                return
            ok, msg = self.launch_controller.dispatch_message(argv)
            self._send_json(202 if ok else 409, {'ok': ok, 'detail': msg})
            return
        if path == '/api/qdel':
            self._send_json(200, _qdel_payload(self, body))
            return
        if path == '/api/campaign':
            result = _campaign_edit_payload(self, body)
            status = 200 if result.get('ok') else 400
            self._send_json(status, result)
            return
        if path == '/api/mcp/discover':
            result = _mcp_discover_payload(self, body)
            status = 200 if result.get('ok') else 400
            self._send_json(status, result)
            return
        if path == '/api/launch-preview':
            # Dry-run: return the exact launcher argv + per-site PBS
            # scripts the next Launch would fire, given the canvas
            # body. Frontend shows these as read-only reference so
            # operators know what they're overriding.
            try:
                result = _launch_preview_payload(self, body)
            except Exception as exc:
                # Anything that leaks out of the preview builder is a
                # dashboard-side bug, not something the operator can
                # fix by clicking harder. Surface the type + message so
                # the browser's modal can show it inline instead of
                # rendering an empty section.
                import traceback
                result = {
                    'ok': False,
                    'error': f'{type(exc).__name__}: {exc}',
                    'traceback': traceback.format_exc(),
                }
            status = 200 if result.get('ok') else 400
            self._send_json(status, result)
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
            # No caching: dev iteration on app.js was serving stale JS
            # because the browser cached forever without a hash query.
            self.send_header('Cache-Control', 'no-store')
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
    # Per-site agent_status/<name>.json is written ONLY by the daemon
    # that owns that agent -- so its filenames are the authoritative
    # local-agent set.
    agent_state_dir = site_dir / "agent_status"
    local_agents: list[str] = []
    per_agent_states: list[dict[str, Any]] = []
    if agent_state_dir.is_dir():
        for p in sorted(agent_state_dir.glob("*.json")):
            if not p.is_file():
                continue
            local_agents.append(p.stem)
            try:
                item = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(item, dict):
                per_agent_states.append(item)
    # status.json's ``agents`` array is written by write_status_snapshot,
    # which races between sites that share run_dir (Eagle mount). Last
    # writer wins → one site's view becomes everyone's view, dropping
    # peers. Replace it with the per-agent files (one file per daemon,
    # no shared-write race) so the UI metric stops alternating 1/1 vs 2/2.
    if per_agent_states:
        status = dict(status)  # don't mutate the raw file payload
        status["agents"] = per_agent_states
    return {
        "updated": updated,
        "schema": schema,
        "status": status,
        "placement": artifacts["placement"],
        "summary": artifacts["summary"],
        "local_agents": local_agents,
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


def _site_log_payload(handler: "DashboardHandler", site: str) -> dict[str, Any]:
    """Fetch the per-site PBS / attach log over ssh, return last ~80 lines.

    The launcher writes per-site logs under the run_dir on the HPC's
    shared FS:
      - submit-mode: <run_dir>/<site>.pbs.log
      - attach-mode: <run_dir>/<site>.attach.log

    Launch-buttons feature is submit-only for v1 so try .pbs.log first
    and fall back to .attach.log for completeness.
    """
    cfg = handler.launch_config
    if cfg is None:
        return {'error': 'launch buttons not enabled', 'lines': []}
    if site not in cfg.get('systems', []):
        return {'error': f'unknown site {site!r}', 'lines': []}

    from chemgraph.academy.runtime.profiles import load_system_profile
    from chemgraph.academy.runtime.remote.ssh_transport import ssh_quote, ssh_run

    try:
        profile = load_system_profile(site)
    except Exception as e:
        return {'error': f'profile load failed: {e}', 'lines': []}
    login_host = profile.remote_host
    # Build the remote run_dir the same way the launcher does -- profile
    # default run_root + run_id. Mirrors _resolve_run_dir in
    # remote_launcher.py when args.run_dir is None.
    remote_run_dir = f"{profile.run_root}/{cfg['run_id']}"
    # PBS quirk: ``#PBS -o /path/file`` only writes the file when the
    # job EXITS; during the run, intermediate stdout lives at
    # ~/STDIN.o<jobid> (PBS Pro) or ~/<jobname>.o<jobid> on the login
    # node's home FS. Probe all candidates so the operator can see
    # diagnostic output mid-run, not just post-mortem.
    # Concrete paths get shell-quoted (safe against weird characters).
    # Glob patterns must NOT be quoted -- the remote shell needs to
    # expand them.
    concrete_paths = [
        f"{remote_run_dir}/{site}.pbs.log",
        f"{remote_run_dir}/{site}.attach.log",
    ]
    glob_patterns = [
        "$HOME/STDIN.o*",       # PBS Pro intermediate
        "$HOME/*.o[0-9]*",      # generic PBS intermediate
    ]
    log_paths = concrete_paths + glob_patterns
    try:
        parts = [
            f"echo '=== {p} ==='; tail -80 {ssh_quote(p)} 2>/dev/null || true"
            for p in concrete_paths
        ] + [
            # Glob: unquoted so shell expansion happens. Loop and tail
            # each match individually so we get a `=== <path> ===` per
            # actual file rather than per pattern.
            f"for f in {pat}; do "
            f"  [ -e \"$f\" ] && {{ echo \"=== $f ===\"; tail -80 \"$f\"; }} "
            f"done 2>/dev/null || true"
            for pat in glob_patterns
        ]
        cmd = " ; ".join(parts)
        r = ssh_run(login_host, cmd, timeout_s=15, check=False)
    except Exception as e:
        return {'error': f'ssh failed: {e}', 'lines': [], 'remote_paths': log_paths}
    lines = (r.stdout or '').splitlines()
    return {
        'site': site,
        'login_host': login_host,
        'remote_paths': log_paths,
        'lines': lines[-200:],  # cap so the response stays small
    }


def _campaign_payload_by_name(name: str) -> dict[str, Any]:
    """Same shape as _campaign_payload but for arbitrary campaign name.

    Powers the canvas "Open campaign" dropdown -- lets the frontend
    load any known campaign (shipped or user copy) into the editor
    without a page reload.
    """
    from chemgraph.academy.core.campaign import _load_jsonc
    from chemgraph.academy.core.campaign_editor import resolve_editable

    try:
        resolved = resolve_editable(name)
    except Exception as e:
        return {'error': f'resolve failed: {e}'}
    if not resolved.path.exists():
        return {'error': f'campaign {name!r} not found'}
    try:
        return {
            'path': str(resolved.path),
            'is_user_copy': resolved.is_user_copy,
            'campaign': _load_jsonc(resolved.path),
        }
    except Exception as e:
        return {'error': f'parse failed: {e}', 'path': str(resolved.path)}


def _campaigns_index_payload() -> dict[str, Any]:
    """List all known campaigns: shipped names + user copies."""
    from chemgraph.academy.campaigns import list_campaigns
    from chemgraph.academy.core.campaign_editor import user_campaigns_root

    shipped = sorted(list_campaigns())
    user_dir = user_campaigns_root()
    user_copies: list[str] = []
    if user_dir.is_dir():
        for child in sorted(user_dir.iterdir()):
            if child.is_dir() and (child / 'campaign.jsonc').exists():
                user_copies.append(child.name)
    # De-dupe: campaigns that have a user copy still appear in shipped
    # (that's fine; frontend chooses one).
    return {
        'shipped': shipped,
        'user_copies': user_copies,
    }


# Simple in-memory cache: (site, name, command) -> tools list. Invalidated
# by passing refresh=True in the request body. Cache lifetime = dashboard
# process. MCP servers are declared per-campaign so cache size is small.
# "site" may be a real HPC name ("crux") or "__local__" for laptop
# discovery. This way switching between HPCs on the canvas doesn't clobber
# each other's cached tool lists.
_MCP_TOOLS_CACHE: dict[tuple[str, str, str], list[dict[str, Any]]] = {}


def _mcp_discover_payload(
    handler: "DashboardHandler",
    body: dict[str, Any],
) -> dict[str, Any]:
    """Return the tool list advertised by an MCP server.

    Body::
        {"name": "...", "command": "...",
         "hpc":  "crux" | null,    # SSH into this HPC's login node
         "refresh": false}

    Response::
        {"ok": true, "site": "crux" | "__local__",
         "tools": [{"name": ..., "description": ...}, ...],
         "cached": true|false}

    Discovery happens where the tool will actually run:

    - ``hpc`` set  → SSH to the site's login node and run the discovery
      probe in the compute venv there. This is the right answer because
      the ``command`` field in the campaign is authored for the compute
      node, not for the laptop.

    - ``hpc`` null → local subprocess discovery. Kept as an escape hatch
      for developing on the laptop with laptop-installed MCP servers.

    Feeds the canvas allowed_tools checklist.
    """
    name = body.get('name')
    command = body.get('command')
    hpc = body.get('hpc') or None
    if not isinstance(name, str) or not isinstance(command, str) or not name or not command:
        return {'ok': False, 'error': 'body must have "name" and "command" strings'}
    if hpc is not None and not isinstance(hpc, str):
        return {'ok': False, 'error': '"hpc" must be a string or omitted'}

    site_key = hpc or "__local__"
    refresh = bool(body.get('refresh'))
    key = (site_key, name, command)
    if not refresh and key in _MCP_TOOLS_CACHE:
        return {'ok': True, 'site': site_key, 'tools': _MCP_TOOLS_CACHE[key], 'cached': True}

    try:
        if hpc:
            tools = _discover_mcp_tools_remote(hpc, name, command)
        else:
            tools = _discover_mcp_tools_local(handler, name, command)
    except Exception as e:
        return {'ok': False, 'site': site_key, 'error': f'discovery failed: {e}'}

    _MCP_TOOLS_CACHE[key] = tools
    return {'ok': True, 'site': site_key, 'tools': tools, 'cached': False}


def _discover_mcp_tools_local(
    handler: "DashboardHandler",
    name: str,
    command: str,
) -> list[dict[str, Any]]:
    import asyncio
    from chemgraph.academy.runtime.mcp_supervisor import discover_mcp_tools
    return asyncio.run(
        discover_mcp_tools(name=name, command=command, run_dir=handler.run_dir),
    )


def _discover_mcp_tools_remote(
    hpc: str,
    name: str,
    command: str,
) -> list[dict[str, Any]]:
    """SSH into the HPC's login node, spawn the MCP server there, return
    the tools it advertises.

    Uses the same discovery snippet the launcher would run at compute
    time, so preview and launch see identical tool lists. Runs inside
    a tempdir on the remote so logs don't accumulate.
    """
    import json
    import shlex
    from chemgraph.academy.runtime.profiles import load_system_profile
    from chemgraph.academy.runtime.remote.ssh_transport import ssh_run

    profile = load_system_profile(hpc)
    host = profile.remote_host

    # The remote python snippet: import discover_mcp_tools, run it with
    # the provided (name, command), emit a MARKER-fenced JSON blob so
    # we can parse it even if stray logs land on stdout.
    snippet = (
        "import asyncio, json, tempfile, pathlib\n"
        "from chemgraph.academy.runtime.mcp_supervisor import discover_mcp_tools\n"
        f"NAME = {json.dumps(name)}\n"
        f"CMD = {json.dumps(command)}\n"
        "async def main():\n"
        "    with tempfile.TemporaryDirectory() as td:\n"
        "        tools = await discover_mcp_tools(name=NAME, command=CMD,\n"
        "            run_dir=pathlib.Path(td))\n"
        "        print('===MCP_TOOLS_BEGIN===')\n"
        "        print(json.dumps([\n"
        "            {'name': t['name'], 'description': t.get('description', '')}\n"
        "            for t in tools]))\n"
        "        print('===MCP_TOOLS_END===')\n"
        "asyncio.run(main())\n"
    )
    remote_cmd = f"python -c {shlex.quote(snippet)}"

    r = ssh_run(host, remote_cmd, timeout_s=180.0, check=False)
    if r.returncode != 0:
        raise RuntimeError(
            f"remote discovery on {hpc!r} exit={r.returncode}: "
            f"stderr={(r.stderr or '').strip()[-500:]}"
        )
    out = r.stdout or ""
    try:
        begin = out.index("===MCP_TOOLS_BEGIN===") + len("===MCP_TOOLS_BEGIN===")
        end = out.index("===MCP_TOOLS_END===")
    except ValueError:
        raise RuntimeError(
            f"remote discovery on {hpc!r}: markers missing in stdout. "
            f"Last 500 chars: {out.strip()[-500:]}"
        )
    return json.loads(out[begin:end].strip())


def _campaign_edit_payload(
    handler: "DashboardHandler",
    body: dict[str, Any],
) -> dict[str, Any]:
    """Apply one action to the campaign + auto-rsync to every HPC.

    Body shape::
        {
          "campaign": "<name>",       # optional; defaults to launch_config.campaign
          "action":   "edit_agent_field" | "add_agent" | "remove_agent"
                    | "rename_agent"   | "set_edge"    | "edit_campaign_field"
                    | "create_blank"   | "clone"        # clone: params.source
                    | "add_mcp_server" | "remove_mcp_server",
          "params":   { ... per-action ... }
        }

    Guards:
    - Refuses when a launcher subprocess is running; a spawn-time
      read of the campaign could split-brain between ranks.
    - Editor module enforces field-level validation.

    Response::
        {
          "ok": true,
          "path": "...",
          "campaign": { ... freshly-loaded doc ... },
          "rsync": [{"site": "polaris", "ok": true, "detail": "..."}, ...],
        }
    """
    ctrl = handler.launch_controller
    cfg = handler.launch_config
    if ctrl is None or cfg is None:
        return {'ok': False, 'error': 'launch buttons not enabled'}
    if ctrl.snapshot().get('launcher_running'):
        return {'ok': False, 'error': 'launcher is running; edits refused'}

    action = body.get('action')
    params = body.get('params') or {}
    campaign_name = body.get('campaign')
    if not campaign_name:
        return {'ok': False, 'error': 'body must have "campaign" (pick one on the canvas)'}
    if not action:
        return {'ok': False, 'error': 'body must have "action"'}

    from chemgraph.academy.core.campaign import _load_jsonc
    from chemgraph.academy.core.campaign_editor import (
        apply_action,
        clone_campaign,
        create_blank_campaign,
    )

    try:
        if action == 'create_blank':
            result = create_blank_campaign(campaign_name)
        elif action == 'clone':
            source = params.get('source')
            if not isinstance(source, str) or not source:
                return {'ok': False, 'error': 'clone action requires params.source'}
            result = clone_campaign(source_name=source, new_name=campaign_name)
        else:
            result = apply_action(
                campaign_name=campaign_name,
                action=action,
                params=params,
            )
    except FileExistsError as e:
        return {'ok': False, 'error': str(e)}
    except ValueError as e:
        return {'ok': False, 'error': str(e)}

    try:
        fresh = _load_jsonc(result.path)
    except Exception as e:
        fresh = {'error': f'reload failed: {e}'}

    rsync_results = _rsync_user_campaign(handler, campaign_name, result.path)
    return {
        'ok': True,
        'path': str(result.path),
        'campaign': fresh,
        'rsync': rsync_results,
    }


def _rsync_user_campaign(
    handler: "DashboardHandler",
    campaign_name: str,
    local_path: Path,
) -> list[dict[str, Any]]:
    """scp the freshly-written user campaign to every HPC in launch_config.

    Target path: ``<profile.remote_root>/user-campaigns/<name>/campaign.jsonc``
    -- matches CHEMGRAPH_USER_CAMPAIGNS_ROOT exported by the PBS
    script, so compute-side resolve_campaign picks up the edit on the
    next launch.
    """
    cfg = handler.launch_config
    results: list[dict[str, Any]] = []
    if not cfg:
        return results

    from chemgraph.academy.runtime.profiles import load_system_profile
    from chemgraph.academy.runtime.remote.ssh_transport import (
        scp_upload,
        ssh_quote,
        ssh_run,
    )

    # rsync to every declared system, not just those with agents in
    # the current draft -- placement can change per launch, but the
    # user-copy needs to be present everywhere before it does.
    for site in cfg.get('systems', []):
        try:
            profile = load_system_profile(site)
        except Exception as e:
            results.append({'site': site, 'ok': False, 'detail': f'profile load: {e}'})
            continue
        remote_dir = f"{profile.remote_root}/user-campaigns/{campaign_name}"
        remote_path = f"{remote_dir}/campaign.jsonc"
        try:
            # ssh_run + scp_upload both reuse the same ControlMaster
            # socket the dashboard-launcher warmed at startup -- without
            # this, every save re-prompts MFA on both sites.
            ssh_run(
                profile.remote_host,
                f"mkdir -p {ssh_quote(remote_dir)}",
                timeout_s=30,
            )
            r = scp_upload(
                str(local_path),
                profile.remote_host,
                remote_path,
                timeout_s=60,
            )
            ok = r.returncode == 0
            detail = (
                (r.stdout or '').strip()
                or (r.stderr or '').strip()
                or 'scp ok'
            )
            results.append({
                'site': site,
                'ok': ok,
                'detail': detail,
                'remote_path': remote_path,
            })
        except Exception as e:
            results.append({'site': site, 'ok': False, 'detail': f'ssh/scp: {e}'})
    return results


def _pbs_preview_payload(
    handler: "DashboardHandler",
    site: str,
) -> dict[str, Any]:
    """Return a starter PBS template for ``site`` with ${VAR} placeholders
    UNexpanded so the operator can edit them in a textarea and save
    back via set_pbs_script.

    Uses the site's submit_defaults from launch_config for the concrete
    resource lines the built-in template would emit (queue, walltime,
    etc.) then leaves the machine-generated ${SPAWN_INVOCATION} /
    ${ENV_EXPORTS} / ${RUN_DIR} / ${BUNDLE_ROOT} / ${ENV_SCRIPT} as
    literal placeholders. That way an operator who wants to change
    walltime can just edit `#PBS -l walltime=...` while leaving the
    spawn-site invocation alone.
    """
    cfg = handler.launch_config
    if not cfg or site not in cfg.get('site_specs', {}):
        return {'ok': False, 'error': f'unknown site {site!r}'}
    spec = cfg['site_specs'][site]
    filesystems = spec.get('filesystems') or ''
    lines = [
        "#!/bin/bash",
        "#PBS -A ${PROJECT}",
        f"#PBS -q {spec['queue']}",
        f"#PBS -l select={spec.get('nodes', 1)},walltime={spec['walltime']}",
    ]
    if filesystems:
        lines.append(f"#PBS -l filesystems={filesystems}")
    lines += [
        "#PBS -j oe",
        "#PBS -o ${RUN_DIR}/${SITE}.pbs.log",
        "",
        "set -e",
        "mkdir -p ${RUN_DIR}",
        "${ENV_EXPORTS}",
        "source ${ENV_SCRIPT}",
        "cd ${BUNDLE_ROOT}",
        "",
        "${SPAWN_INVOCATION}",
        "",
    ]
    return {
        'ok': True,
        'site': site,
        'template': "\n".join(lines),
        'available_vars': [
            'PROJECT', 'QUEUE', 'WALLTIME', 'NODES', 'FILESYSTEMS',
            'RUN_DIR', 'BUNDLE_ROOT', 'ENV_SCRIPT', 'ENV_EXPORTS',
            'SPAWN_INVOCATION', 'SITE', 'RUN_ID', 'CAMPAIGN',
        ],
    }


def _launch_preview_payload(
    handler: "DashboardHandler",
    body: dict[str, Any],
) -> dict[str, Any]:
    """Return the exact launcher argv + rendered PBS script(s) the next
    /api/launch would fire, given the current canvas body.

    Body: same shape as /api/launch (campaign + site_agents + optional
    sites filter).

    Response::
      {
        "ok": true,
        "argv": ["swarm", "launch", "--", ...],
        "argv_string": "swarm launch -- ...",  # shell-safe
        "sites": {
          "aurora": {
            "pbs_default": "<built-in template fully expanded>",
            "pbs_effective": "<what actually gets qsub'd>",
            "has_override": true|false,
          }, ...
        }
      }

    Reads canvas-authored launch_defaults so the operator's edits show
    up in `pbs_effective`. `pbs_default` is always the built-in
    template (what you'd get if you cleared the override).
    """
    cfg = handler.launch_config
    if cfg is None:
        return {'ok': False, 'error': 'launch buttons not enabled'}

    try:
        argv = _build_launch_argv(cfg, body)
    except ValueError as e:
        return {'ok': False, 'error': str(e)}

    import shlex
    from chemgraph.academy.runtime.profiles import load_system_profile
    from chemgraph.academy.runtime.remote.remote_launcher import (
        _collect_remote_env, _resolve_venv_activate,
    )
    from chemgraph.academy.runtime.remote.site_spec import parse_site
    from chemgraph.academy.runtime.remote.submit_backend import (
        SubmitConfig, render_pbs_script,
    )

    argv_string = "swarm launch -- " + " ".join(
        shlex.quote(x) for x in argv
    )

    # Rebuild per-site SubmitConfigs from the argv we just generated
    # so the rendered PBS matches what qsub would receive verbatim.
    # Parse --site tokens back out to feed SubmitConfig.
    site_specs: dict[str, dict[str, str]] = {}
    launch_defaults = _read_campaign_launch_defaults(body.get('campaign') or '')
    per_site_overrides = launch_defaults.get('per_site_overrides') or {}

    # Extract --site values from argv (paired with the flag).
    site_raws: list[str] = []
    for i, tok in enumerate(argv):
        if tok == '--site' and i + 1 < len(argv):
            site_raws.append(argv[i + 1])

    exchange_type = 'http'  # matches launcher default
    remote_env = _collect_remote_env(exchange_type=exchange_type)
    # CHEMGRAPH_USER_CAMPAIGNS_ROOT: launcher sets this per-site but
    # for preview we just need something plausible so ${ENV_EXPORTS}
    # renders meaningfully.
    class _StubArgs:
        venv_activate = None

    result_sites: dict[str, dict[str, Any]] = {}
    for raw in site_raws:
        spec = parse_site(raw)
        profile = load_system_profile(spec.name)
        # remote_env with the per-site campaigns-root override, as the
        # real launcher would set it.
        env = dict(remote_env)
        env.setdefault(
            "CHEMGRAPH_USER_CAMPAIGNS_ROOT",
            f"{profile.remote_root}/user-campaigns",
        )
        run_dir = str(Path(profile.run_root) / cfg['run_id'])
        bundle_root = spec.bundle_root or cfg['bundle_root']
        venv_activate = _resolve_venv_activate(_StubArgs(), spec)  # type: ignore[arg-type]

        common_kwargs = dict(
            site=spec,
            run_id=cfg['run_id'],
            campaign=body.get('campaign') or '',
            login_host=profile.remote_host,
            bundle_root=bundle_root,
            env_script=venv_activate,
            run_dir=run_dir,
            exchange_type=exchange_type,
            http_exchange_url=None,
            project=cfg['project'],
            remote_env=env,
        )
        default_cfg = SubmitConfig(**common_kwargs)  # type: ignore[arg-type]
        try:
            pbs_default = render_pbs_script(default_cfg)
        except (ValueError, AssertionError) as exc:
            pbs_default = f"(default template render failed: {exc})"

        override_script = (per_site_overrides.get(spec.name) or {}).get('pbs_script')
        if override_script:
            override_cfg = SubmitConfig(
                **common_kwargs, pbs_script_template=override_script,  # type: ignore[arg-type]
            )
            try:
                pbs_effective = render_pbs_script(override_cfg)
            except ValueError as exc:
                pbs_effective = f"(override render failed: {exc})"
        else:
            pbs_effective = pbs_default

        # Also expose the un-substituted template. That's what the
        # editable pane should seed from -- if the operator's saved
        # override contains ${RUN_DIR} etc. literally, next launch's
        # substitution will fill in the CURRENT run_dir instead of
        # pinning yesterday's stale one.
        default_template_payload = _pbs_preview_payload(handler, spec.name)
        pbs_default_template = default_template_payload.get('template', pbs_default)

        result_sites[spec.name] = {
            'pbs_default': pbs_default,               # substituted; for the read-only reference pane
            'pbs_default_template': pbs_default_template,  # unsubstituted; seeds the editable pane
            'pbs_effective': pbs_effective,
            'has_override': bool(override_script),
        }

    return {
        'ok': True,
        'argv': argv,
        'argv_string': argv_string,
        'sites': result_sites,
    }


def _wipe_run_dirs(
    handler: "DashboardHandler",
    sites: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Delete each site's ``<run_root>/<run_id>`` before a fresh launch.

    Rationale: canvas-driven iteration means the same run_id is reused
    across attempts. Without a wipe, a second launch inherits the
    prior attempt's ``agent_status/``, ``placement.json``,
    ``lm_config.<site>.json`` -- readers see phantom agents from the
    old placement and daemons dial the wrong relay. Per-launch wipe
    matches operator intent: same campaign name = start-over.

    Uses the ``mv <run_dir> <trash>/... && rm -rf <trash>`` pattern
    (safe against concurrent readers) inherited from the retired
    ``--overwrite-run`` path. Skips sites whose profile can't be
    loaded (unknown site name is a user error we don't want to hide).
    """
    cfg = handler.launch_config
    if not cfg or not sites:
        return []

    from chemgraph.academy.runtime.profiles import load_system_profile
    from chemgraph.academy.runtime.remote.ssh_transport import ssh_quote, ssh_run

    run_id = cfg['run_id']
    if not run_id or '/' in run_id or run_id in {'.', '..'}:
        return [{'site': '*', 'ok': False, 'detail': f'unsafe run_id {run_id!r}'}]

    results: list[dict[str, Any]] = []
    for site in sites:
        try:
            profile = load_system_profile(site)
        except Exception as e:
            results.append({'site': site, 'ok': False, 'detail': f'profile load failed: {e}'})
            continue
        run_root = profile.run_root
        # Move-then-rm so an in-flight rsync/reader can't race the
        # delete. Retries the rm a few times because the compute-node
        # bind-mount sometimes holds a stale fd for a beat after the mv.
        script = (
            f"set -euo pipefail; "
            f"run_root={ssh_quote(run_root)}; "
            f"run_id={ssh_quote(run_id)}; "
            f'case "$run_id" in ""|.|..|*/*) echo "unsafe run id" >&2; exit 2;; esac; '
            f'run_dir="$run_root/$run_id"; '
            f'trash_root="$run_root/.deleted-runs"; '
            f'if [ -e "$run_dir" ]; then '
            f'  mkdir -p "$trash_root"; '
            f'  trash_dir="$trash_root/${{run_id}}.$(date +%Y%m%d%H%M%S).$$"; '
            f'  mv -- "$run_dir" "$trash_dir"; '
            f'  for delay in 0 1 2 5 10; do '
            f'    sleep "$delay"; '
            f'    if rm -rf -- "$trash_dir" 2>/dev/null; then break; fi; '
            f'  done; '
            f'fi; '
            f'mkdir -p "$run_dir"'
        )
        try:
            r = ssh_run(profile.remote_host, script, timeout_s=60, check=False)
            ok = r.returncode == 0
            detail = (r.stderr or r.stdout or '').strip() or 'wiped'
            results.append({'site': site, 'ok': ok, 'detail': detail})
        except Exception as e:
            results.append({'site': site, 'ok': False, 'detail': f'ssh: {e}'})
    return results


def _qdel_payload(handler: "DashboardHandler", body: dict[str, Any]) -> dict[str, Any]:
    """qdel selected sites' PBS jobs over ssh.

    body: {"sites": ["polaris", "crux"]}  # defaults to every site with
                                          # a known job_id
    """
    ctrl = handler.launch_controller
    cfg = handler.launch_config
    if ctrl is None or cfg is None:
        return {'ok': False, 'error': 'launch buttons not enabled', 'results': []}

    from chemgraph.academy.runtime.profiles import load_system_profile
    from chemgraph.academy.runtime.remote.ssh_transport import ssh_quote, ssh_run

    snap = ctrl.snapshot()
    requested = body.get('sites') or list(snap['sites'].keys())
    results: list[dict[str, Any]] = []
    for site in requested:
        site_snap = snap['sites'].get(site)
        if not site_snap or not site_snap.get('job_id'):
            results.append({'site': site, 'ok': False, 'detail': 'no job_id known'})
            continue
        try:
            profile = load_system_profile(site)
        except Exception as e:
            results.append({'site': site, 'ok': False, 'detail': f'profile load failed: {e}'})
            continue
        try:
            r = ssh_run(
                profile.remote_host,
                f"qdel {ssh_quote(site_snap['job_id'])}",
                timeout_s=30,
                check=False,
            )
            ok = r.returncode == 0
            detail = (r.stdout or '').strip() or (r.stderr or '').strip() or 'qdel sent'
            results.append({
                'site': site, 'job_id': site_snap['job_id'],
                'ok': ok, 'detail': detail,
            })
        except Exception as e:
            results.append({'site': site, 'ok': False, 'detail': f'ssh failed: {e}'})
    return {'ok': all(r['ok'] for r in results), 'results': results}


def _build_launch_argv(
    config: dict[str, Any],
    body: dict[str, Any],
) -> list[str]:
    """Translate launch-config + per-request body into the launcher's argv.

    body MUST include::
        {
          "campaign": "<name>",                # from canvas
          "site_agents": {"aurora": ["a1"],    # from canvas swimlanes
                          "crux":   ["a2"]},
          "sites":         ["aurora", "crux"], # optional subset filter
        }

    Reads ``launch_defaults.extra_launcher_argv`` (shlex-tokenised)
    and ``launch_defaults.per_site_overrides.<site>.pbs_script`` from
    the campaign so canvas-authored resource knobs and custom PBS
    scripts reach the launcher subprocess.

    Raises ValueError if campaign or site_agents missing -- these are
    canvas-authored, not startup defaults.
    """
    import shlex

    campaign = body.get('campaign')
    site_agents = body.get('site_agents') or {}
    if not campaign:
        raise ValueError("body missing 'campaign' (pick one on the canvas)")
    if not isinstance(site_agents, dict) or not site_agents:
        raise ValueError("body missing 'site_agents' (drag agents onto swimlanes)")
    selected = body.get('sites') or list(site_agents.keys())

    launch_defaults = _read_campaign_launch_defaults(campaign)
    per_site_overrides = launch_defaults.get('per_site_overrides') or {}

    argv = [
        "--run-id", config['run_id'],
        "--campaign", campaign,
        "--bundle-root", config['bundle_root'],
        "--project", config['project'],
    ]
    for site_name in selected:
        agents = site_agents.get(site_name)
        if not agents:
            continue
        if site_name not in config['site_specs']:
            raise ValueError(
                f"site {site_name!r} is not in --system (dashboard startup)"
            )
        spec = config['site_specs'][site_name]
        kvs = [
            f"queue={spec['queue']}",
            f"walltime={spec['walltime']}",
            f"nodes={spec.get('nodes', 1)}",
            f"agents={','.join(agents)}",
            f"filesystems={spec['filesystems']}",
        ]
        if spec.get('bundle_root_override'):
            kvs.append(f"bundle_root={spec['bundle_root_override']}")
        argv.extend(["--site", f"{site_name}:" + ";".join(kvs)])
        # Per-site custom PBS script: materialise to /tmp so the
        # launcher CLI can read it with a path argument.
        site_override = per_site_overrides.get(site_name) or {}
        script = site_override.get('pbs_script')
        if script:
            path = _materialise_pbs_override(campaign, site_name, script)
            argv.extend(["--pbs-script-override", f"{site_name}={path}"])
    # No --ready-timeout-s: PBS walltime is the ceiling. See
    # remote_launcher.py for the retirement note.
    # No --auto-bootstrap: kickoff is an operator action via the
    # Inject-a-message panel; the launcher just brings agents up
    # ready and exits.
    # Campaign-level free-form argv tail. Shipped last so operators
    # can override earlier flags if they really want to.
    extra = launch_defaults.get('extra_launcher_argv')
    if isinstance(extra, str) and extra.strip():
        try:
            argv.extend(shlex.split(extra))
        except ValueError as exc:
            raise ValueError(
                f"launch_defaults.extra_launcher_argv is not a valid "
                f"shell string: {exc}",
            ) from exc
    return argv


def _read_campaign_launch_defaults(campaign: str) -> dict[str, Any]:
    """Read the ``launch_defaults`` block from the current on-disk
    copy of ``campaign`` (user copy if present, else shipped).
    Returns ``{}`` if the campaign is missing or the block is absent.
    """
    from chemgraph.academy.core.campaign import _load_jsonc
    from chemgraph.academy.core.campaign_editor import resolve_editable

    try:
        resolved = resolve_editable(campaign)
        data = _load_jsonc(resolved.path)
    except Exception:
        return {}
    block = data.get('launch_defaults')
    return block if isinstance(block, dict) else {}


def _materialise_pbs_override(campaign: str, site: str, script: str) -> str:
    """Write the operator's PBS script to a temp file the launcher can
    read. One file per (campaign, site) so concurrent site launches
    don't stomp each other.
    """
    import tempfile

    tmp_dir = Path(tempfile.gettempdir()) / "chemgraph-dashboard-pbs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path = tmp_dir / f"{campaign}.{site}.pbs.tmpl"
    path.write_text(script, encoding="utf-8")
    return str(path)


def _build_inject_argv(
    config: dict[str, Any],
    body: dict[str, Any],
) -> list[str]:
    """Argv for operator-to-agent message injection.

    Body::
        {"campaign":  "<name>",        # required
         "recipient": "<agent-name>",  # required
         "content":   "<message text>",# required
         "tldr":      "<short>",       # optional
         "reason":    "<free text>",   # optional
         "sender":    "<label>",       # optional (default "operator")
         "kind":      "message|question|nudge"}  # optional
    """
    campaign = body.get('campaign')
    recipient = body.get('recipient')
    content = body.get('content')
    if not campaign:
        raise ValueError("body missing 'campaign' (pick one on the canvas)")
    if not recipient:
        raise ValueError("body missing 'recipient'")
    if not content:
        raise ValueError("body missing 'content'")
    argv = [
        "--run-id", config['run_id'],
        "--campaign", campaign,
        "--exchange-type", "http",
        "--recipient", recipient,
        "--content", content,
    ]
    if body.get('sender'):
        argv += ["--sender", body['sender']]
    if body.get('kind'):
        argv += ["--kind", body['kind']]
    if body.get('tldr'):
        argv += ["--tldr", body['tldr']]
    if body.get('reason'):
        argv += ["--reason", body['reason']]
    return argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--enable-launch-buttons", action="store_true",
        help=(
            "Expose POST /api/launch and POST /api/inject-message so the "
            "dashboard can drive the launcher subprocess. Requires "
            "--run-id, --bundle-root, --project. The campaign + per-site "
            "agent placement are chosen in the canvas at launch time, "
            "not here at dashboard startup."
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--bundle-root")
    parser.add_argument("--project")
    # --system remains here because it enumerates which HPCs THIS
    # laptop is set up to talk to (per-site relays, ssh keys). Site
    # placement of agents happens in the canvas.
    parser.add_argument(
        "--system", default="",
        help=(
            "Comma-separated system profile names this dashboard can "
            "launch against, e.g. 'aurora,crux'. Populates the canvas "
            "swimlanes."
        ),
    )
    return parser.parse_args()


def _build_launch_config(args: argparse.Namespace) -> dict[str, Any]:
    """Startup snapshot of what THIS laptop can launch (infrastructure).

    Campaign selection + per-agent-site placement are canvas-time
    decisions passed in the /api/launch body, not startup args.
    Systems is the enumeration of HPCs this laptop is set up for --
    populates the canvas swimlanes.

    Per-site PBS defaults (queue, walltime, nodes, filesystems) come
    from each system profile's ``submit_defaults`` block. Operators
    override by editing the profile JSON.
    """
    from chemgraph.academy.runtime.profiles import load_system_profile

    for required in ('run_id', 'bundle_root', 'project'):
        if not getattr(args, required):
            raise SystemExit(
                f"--enable-launch-buttons requires --{required.replace('_','-')}",
            )
    systems = [s.strip() for s in (args.system or "").split(',') if s.strip()]
    if not systems:
        raise SystemExit(
            "--enable-launch-buttons requires --system (e.g. 'aurora,crux')",
        )
    site_specs: dict[str, dict[str, Any]] = {}
    for site in systems:
        try:
            profile = load_system_profile(site)
        except Exception as exc:
            raise SystemExit(
                f"--system references profile {site!r} that failed to load: {exc}",
            ) from exc
        site_specs[site] = profile.submit_defaults.model_dump()
    return {
        "run_id": args.run_id,
        "bundle_root": args.bundle_root,
        "project": args.project,
        "systems": systems,
        "site_specs": site_specs,
    }


def main() -> int:
    args = parse_args()
    launch_config: dict[str, Any] | None = None
    launch_controller: LaunchController | None = None
    if args.enable_launch_buttons:
        launch_config = _build_launch_config(args)
        # Empty sites: /api/launch resets to the site_agents keys from
        # the canvas-provided body at launch time.
        launch_controller = LaunchController()
    return serve_dashboard(
        run_dir=Path(args.run_dir).resolve(),
        host=args.host,
        port=args.port,
        launch_config=launch_config,
        launch_controller=launch_controller,
    )


def serve_dashboard(
    *,
    run_dir: Path,
    host: str,
    port: int,
    launch_config: dict[str, Any] | None = None,
    launch_controller: LaunchController | None = None,
) -> int:
    DashboardHandler.run_dir = run_dir
    DashboardHandler.launch_config = launch_config
    DashboardHandler.launch_controller = launch_controller
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"Serving {run_dir} at http://{host}:{port}", flush=True)
    if launch_controller is not None:
        print(
            f"Launch buttons enabled; systems available: "
            f"{launch_config['systems']}",  # type: ignore[index]
            flush=True,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.", flush=True)
    finally:
        if launch_controller is not None:
            launch_controller.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
