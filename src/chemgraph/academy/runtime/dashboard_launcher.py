from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shlex
import shutil
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import Path
from typing import Any

from chemgraph.academy.dashboard import serve_dashboard
from chemgraph.academy.campaigns import campaign_launch_defaults
from chemgraph.academy.runtime.profiles import list_builtin_system_profiles
from chemgraph.academy.runtime.profiles import load_system_profile
from chemgraph.academy.runtime.profiles.system import SystemProfile


@dataclasses.dataclass
class _SiteHandle:
    """Per-site state held by the launcher's main loop.

    One of these per ``--system`` value when launching a federated
    dashboard. Single-site invocations build exactly one. The fields
    track everything the cleanup ``finally`` block needs to tear down
    (relay subprocess, ControlMaster ownership) plus the values the
    rsync loop and dashboard server need (local mirror dir, the
    composed ``lm_base_url`` for the site's compute nodes).
    """

    profile: SystemProfile
    remote_host: str
    control_path: str
    local_mirror_dir: Path  # the per-site dir (multi) or top-level (single)
    relay_port: int
    relay_host: str | None = None
    lm_base_url: str | None = None
    relay_process: subprocess.Popen[str] | None = None
    started_master: bool = False


def _parse_systems_list(raw: str) -> tuple[str, ...]:
    """Parse a comma-list of system profile names ('aurora,crux').

    Whitespace-tolerant; trailing commas dropped. Empty input is a
    user error and surfaces a clean message at argparse-resolve time
    rather than later in the setup loop.
    """
    names = tuple(name.strip() for name in raw.split(',') if name.strip())
    if not names:
        raise argparse.ArgumentTypeError(
            "--system requires at least one profile name",
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(
            f"--system has duplicate profile names: {names}",
        )
    return names

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="chemgraph academy dashboard")
    a = p.add_argument
    a("run_id")
    a(
        "--system",
        type=_parse_systems_list,
        default=("aurora",),
        help=(
            "One profile name for a single-site campaign, or a comma "
            "list ('aurora,crux') for a federated dashboard that brings "
            "up per-site relays + rsync mirrors and serves a merged "
            "view. Built-ins: " + ", ".join(list_builtin_system_profiles())
        ),
    )
    a("--campaign", default="mace-ensemble-screening-20")
    a("--lm-connect", choices=("mac-argo-relay", "direct"), default="mac-argo-relay")
    a("--lm-base-url")
    a("--remote-host")
    a("--ssh-control-path")
    a("--keep-ssh-master", action="store_true")
    a("--local-argo-host", default="127.0.0.1")
    a("--local-argo-port", type=int, default=18085)
    a(
        "--reverse-port", type=int, default=18185,
        help=(
            "Reverse-tunnel local port. In multi-site mode each site "
            "gets reverse_port + offset (offset = i for the i-th system)."
        ),
    )
    a("--relay-port", type=int)
    a("--relay-python")
    a("--rsync-interval-s", type=float, default=2.0)
    a("--local-mirror-root", default=str(Path.home() / "projects/chemgraph-academy/remote-runs"))
    a("--local-run-dir")
    a("--dashboard-host", default="127.0.0.1")
    a("--dashboard-port", type=int, default=8765)
    a("--local", action="store_true", help="Only serve an already mirrored local run.")
    a("--no-dashboard", action="store_true")
    a("--overwrite-run", action="store_true")
    args = p.parse_args()
    # Per-site override flags only make sense in single-site mode --
    # in multi-site they'd silently apply to all sites and almost
    # always be wrong (e.g. one Aurora remote_host doesn't fit Crux).
    # Force operators to encode site-specific quirks in the profile JSON.
    if len(args.system) > 1:
        forbidden = [
            (name, getattr(args, attr))
            for name, attr in (
                ("--remote-host", "remote_host"),
                ("--ssh-control-path", "ssh_control_path"),
                ("--relay-port", "relay_port"),
                ("--lm-base-url", "lm_base_url"),
                ("--local-run-dir", "local_run_dir"),
            )
            if getattr(args, attr)
        ]
        if forbidden:
            names = ", ".join(flag for flag, _ in forbidden)
            p.error(
                f"multi-site --system rejects single-site overrides {names}; "
                f"encode per-site differences in the system profile JSON "
                f"instead.",
            )
    return args

def template(name: str) -> str:
    return files("chemgraph.academy.runtime.templates").joinpath(name).read_text()


REMOTE_RELAY_SUBPATH = ".chemgraph/uan_http_relay.py"


def stage_relay_script(profile: SystemProfile, host: str, control_path: str) -> str:
    """Copy the bundled UAN relay script to the remote host.

    The relay script is shipped inside the chemgraph package so we no longer
    require a separate ``academy`` source checkout on the remote system.
    We materialize it under ``$REMOTE_ROOT/.chemgraph/uan_http_relay.py``
    on every dashboard launch (idempotent overwrite), then return that
    absolute path for the start_relay shell template to reference.
    """
    relay_dir = f"{profile.remote_root}/.chemgraph"
    relay_path = f"{relay_dir}/uan_http_relay.py"
    contents = template("uan_http_relay.py")
    cmd = (
        f"mkdir -p {shlex.quote(relay_dir)} && "
        f"cat > {shlex.quote(relay_path)}"
    )
    ssh(host, cmd, control_path=control_path, input_text=contents)
    return relay_path

def ssh(host: str, command: str | list[str] | None, *, control_path: str, input_text: str | None = None, check: bool = True, capture: bool = False, batch_mode: bool = True, extra: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = ["ssh"]
    if batch_mode:
        cmd += ["-o", "BatchMode=yes"]
    cmd += ["-o", f"ControlPath={control_path}", "-o", "ControlMaster=auto", "-o", "ControlPersist=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4"]
    cmd += extra or []
    cmd.append(host)
    cmd += command if isinstance(command, list) else ([command] if command else [])
    return subprocess.run(cmd, input=input_text, text=True, check=check, stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None)

def wrapper(profile: SystemProfile) -> str:
    return (
        template("compute_wrapper.sh.tmpl")
        .replace("%{path_prefix}%", ":".join([profile.redis_bin_dir, f"{profile.remote_root}/bin"]))
        .replace("%{pythonpath}%", ":".join(profile.pythonpath_entries))
        .replace("%{venv_python}%", profile.venv_python)
    )

def start_relay(profile: SystemProfile, host: str, control_path: str, args: argparse.Namespace, relay_port: int, relay_python: str, log_path: Path, relay_script: str) -> subprocess.Popen[str]:
    relay_args = ["bash", "-s", "--", profile.remote_root, relay_script, profile.relay_host_file, f"{profile.remote_root}/uan-relay-{relay_port}.pid", f"{profile.remote_root}/uan-relay-{relay_port}.log", str(relay_port), str(args.reverse_port), relay_python]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ControlPath={control_path}", "-o", "ControlMaster=auto", "-o", "ControlPersist=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4", "-R", f"127.0.0.1:{args.reverse_port}:{args.local_argo_host}:{args.local_argo_port}", host, *relay_args]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log_path.open("w", encoding="utf-8"), stderr=subprocess.STDOUT, text=True)
    assert process.stdin is not None
    process.stdin.write(template("start_relay.sh"))
    process.stdin.close()
    return process

def wait_relay(profile: SystemProfile, host: str, control_path: str, relay_port: int, process: subprocess.Popen[str], log_path: Path) -> str:
    print("Waiting for relay readiness...", flush=True)
    check = f"host=$(cat {shlex.quote(profile.relay_host_file)} 2>/dev/null || true); test -n \"$host\" && curl -fsS \"http://${{host}}:{relay_port}/v1/models\" >/dev/null"
    for _ in range(60):
        if ssh(host, check, control_path=control_path, check=False).returncode == 0:
            relay_host = ssh(host, ["cat", profile.relay_host_file], control_path=control_path, capture=True).stdout.strip()
            print(f"{profile.name} relay host: {relay_host}", flush=True)
            return relay_host
        if process.poll() is not None:
            raise RuntimeError("Relay SSH session exited before readiness. Local relay log:\n" + log_path.read_text(encoding="utf-8", errors="replace"))
        time.sleep(1)
    raise RuntimeError("Relay readiness timed out. Local relay log:\n" + log_path.read_text(encoding="utf-8", errors="replace"))

def start_rsync(host: str, control_path: str, remote_run_dir: str, local_run_dir: Path, interval_s: float, stop: threading.Event) -> None:
    local_run_dir.mkdir(parents=True, exist_ok=True)
    rsync_args = [host, control_path, remote_run_dir, str(local_run_dir), str(interval_s), str(local_run_dir / "rsync.log")]

    def loop() -> None:
        process = subprocess.Popen(["bash", "-s", "--", *rsync_args], stdin=subprocess.PIPE, text=True, start_new_session=True)
        assert process.stdin is not None
        process.stdin.write(template("rsync_loop.sh"))
        process.stdin.close()
        stop.wait()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)

    threading.Thread(target=loop, name="chemgraph-academy-rsync", daemon=True).start()

def compute_lines(profile: SystemProfile, wrapper_path: str, run_id: str, campaign: str) -> list[str]:
    lines = ["  module use /soft/modulefiles", "  module load conda", "  conda activate base"] if profile.name == "polaris" else ["  module load frameworks"]
    return lines + [f"  source {profile.remote_root}/venvs/academy-swarm/bin/activate", f"  export PATH={profile.remote_root}/bin:$PATH", "  chemgraph academy run-compute \\", f"    --system {profile.name} \\", f"    --run-id {run_id} \\", f"    --campaign {campaign}", "", "If PATH is not configured, use:", f"  {wrapper_path} \\", f"    --system {profile.name} \\", f"    --run-id {run_id} \\", f"    --campaign {campaign}"]

def _resolve_local_run_root(args: argparse.Namespace) -> Path:
    """Top-level dashboard dir on the Mac.

    Single-site mode: ``<root>/<run_id>/`` -- byte-identical to the
    pre-multi-site layout, so existing dashboard URLs / mirror paths
    keep working unchanged.

    Multi-site mode: ``<root>/<run_id>/`` is a PARENT containing
    per-site subdirs (``<root>/<run_id>/aurora/``, ``.../crux/``).
    The dashboard server walks that tree and merges per-site event
    streams into one view.
    """
    if args.local_run_dir:
        return Path(args.local_run_dir).expanduser()
    return (Path(args.local_mirror_root) / args.run_id).expanduser()


def _site_mirror_dir(
    local_run_root: Path,
    profile_name: str,
    *,
    multi_site: bool,
) -> Path:
    return local_run_root / profile_name if multi_site else local_run_root


def _setup_site(
    *,
    profile_name: str,
    args: argparse.Namespace,
    local_run_root: Path,
    multi_site: bool,
    site_index: int,
    stop: threading.Event,
) -> _SiteHandle:
    """Bring up one site's ControlMaster + UAN relay + rsync mirror.

    Pulled out of ``main`` so the multi-site loop has one place to call.
    The single-site path also goes through this function (with
    ``multi_site=False`` so the mirror dir + reverse-port stay
    backward-compatible). Returns a ``_SiteHandle`` carrying everything
    the cleanup ``finally`` needs.
    """
    profile = load_system_profile(profile_name)
    remote_host = args.remote_host or profile.remote_host
    control_path = (
        args.ssh_control_path
        or str(Path.home() / f".ssh/{profile.name}-dashboard-%r@%h:%p")
    )
    relay_port = args.relay_port or profile.relay_port
    remote_run_dir = f"{profile.run_root}/{args.run_id}"
    local_mirror_dir = _site_mirror_dir(local_run_root, profile.name, multi_site=multi_site)
    site = _SiteHandle(
        profile=profile,
        remote_host=remote_host,
        control_path=control_path,
        local_mirror_dir=local_mirror_dir,
        relay_port=relay_port,
    )

    Path(control_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    if ssh(remote_host, None, control_path=control_path, extra=["-O", "check"], check=False, batch_mode=False).returncode != 0:
        print(f"[{profile.name}] Starting SSH ControlMaster for {remote_host}...", flush=True)
        ssh(remote_host, None, control_path=control_path, extra=["-M", "-N", "-f", "-o", "ControlMaster=yes"], batch_mode=False)
        site.started_master = True

    if args.overwrite_run:
        if not args.run_id or "/" in args.run_id or args.run_id in {".", ".."}:
            raise RuntimeError(f"Refusing to overwrite unsafe run id: {args.run_id!r}")
        print(f"[{profile.name}] Deleting existing run artifacts (--overwrite-run):", flush=True)
        print(f"  remote: {remote_host}:{remote_run_dir}", flush=True)
        print(f"  local:  {local_mirror_dir}", flush=True)
        delete = f"set -euo pipefail; run_root={shlex.quote(profile.run_root)}; run_id={shlex.quote(args.run_id)}; case \"$run_id\" in \"\"|.|..|*/*) echo \"unsafe run id\" >&2; exit 2;; esac; run_dir=\"$run_root/$run_id\"; trash_root=\"$run_root/.deleted-runs\"; if [ -e \"$run_dir\" ]; then mkdir -p \"$trash_root\"; trash_dir=\"$trash_root/${{run_id}}.$(date +%Y%m%d%H%M%S).$$\"; mv -- \"$run_dir\" \"$trash_dir\"; for delay in 0 1 2 5 10; do sleep \"$delay\"; if rm -rf -- \"$trash_dir\" 2>/dev/null; then break; fi; done; fi; mkdir -p \"$run_dir\""
        ssh(remote_host, delete, control_path=control_path)
        if local_mirror_dir.exists():
            shutil.rmtree(local_mirror_dir)

    wrapper_path = f"{profile.remote_root}/bin/chemgraph-academy-run"
    print(f"[{profile.name}] Installing compute wrapper at {wrapper_path}...", flush=True)
    ssh(remote_host, f"mkdir -p {shlex.quote(profile.remote_root + '/bin')} && cat > {shlex.quote(wrapper_path)} && chmod +x {shlex.quote(wrapper_path)}", control_path=control_path, input_text=wrapper(profile))

    relay_host = None
    if args.lm_connect == "mac-argo-relay":
        # Each site gets its own reverse port (base + site_index) so two
        # SSH -R tunnels don't fight over the same local port. The remote
        # relay always listens on the profile's relay_port; only the SSH
        # tunneling end on the Mac shifts.
        per_site_args = argparse.Namespace(**vars(args))
        per_site_args.reverse_port = args.reverse_port + site_index
        print(f"[{profile.name}] Staging UAN relay script...", flush=True)
        relay_script = stage_relay_script(profile, remote_host, control_path)
        print(f"[{profile.name}] Starting UAN relay through {remote_host} (reverse port {per_site_args.reverse_port})...", flush=True)
        relay_log = Path(f"/tmp/chemgraph-academy-{args.run_id}-{profile.name}-relay.log")
        site.relay_process = start_relay(
            profile, remote_host, control_path, per_site_args,
            relay_port, args.relay_python or profile.venv_python,
            relay_log, relay_script,
        )
        relay_host = wait_relay(profile, remote_host, control_path, relay_port, site.relay_process, relay_log)
        site.relay_host = relay_host

    lm_base_url = (
        f"http://{relay_host}:{relay_port}/argoapi/v1"
        if relay_host else str(args.lm_base_url)
    )
    site.lm_base_url = lm_base_url
    print(f"[{profile.name}] Compute-node LM URL: {lm_base_url}", flush=True)

    metadata: dict[str, Any] = {
        "created_at": time.time(),
        "created_by": "chemgraph academy dashboard",
        "run_id": args.run_id,
        "system": profile.name,
        "campaign": args.campaign,
        "remote_run_dir": remote_run_dir,
        "remote_host": remote_host,
        "lm_connect": args.lm_connect,
        "lm_base_url": lm_base_url,
        "workspace_root": profile.remote_root,
        "chemgraph_repo_root": profile.repo_root,
    }
    if relay_host:
        metadata.update({"relay_host": relay_host, "relay_port": relay_port})
    print(f"[{profile.name}] Writing run metadata: {remote_host}:{remote_run_dir}/dashboard_metadata.json", flush=True)
    ssh(remote_host, f"mkdir -p {shlex.quote(remote_run_dir)} && cat > {shlex.quote(remote_run_dir + '/dashboard_metadata.json')}", control_path=control_path, input_text=json.dumps(metadata, indent=2) + "\n")

    print(f"[{profile.name}] Starting rsync mirror:", flush=True)
    print(f"  {remote_host}:{remote_run_dir}/", flush=True)
    print(f"  {local_mirror_dir}/", flush=True)
    start_rsync(remote_host, control_path, remote_run_dir, local_mirror_dir, args.rsync_interval_s, stop)

    print(f"\n[{profile.name}] Compute-node command:", flush=True)
    print("\n".join(compute_lines(profile, wrapper_path, args.run_id, args.campaign)), flush=True)

    return site


def _teardown_site(site: _SiteHandle, *, keep_ssh_master: bool) -> None:
    if site.relay_process is not None and site.relay_process.poll() is None:
        site.relay_process.terminate()
        try:
            site.relay_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            site.relay_process.kill()
    if site.started_master and not keep_ssh_master:
        ssh(
            site.remote_host, None,
            control_path=site.control_path,
            extra=["-O", "exit"], check=False, batch_mode=False,
        )


# Note about local-argo reachability: we only check the local argo-shim
# once at the top of main(), even in multi-site mode -- all sites share
# the same Mac shim, so one check covers them all.


def main() -> int:
    args = parse_args()
    # Tolerate args.system being a plain string (legacy single-site
    # callers / older tests) as well as the tuple form produced by the
    # new --system parser. Without this, "aurora" would iterate
    # character-by-character.
    systems: tuple[str, ...] = (
        (args.system,) if isinstance(args.system, str) else tuple(args.system)
    )
    multi_site = len(systems) > 1
    campaign_launch_defaults(args.campaign)
    local_run_root = _resolve_local_run_root(args)
    local_run_root.mkdir(parents=True, exist_ok=True)

    if args.local:
        if args.overwrite_run:
            raise RuntimeError("--overwrite-run cannot be used with --local")
        # Dashboard server walks the tree either way -- single-site
        # mirror dir or multi-site parent both work as inputs.
        return 0 if args.no_dashboard else serve_dashboard(
            run_dir=local_run_root,
            host=args.dashboard_host, port=args.dashboard_port,
        )

    if args.lm_connect == "direct" and not args.lm_base_url:
        raise RuntimeError("--lm-connect direct requires --lm-base-url")
    if args.lm_connect == "mac-argo-relay":
        try:
            with urllib.request.urlopen(f"http://{args.local_argo_host}:{args.local_argo_port}/v1/models", timeout=5) as response:
                if int(response.status) >= 300:
                    raise OSError
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise RuntimeError("Local argo-shim is not reachable. Start it before using --lm-connect mac-argo-relay.") from exc

    stop = threading.Event()
    sites: list[_SiteHandle] = []
    try:
        for index, profile_name in enumerate(systems):
            site = _setup_site(
                profile_name=profile_name,
                args=args,
                local_run_root=local_run_root,
                multi_site=multi_site,
                site_index=index,
                stop=stop,
            )
            sites.append(site)

        print("\nDashboard launcher is ready.", flush=True)
        if multi_site:
            print(f"Federated mirror tree: {local_run_root}/<system>/", flush=True)

        if args.no_dashboard:
            return 0

        print(f"\nStarting dashboard at http://{args.dashboard_host}:{args.dashboard_port}", flush=True)
        print("Ctrl-C stops the local dashboard, rsync loops, and relay tunnels.", flush=True)
        return serve_dashboard(
            run_dir=local_run_root,
            host=args.dashboard_host, port=args.dashboard_port,
        )
    finally:
        stop.set()
        for site in sites:
            _teardown_site(site, keep_ssh_master=args.keep_ssh_master)


if __name__ == "__main__":
    raise SystemExit(main())
