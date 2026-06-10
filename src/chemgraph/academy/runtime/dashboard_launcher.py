from __future__ import annotations

import argparse
import json
import os, shlex, shutil, signal, subprocess, threading
import time
import urllib.error
import urllib.request
from importlib.resources import files
from pathlib import Path

from chemgraph.academy.dashboard import serve_dashboard
from chemgraph.academy.examples import campaign_launch_defaults
from chemgraph.academy.runtime.profiles import list_builtin_system_profiles
from chemgraph.academy.runtime.profiles import load_system_profile
from chemgraph.academy.runtime.profiles.system import SystemProfile

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="chemgraph academy dashboard")
    a = p.add_argument
    a("run_id")
    a("--system", default="aurora", help="Built-ins: " + ", ".join(list_builtin_system_profiles()))
    a("--campaign", default="mace-ensemble-screening-20")
    a("--lm-connect", choices=("mac-argo-relay", "direct"), default="mac-argo-relay")
    a("--lm-base-url")
    a("--remote-host")
    a("--ssh-control-path")
    a("--keep-ssh-master", action="store_true")
    a("--local-argo-host", default="127.0.0.1")
    a("--local-argo-port", type=int, default=18085)
    a("--reverse-port", type=int, default=18185)
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
    return p.parse_args()

def template(name: str) -> str:
    return files("chemgraph.academy.runtime.templates").joinpath(name).read_text()

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

def start_relay(profile: SystemProfile, host: str, control_path: str, args: argparse.Namespace, relay_port: int, relay_python: str, log_path: Path) -> subprocess.Popen[str]:
    relay_script = f"{profile.academy_repo_root}/examples/09-polaris-lm-swarm/uan_http_relay.py"
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
    return lines + [f"  source {profile.remote_root}/venvs/academy-swarm/bin/activate", f"  export PATH={profile.remote_root}/bin:$PATH", "  chemgraph-academy-run \\", f"    --system {profile.name} \\", f"    --run-id {run_id} \\", f"    --campaign {campaign}", "", "If PATH is not configured, use:", f"  {wrapper_path} \\", f"    --system {profile.name} \\", f"    --run-id {run_id} \\", f"    --campaign {campaign}"]

def main() -> int:
    args = parse_args()
    profile = load_system_profile(args.system)
    campaign_launch_defaults(args.campaign)
    local_run_dir = Path(args.local_run_dir or Path(args.local_mirror_root) / args.run_id).expanduser()
    local_run_dir.mkdir(parents=True, exist_ok=True)
    if args.local:
        if args.overwrite_run:
            raise RuntimeError("--overwrite-run cannot be used with --local")
        return 0 if args.no_dashboard else serve_dashboard(run_dir=local_run_dir, host=args.dashboard_host, port=args.dashboard_port)
    if args.lm_connect == "direct" and not args.lm_base_url:
        raise RuntimeError("--lm-connect direct requires --lm-base-url")
    if args.lm_connect == "mac-argo-relay":
        try:
            with urllib.request.urlopen(f"http://{args.local_argo_host}:{args.local_argo_port}/v1/models", timeout=5) as response:
                if int(response.status) >= 300:
                    raise OSError
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            raise RuntimeError("Local argo-shim is not reachable. Start it before using --lm-connect mac-argo-relay.") from exc

    remote_host = args.remote_host or profile.remote_host
    control_path = args.ssh_control_path or str(Path.home() / f".ssh/{profile.name}-dashboard-%r@%h:%p")
    relay_port = args.relay_port or profile.relay_port
    remote_run_dir = f"{profile.run_root}/{args.run_id}"
    relay_process: subprocess.Popen[str] | None = None
    stop = threading.Event()
    started_master = False
    try:
        Path(control_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        if ssh(remote_host, None, control_path=control_path, extra=["-O", "check"], check=False, batch_mode=False).returncode != 0:
            print(f"Starting SSH ControlMaster for {remote_host}...", flush=True)
            ssh(remote_host, None, control_path=control_path, extra=["-M", "-N", "-f", "-o", "ControlMaster=yes"], batch_mode=False)
            started_master = True
        if args.overwrite_run:
            if not args.run_id or "/" in args.run_id or args.run_id in {".", ".."}:
                raise RuntimeError(f"Refusing to overwrite unsafe run id: {args.run_id!r}")
            print("Deleting existing run artifacts because --overwrite-run was set:", flush=True)
            print(f"  remote: {remote_host}:{remote_run_dir}", flush=True)
            print(f"  local:  {local_run_dir}", flush=True)
            delete = f"set -euo pipefail; run_root={shlex.quote(profile.run_root)}; run_id={shlex.quote(args.run_id)}; case \"$run_id\" in \"\"|.|..|*/*) echo \"unsafe run id\" >&2; exit 2;; esac; run_dir=\"$run_root/$run_id\"; trash_root=\"$run_root/.deleted-runs\"; if [ -e \"$run_dir\" ]; then mkdir -p \"$trash_root\"; trash_dir=\"$trash_root/${{run_id}}.$(date +%Y%m%d%H%M%S).$$\"; mv -- \"$run_dir\" \"$trash_dir\"; for delay in 0 1 2 5 10; do sleep \"$delay\"; if rm -rf -- \"$trash_dir\" 2>/dev/null; then break; fi; done; fi; mkdir -p \"$run_dir\""
            ssh(remote_host, delete, control_path=control_path)
            if local_run_dir.exists():
                shutil.rmtree(local_run_dir)
        wrapper_path = f"{profile.remote_root}/bin/chemgraph-academy-run"
        print(f"Installing compute wrapper at {wrapper_path}...", flush=True)
        ssh(remote_host, f"mkdir -p {shlex.quote(profile.remote_root + '/bin')} && cat > {shlex.quote(wrapper_path)} && chmod +x {shlex.quote(wrapper_path)}", control_path=control_path, input_text=wrapper(profile))
        relay_host = None
        if args.lm_connect == "mac-argo-relay":
            print(f"Starting {profile.name} UAN relay through {remote_host}...", flush=True)
            relay_process = start_relay(profile, remote_host, control_path, args, relay_port, args.relay_python or profile.venv_python, Path(f"/tmp/chemgraph-academy-{args.run_id}-relay.log"))
            relay_host = wait_relay(profile, remote_host, control_path, relay_port, relay_process, Path(f"/tmp/chemgraph-academy-{args.run_id}-relay.log"))
        lm_base_url = f"http://{relay_host}:{relay_port}/argoapi/v1" if relay_host else str(args.lm_base_url)
        print(f"Compute-node LM URL: {lm_base_url}", flush=True)
        metadata = {"created_at": time.time(), "created_by": "chemgraph-academy-dashboard", "run_id": args.run_id, "system": profile.name, "campaign": args.campaign, "remote_run_dir": remote_run_dir, "remote_host": remote_host, "lm_connect": args.lm_connect, "lm_base_url": lm_base_url, "workspace_root": profile.remote_root, "academy_repo_root": profile.academy_repo_root, "chemgraph_repo_root": profile.repo_root}
        if relay_host:
            metadata.update({"relay_host": relay_host, "relay_port": relay_port})
        print(f"Writing run metadata: {remote_host}:{remote_run_dir}/dashboard_metadata.json", flush=True)
        ssh(remote_host, f"mkdir -p {shlex.quote(remote_run_dir)} && cat > {shlex.quote(remote_run_dir + '/dashboard_metadata.json')}", control_path=control_path, input_text=json.dumps(metadata, indent=2) + "\n")
        print("Starting rsync mirror:", flush=True)
        print(f"  {remote_host}:{remote_run_dir}/", flush=True)
        print(f"  {local_run_dir}/", flush=True)
        start_rsync(remote_host, control_path, remote_run_dir, local_run_dir, args.rsync_interval_s, stop)
        print("\nDashboard launcher is ready.\n", flush=True)
        print(f"On the {profile.name} compute node, use:", flush=True)
        print("\n".join(compute_lines(profile, wrapper_path, args.run_id, args.campaign)), flush=True)
        if args.no_dashboard:
            return 0
        print(f"\nStarting dashboard at http://{args.dashboard_host}:{args.dashboard_port}", flush=True)
        print("Ctrl-C stops the local dashboard, rsync loop, and relay tunnel.", flush=True)
        return serve_dashboard(run_dir=local_run_dir, host=args.dashboard_host, port=args.dashboard_port)
    finally:
        stop.set()
        if relay_process is not None and relay_process.poll() is None:
            relay_process.terminate()
            try:
                relay_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                relay_process.kill()
        if started_master and not args.keep_ssh_master:
            ssh(remote_host, None, control_path=control_path, extra=["-O", "exit"], check=False, batch_mode=False)

if __name__ == "__main__":
    raise SystemExit(main())
