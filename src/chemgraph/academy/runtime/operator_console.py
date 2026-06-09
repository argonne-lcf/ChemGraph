from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from chemgraph.academy.examples import campaign_launch_defaults
from chemgraph.academy.runtime.profiles import list_builtin_system_profiles
from chemgraph.academy.runtime.profiles import load_system_profile
from chemgraph.academy.runtime.profiles.system import SystemProfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start the local operator console for a ChemGraph Academy run. "
            "This prepares remote run metadata, starts the local dashboard, "
            "and optionally starts the temporary Mac-to-UAN Argo relay."
        ),
    )
    parser.add_argument("run_id")
    parser.add_argument(
        "--system",
        default="aurora",
        help=(
            "Built-in system profile or profile JSON path. Built-ins: "
            + ", ".join(list_builtin_system_profiles())
        ),
    )
    parser.add_argument("--campaign", default="mace-ensemble-screening-20")
    parser.add_argument(
        "--lm-connect",
        choices=("mac-argo-relay", "direct"),
        default="mac-argo-relay",
        help=(
            "How the compute job should reach the LM endpoint. "
            "mac-argo-relay starts the current SSH reverse tunnel and UAN "
            "relay. direct writes --lm-base-url to run metadata without "
            "starting relay infrastructure."
        ),
    )
    parser.add_argument(
        "--lm-base-url",
        help="Required for --lm-connect direct. Overrides generated relay URL.",
    )
    parser.add_argument("--operator-host", help="SSH target for the login/UAN host.")
    parser.add_argument("--ssh-control-path")
    parser.add_argument("--keep-ssh-master", action="store_true")
    parser.add_argument("--local-argo-host", default="127.0.0.1")
    parser.add_argument("--local-argo-port", type=int, default=18085)
    parser.add_argument("--reverse-port", type=int, default=18185)
    parser.add_argument("--relay-port", type=int)
    parser.add_argument("--relay-python")
    parser.add_argument("--rsync-interval-s", type=float, default=2.0)
    parser.add_argument(
        "--local-mirror-root",
        default=str(Path.home() / "projects/chemgraph-academy/remote-runs"),
    )
    parser.add_argument("--local-run-dir")
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Only serve an already mirrored local run. No SSH, relay, or rsync.",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Prepare operator metadata and return without serving dashboard.",
    )
    parser.add_argument(
        "--overwrite-run",
        action="store_true",
        help=(
            "Delete the remote run directory and local mirror before starting. "
            "This does not stop an already-running compute job."
        ),
    )
    return parser.parse_args()


def _log(message: str) -> None:
    print(message, flush=True)


def _http_ok(url: str, *, timeout_s: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError, urllib.error.HTTPError):
        return False


def _run(command: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        input=input_text,
        text=True,
        check=True,
    )


def _ssh_options(control_path: str, *, batch_mode: bool = True) -> list[str]:
    opts = [
        "-o",
        f"ControlPath={control_path}",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=4",
    ]
    if batch_mode:
        opts[:0] = ["-o", "BatchMode=yes"]
    return opts


def _start_ssh_master(*, host: str, control_path: str) -> bool:
    Path(control_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    check = subprocess.run(
        ["ssh", "-o", f"ControlPath={control_path}", "-O", "check", host],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if check.returncode == 0:
        return False

    _log(f"Starting SSH ControlMaster for {host}...")
    _run(
        [
            "ssh",
            "-M",
            "-N",
            "-f",
            "-o",
            "ControlMaster=yes",
            "-o",
            f"ControlPath={control_path}",
            "-o",
            "ControlPersist=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=4",
            host,
        ],
    )
    return True


def _stop_ssh_master(*, host: str, control_path: str) -> None:
    subprocess.run(
        ["ssh", "-o", f"ControlPath={control_path}", "-O", "exit", host],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )


def _wrapper_text(profile: SystemProfile) -> str:
    path_prefix = ":".join([profile.redis_bin_dir, f"{profile.remote_root}/bin"])
    pythonpath = ":".join(profile.pythonpath_entries)
    return f"""#!/bin/bash
set -euo pipefail

log() {{
  printf '[chemgraph-academy-run] %s\\n' "$*" >&2
}}

export PATH="{path_prefix}:${{PATH}}"
export PYTHONPATH="{pythonpath}:${{PYTHONPATH:-}}"

PYTHON_BIN="${{CHEMGRAPH_ACADEMY_PYTHON:-python}}"
if ! command -v "${{PYTHON_BIN}}" >/dev/null 2>&1; then
  log "Python command not found: ${{PYTHON_BIN}}"
  log "Load your site module and activate the ChemGraph/Academy environment first."
  log "Profile Python, if you want to use it explicitly: {profile.venv_python}"
  exit 1
fi

ACTIVE_PYTHON="$("${{PYTHON_BIN}}" -c 'import sys; print(sys.executable)')"
log "using active Python: ${{ACTIVE_PYTHON}}"
log "not loading modules or activating a venv inside this wrapper"

if ! "${{PYTHON_BIN}}" -c 'import chemgraph.academy.runtime.compute_launcher' >/dev/null 2>&1; then
  log "active Python cannot import chemgraph.academy.runtime.compute_launcher"
  log "Load the proper site module and venv before running this command."
  log "Profile Python, if you want to use it explicitly: {profile.venv_python}"
  exit 1
fi

log "starting ChemGraph Academy compute launcher"
exec "${{PYTHON_BIN}}" -m chemgraph.academy.runtime.compute_launcher "$@"
"""


def _install_compute_wrapper(
    *,
    profile: SystemProfile,
    host: str,
    ssh_opts: list[str],
) -> str:
    wrapper_bin_dir = f"{profile.remote_root}/bin"
    wrapper_path = f"{wrapper_bin_dir}/chemgraph-academy-run"
    _log(f"Installing compute wrapper at {wrapper_path}...")
    remote_command = (
        f"mkdir -p {shlex.quote(wrapper_bin_dir)} && "
        f"cat > {shlex.quote(wrapper_path)} && "
        f"chmod +x {shlex.quote(wrapper_path)}"
    )
    _run(
        ["ssh", *ssh_opts, host, remote_command],
        input_text=_wrapper_text(profile),
    )
    return wrapper_path


def _relay_script_text() -> str:
    return r"""
set -euo pipefail

REMOTE_ROOT="$1"
RELAY_SCRIPT="$2"
RELAY_HOST_FILE="$3"
RELAY_PID_FILE="$4"
RELAY_LOG_FILE="$5"
RELAY_PORT="$6"
REVERSE_PORT="$7"
RELAY_PYTHON="$8"

cd "${REMOTE_ROOT}"
UAN_HOST="$(hostname -f)"
printf '%s\n' "${UAN_HOST}" > "${RELAY_HOST_FILE}"

if [ -f "${RELAY_PID_FILE}" ]; then
  OLD_PID="$(cat "${RELAY_PID_FILE}" 2>/dev/null || true)"
  case "${OLD_PID}" in
    ''|*[!0-9]*) ;;
    *) kill "${OLD_PID}" 2>/dev/null || true ;;
  esac
fi

"${RELAY_PYTHON}" "${RELAY_SCRIPT}" \
  --listen-host 0.0.0.0 \
  --listen-port "${RELAY_PORT}" \
  --target-host 127.0.0.1 \
  --target-port "${REVERSE_PORT}" \
  > "${RELAY_LOG_FILE}" 2>&1 &
RELAY_PID="$!"
printf '%s\n' "${RELAY_PID}" > "${RELAY_PID_FILE}"

cleanup_remote() {
  kill "${RELAY_PID}" 2>/dev/null || true
}
trap cleanup_remote EXIT

deadline=$((SECONDS + 45))
while ! curl -fsS "http://${UAN_HOST}:${RELAY_PORT}/v1/models" >/dev/null; do
  if ! kill -0 "${RELAY_PID}" 2>/dev/null; then
    echo "UAN relay process exited before readiness. Last relay log lines:" >&2
    tail -n 80 "${RELAY_LOG_FILE}" >&2 || true
    exit 1
  fi
  if [ "${SECONDS}" -gt "${deadline}" ]; then
    echo "UAN relay did not become ready. Last relay log lines:" >&2
    tail -n 80 "${RELAY_LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "UAN_RELAY_HOST=${UAN_HOST}"
echo "UAN relay ready at http://${UAN_HOST}:${RELAY_PORT}/argoapi/v1"

while true; do
  sleep 3600
done
"""


def _start_mac_argo_relay(
    *,
    profile: SystemProfile,
    host: str,
    ssh_opts: list[str],
    local_argo_host: str,
    local_argo_port: int,
    reverse_port: int,
    relay_port: int,
    relay_python: str,
    local_log_path: Path,
) -> subprocess.Popen[str]:
    relay_script = f"{profile.academy_repo_root}/examples/09-polaris-lm-swarm/uan_http_relay.py"
    relay_pid_file = f"{profile.remote_root}/uan-relay-{relay_port}.pid"
    relay_log_file = f"{profile.remote_root}/uan-relay-{relay_port}.log"
    local_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = local_log_path.open("w", encoding="utf-8")

    _log(f"Starting {profile.name} UAN relay through {host}...")
    command = [
        "ssh",
        *ssh_opts,
        "-R",
        f"127.0.0.1:{reverse_port}:{local_argo_host}:{local_argo_port}",
        host,
        "bash",
        "-s",
        "--",
        profile.remote_root,
        relay_script,
        profile.relay_host_file,
        relay_pid_file,
        relay_log_file,
        str(relay_port),
        str(reverse_port),
        relay_python,
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdin is not None
    process.stdin.write(_relay_script_text())
    process.stdin.close()
    return process


def _remote_relay_ready(
    *,
    host: str,
    ssh_opts: list[str],
    relay_host_file: str,
    relay_port: int,
) -> bool:
    command = (
        f"host=$(cat {shlex.quote(relay_host_file)} 2>/dev/null || true); "
        f'test -n "$host" && '
        f'curl -fsS "http://${{host}}:{relay_port}/v1/models" >/dev/null'
    )
    result = subprocess.run(
        ["ssh", *ssh_opts, host, command],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _read_remote_file(
    *,
    host: str,
    ssh_opts: list[str],
    path: str,
) -> str:
    result = subprocess.run(
        ["ssh", *ssh_opts, host, "cat", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _wait_for_relay(
    *,
    profile: SystemProfile,
    host: str,
    ssh_opts: list[str],
    relay_port: int,
    relay_process: subprocess.Popen[str],
    local_log_path: Path,
) -> str:
    _log("Waiting for relay readiness...")
    deadline = time.time() + 60
    while time.time() < deadline:
        if _remote_relay_ready(
            host=host,
            ssh_opts=ssh_opts,
            relay_host_file=profile.relay_host_file,
            relay_port=relay_port,
        ):
            relay_host = _read_remote_file(
                host=host,
                ssh_opts=ssh_opts,
                path=profile.relay_host_file,
            )
            _log(f"{profile.name} relay host: {relay_host}")
            return relay_host
        if relay_process.poll() is not None:
            detail = local_log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                "Relay SSH session exited before readiness. Local relay log:\n"
                + detail,
            )
        time.sleep(1)
    detail = local_log_path.read_text(encoding="utf-8", errors="replace")
    raise RuntimeError("Relay readiness timed out. Local relay log:\n" + detail)


def _write_operator_metadata(
    *,
    profile: SystemProfile,
    host: str,
    ssh_opts: list[str],
    run_id: str,
    campaign: str,
    lm_connect: str,
    lm_base_url: str,
    relay_host: str | None,
    relay_port: int | None,
) -> None:
    remote_run_dir = f"{profile.run_root}/{run_id}"
    payload: dict[str, Any] = {
        "created_at": time.time(),
        "created_by": "chemgraph-academy-console",
        "run_id": run_id,
        "system": profile.name,
        "campaign": campaign,
        "remote_run_dir": remote_run_dir,
        "operator_host": host,
        "lm_connect": lm_connect,
        "lm_base_url": lm_base_url,
        "workspace_root": profile.remote_root,
        "academy_repo_root": profile.academy_repo_root,
        "chemgraph_repo_root": profile.repo_root,
    }
    if relay_host:
        payload["relay_host"] = relay_host
    if relay_port is not None:
        payload["relay_port"] = relay_port

    metadata = json.dumps(payload, indent=2) + "\n"
    remote_path = f"{remote_run_dir}/operator_metadata.json"
    remote_command = (
        f"mkdir -p {shlex.quote(remote_run_dir)} && "
        f"cat > {shlex.quote(remote_path)}"
    )
    _log(f"Writing run metadata: {host}:{remote_run_dir}/operator_metadata.json")
    _run(
        ["ssh", *ssh_opts, host, remote_command],
        input_text=metadata,
    )


def _run_id_allows_delete(run_id: str) -> bool:
    return bool(run_id) and "/" not in run_id and run_id not in {".", ".."}


def _delete_existing_run(
    *,
    profile: SystemProfile,
    host: str,
    ssh_opts: list[str],
    run_id: str,
    local_run_dir: Path,
) -> None:
    if not _run_id_allows_delete(run_id):
        raise RuntimeError(f"Refusing to overwrite unsafe run id: {run_id!r}")

    remote_run_dir = f"{profile.run_root}/{run_id}"
    _log("Deleting existing run artifacts because --overwrite-run was set:")
    _log(f"  remote: {host}:{remote_run_dir}")
    _log(f"  local:  {local_run_dir}")

    remote_command = (
        "set -euo pipefail; "
        f"run_root={shlex.quote(profile.run_root)}; "
        f"run_id={shlex.quote(run_id)}; "
        'case "$run_id" in ""|.|..|*/*) echo "unsafe run id" >&2; exit 2;; esac; '
        'run_dir="$run_root/$run_id"; '
        'trash_root="$run_root/.deleted-runs"; '
        'if [ -e "$run_dir" ]; then '
        'mkdir -p "$trash_root"; '
        'trash_dir="$trash_root/${run_id}.$(date +%Y%m%d%H%M%S).$$"; '
        'mv -- "$run_dir" "$trash_dir"; '
        'for delay in 0 1 2 5 10; do '
        'sleep "$delay"; '
        'if rm -rf -- "$trash_dir" 2>/dev/null; then break; fi; '
        'done; '
        'fi; '
        'mkdir -p "$run_dir"'
    )
    _run(["ssh", *ssh_opts, host, remote_command])
    if local_run_dir.exists():
        shutil.rmtree(local_run_dir)


def _start_rsync_loop(
    *,
    host: str,
    control_path: str,
    remote_run_dir: str,
    local_run_dir: Path,
    interval_s: float,
    stop_event: threading.Event,
) -> threading.Thread:
    local_run_dir.mkdir(parents=True, exist_ok=True)
    log_path = local_run_dir / "rsync.log"

    def loop() -> None:
        ssh_command = (
            "ssh "
            "-o BatchMode=yes "
            "-o ControlMaster=auto "
            f"-o ControlPath={shlex.quote(control_path)} "
            "-o ControlPersist=yes"
        )
        while not stop_event.is_set():
            with log_path.open("a", encoding="utf-8") as log:
                subprocess.run(
                    [
                        "rsync",
                        "-az",
                        "--delete",
                        "-e",
                        ssh_command,
                        f"{host}:{remote_run_dir}/",
                        f"{local_run_dir}/",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            stop_event.wait(interval_s)

    thread = threading.Thread(target=loop, name="chemgraph-academy-rsync", daemon=True)
    thread.start()
    return thread


def _run_dashboard(*, local_run_dir: Path, host: str, port: int) -> int:
    from chemgraph.academy import dashboard

    old_argv = sys.argv
    try:
        sys.argv = [
            "chemgraph-academy-console dashboard",
            "--run-dir",
            str(local_run_dir),
            "--host",
            host,
            "--port",
            str(port),
        ]
        return dashboard.main()
    finally:
        sys.argv = old_argv


def _print_compute_command(
    *,
    profile: SystemProfile,
    wrapper_path: str,
    run_id: str,
    campaign: str,
) -> None:
    _log("")
    _log("Operator console is ready.")
    _log("")
    _log(f"On the {profile.name} compute node, use:")
    if profile.name == "polaris":
        _log("  module use /soft/modulefiles")
        _log("  module load conda")
        _log("  conda activate base")
        _log(f"  source {profile.remote_root}/venvs/academy-swarm/bin/activate")
    else:
        _log("  module load frameworks")
        _log(f"  source {profile.remote_root}/venvs/academy-swarm/bin/activate")
    _log(f"  export PATH={profile.remote_root}/bin:$PATH")
    _log("  chemgraph-academy-run \\")
    _log(f"    --system {profile.name} \\")
    _log(f"    --run-id {run_id} \\")
    _log(f"    --campaign {campaign}")
    _log("")
    _log("If PATH is not configured, use:")
    _log(f"  {wrapper_path} \\")
    _log(f"    --system {profile.name} \\")
    _log(f"    --run-id {run_id} \\")
    _log(f"    --campaign {campaign}")


def _validate_campaign_name(campaign: str) -> None:
    campaign_launch_defaults(campaign)


def main() -> int:
    args = parse_args()
    profile = load_system_profile(args.system)
    _validate_campaign_name(args.campaign)

    local_run_dir = Path(
        args.local_run_dir or Path(args.local_mirror_root) / args.run_id,
    ).expanduser()
    local_run_dir.mkdir(parents=True, exist_ok=True)

    if args.local and args.overwrite_run:
        raise RuntimeError("--overwrite-run cannot be used with --local")

    if args.local:
        if args.no_dashboard:
            _log(f"Local run directory: {local_run_dir}")
            return 0
        return _run_dashboard(
            local_run_dir=local_run_dir,
            host=args.dashboard_host,
            port=args.dashboard_port,
        )

    operator_host = args.operator_host or profile.operator_host
    control_path = (
        args.ssh_control_path
        or str(Path.home() / f".ssh/{profile.name}-dashboard-%r@%h:%p")
    )
    relay_port = args.relay_port or profile.relay_port
    relay_python = args.relay_python or profile.venv_python
    local_relay_log = Path(f"/tmp/chemgraph-academy-{args.run_id}-relay.log")
    remote_run_dir = f"{profile.run_root}/{args.run_id}"

    relay_process: subprocess.Popen[str] | None = None
    stop_rsync = threading.Event()
    started_ssh_master = False

    try:
        if args.lm_connect == "mac-argo-relay":
            health_url = f"http://{args.local_argo_host}:{args.local_argo_port}/v1/models"
            if not _http_ok(health_url):
                raise RuntimeError(
                    "Local argo-shim is not reachable: "
                    f"{health_url}\n"
                    "Start it before using --lm-connect mac-argo-relay.",
                )
        elif not args.lm_base_url:
            raise RuntimeError("--lm-connect direct requires --lm-base-url")

        started_ssh_master = _start_ssh_master(
            host=operator_host,
            control_path=control_path,
        )
        ssh_opts = _ssh_options(control_path)
        if args.overwrite_run:
            _delete_existing_run(
                profile=profile,
                host=operator_host,
                ssh_opts=ssh_opts,
                run_id=args.run_id,
                local_run_dir=local_run_dir,
            )
        wrapper_path = _install_compute_wrapper(
            profile=profile,
            host=operator_host,
            ssh_opts=ssh_opts,
        )

        relay_host: str | None = None
        if args.lm_connect == "mac-argo-relay":
            relay_process = _start_mac_argo_relay(
                profile=profile,
                host=operator_host,
                ssh_opts=ssh_opts,
                local_argo_host=args.local_argo_host,
                local_argo_port=args.local_argo_port,
                reverse_port=args.reverse_port,
                relay_port=relay_port,
                relay_python=relay_python,
                local_log_path=local_relay_log,
            )
            relay_host = _wait_for_relay(
                profile=profile,
                host=operator_host,
                ssh_opts=ssh_opts,
                relay_port=relay_port,
                relay_process=relay_process,
                local_log_path=local_relay_log,
            )
            lm_base_url = f"http://{relay_host}:{relay_port}/argoapi/v1"
        else:
            lm_base_url = str(args.lm_base_url)

        _log(f"Compute-node LM URL: {lm_base_url}")
        _write_operator_metadata(
            profile=profile,
            host=operator_host,
            ssh_opts=ssh_opts,
            run_id=args.run_id,
            campaign=args.campaign,
            lm_connect=args.lm_connect,
            lm_base_url=lm_base_url,
            relay_host=relay_host,
            relay_port=relay_port if relay_host else None,
        )

        _log("Starting rsync mirror:")
        _log(f"  {operator_host}:{remote_run_dir}/")
        _log(f"  {local_run_dir}/")
        _start_rsync_loop(
            host=operator_host,
            control_path=control_path,
            remote_run_dir=remote_run_dir,
            local_run_dir=local_run_dir,
            interval_s=args.rsync_interval_s,
            stop_event=stop_rsync,
        )

        _print_compute_command(
            profile=profile,
            wrapper_path=wrapper_path,
            run_id=args.run_id,
            campaign=args.campaign,
        )

        if args.no_dashboard:
            return 0

        _log("")
        _log(f"Starting dashboard at http://{args.dashboard_host}:{args.dashboard_port}")
        _log("Ctrl-C stops the local dashboard, rsync loop, and relay tunnel.")
        return _run_dashboard(
            local_run_dir=local_run_dir,
            host=args.dashboard_host,
            port=args.dashboard_port,
        )
    finally:
        stop_rsync.set()
        if relay_process is not None and relay_process.poll() is None:
            relay_process.terminate()
            try:
                relay_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                relay_process.kill()
        keep = args.keep_ssh_master or os.environ.get("CHEMGRAPH_ACADEMY_KEEP_SSH_MASTER") == "1"
        if started_ssh_master and not keep:
            _stop_ssh_master(host=operator_host, control_path=control_path)


if __name__ == "__main__":
    raise SystemExit(main())
