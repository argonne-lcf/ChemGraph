from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from chemgraph.academy.examples import campaign_launch_defaults
from chemgraph.academy.examples import resolve_builtin_campaign
from chemgraph.academy.examples import resolve_builtin_lm_config_template
from chemgraph.academy.runtime.profiles import list_builtin_system_profiles
from chemgraph.academy.runtime.profiles import load_system_profile
from chemgraph.academy.runtime.profiles.system import SystemProfile


DASHBOARD_METADATA_FILE = "dashboard_metadata.json"


@dataclasses.dataclass(frozen=True)
class AllocationPlan:
    """Resolved parameters needed to launch one MPI-backed campaign."""

    run_dir: Path
    run_token: str
    agent_count: int
    agents_per_node: int
    campaign_config: Path
    lm_config: Path
    max_decisions: int
    poll_timeout_s: float
    idle_timeout_s: float
    startup_timeout_s: float
    completion_timeout_s: float
    status_interval_s: float
    redis_host: str
    redis_port: int
    redis_bind: str
    redis_protected_mode: str
    redis_namespace: str
    start_redis: bool
    mpiexec: str
    chemgraph_repo_root: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a built-in ChemGraph Academy campaign inside the current "
            "HPC compute allocation."
        ),
    )
    parser.add_argument(
        "--system",
        required=True,
        help=(
            "Built-in system profile or profile JSON path. Built-ins: "
            + ", ".join(list_builtin_system_profiles())
        ),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--lm-base-url")
    parser.add_argument("--relay-host")
    parser.add_argument("--lm-model")
    parser.add_argument("--lm-user")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--agent-count", type=int)
    parser.add_argument("--agents-per-node", type=int)
    parser.add_argument("--max-decisions", type=int)
    parser.add_argument("--redis-port", type=int)
    parser.add_argument("--no-start-redis", action="store_true")
    return parser.parse_args(argv)


def _prepend_path(name: str, entries: list[str]) -> None:
    existing = os.environ.get(name, "")
    values = [entry for entry in entries if entry]
    if existing:
        values.append(existing)
    os.environ[name] = os.pathsep.join(values)


def _prepare_environment(profile: SystemProfile) -> None:
    for name in profile.unset_env:
        os.environ.pop(name, None)
    _prepend_path("PATH", profile.path_entries)
    _prepend_path("PYTHONPATH", profile.pythonpath_entries)
    for name, value in profile.env.items():
        os.environ.setdefault(name, value)
    os.environ["no_proxy"] = profile.no_proxy
    os.environ["NO_PROXY"] = profile.no_proxy


def _load_dashboard_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir / DASHBOARD_METADATA_FILE
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return data


def _relay_host_from_profile(profile: SystemProfile) -> str:
    path = Path(profile.relay_host_file)
    if not path.exists():
        raise RuntimeError(
            "Could not determine UAN relay host. Start the Mac dashboard "
            f"first, or pass --lm-base-url. Missing: {path}",
        )
    host = path.read_text(encoding="utf-8").strip()
    if not host:
        raise RuntimeError(f"Relay host file is empty: {path}")
    return host


def _resolve_lm_base_url(
    *,
    args: argparse.Namespace,
    profile: SystemProfile,
    metadata: dict[str, Any],
) -> str:
    if args.lm_base_url:
        return args.lm_base_url
    value = metadata.get("lm_base_url")
    if isinstance(value, str) and value.strip():
        return value.strip()
    relay_host = args.relay_host or metadata.get("relay_host")
    if not isinstance(relay_host, str) or not relay_host.strip():
        relay_host = _relay_host_from_profile(profile)
    return f"http://{relay_host.strip()}:{profile.relay_port}/argoapi/v1"


def _write_lm_config(
    *,
    run_dir: Path,
    template_name: str,
    base_url: str,
    lm_model: str | None,
    lm_user: str | None,
    max_tokens: int | None,
) -> Path:
    template_path = resolve_builtin_lm_config_template(template_name)
    data = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"LM template must contain a JSON object: {template_path}")
    data["base_url"] = base_url
    if lm_model:
        data["model"] = lm_model
    if lm_user:
        data["user"] = lm_user
    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    path = run_dir / "lm_config.json"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _write_compute_launch_metadata(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    profile: SystemProfile,
    lm_config: Path,
    lm_base_url: str,
    agent_count: int,
    agents_per_node: int,
    max_decisions: int,
    redis_port: int,
) -> None:
    payload = {
        "system": profile.name,
        "run_id": args.run_id,
        "campaign": args.campaign,
        "run_dir": str(run_dir),
        "lm_base_url": lm_base_url,
        "lm_config": str(lm_config),
        "agent_count": agent_count,
        "agents_per_node": agents_per_node,
        "max_decisions": max_decisions,
        "redis_host": socket.getfqdn(),
        "redis_port": redis_port,
        "repo_root": profile.repo_root,
    }
    (run_dir / "compute_launch.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _export_workflow_lm_environment(lm_config: Path) -> None:
    data = json.loads(lm_config.read_text(encoding="utf-8"))
    values = {
        "CHEMGRAPH_WORKFLOW_BASE_URL": data.get("base_url"),
        "CHEMGRAPH_WORKFLOW_MODEL": data.get("model"),
        "CHEMGRAPH_WORKFLOW_API_KEY": data.get("api_key"),
        "CHEMGRAPH_WORKFLOW_ARGO_USER": data.get("user"),
        "ARGO_USER": data.get("user"),
    }
    for name, value in values.items():
        if isinstance(value, str) and value:
            os.environ.setdefault(name, value)


def _run_token() -> str:
    return f"{int(time.time())}-{os.getpid()}"


def prepare_compute_launch(args: argparse.Namespace) -> AllocationPlan:
    """Resolve a system profile and dashboard metadata into an allocation plan."""
    profile = load_system_profile(args.system)
    _prepare_environment(profile)

    defaults = campaign_launch_defaults(args.campaign)
    run_dir = Path(args.run_dir or Path(profile.run_root) / args.run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = _load_dashboard_metadata(run_dir)
    metadata_campaign = metadata.get("campaign")
    if metadata_campaign and metadata_campaign != args.campaign:
        raise RuntimeError(
            f"Run metadata campaign {metadata_campaign!r} does not match "
            f"--campaign {args.campaign!r}",
        )

    lm_base_url = _resolve_lm_base_url(
        args=args,
        profile=profile,
        metadata=metadata,
    )
    lm_config = _write_lm_config(
        run_dir=run_dir,
        template_name=defaults.lm_config_template,
        base_url=lm_base_url,
        lm_model=args.lm_model,
        lm_user=args.lm_user,
        max_tokens=args.max_tokens,
    )
    _export_workflow_lm_environment(lm_config)
    agent_count = args.agent_count or defaults.agent_count
    agents_per_node = args.agents_per_node or defaults.agents_per_node
    max_decisions = args.max_decisions or defaults.max_decisions
    redis_port = args.redis_port or profile.redis_port

    _write_compute_launch_metadata(
        run_dir=run_dir,
        args=args,
        profile=profile,
        lm_config=lm_config,
        lm_base_url=lm_base_url,
        agent_count=agent_count,
        agents_per_node=agents_per_node,
        max_decisions=max_decisions,
        redis_port=redis_port,
    )

    campaign_config = resolve_builtin_campaign(args.campaign)
    if not campaign_config.exists():
        campaign_config = Path(args.campaign).resolve()

    return AllocationPlan(
        run_dir=run_dir,
        run_token=_run_token(),
        agent_count=agent_count,
        agents_per_node=agents_per_node,
        campaign_config=campaign_config,
        lm_config=lm_config,
        max_decisions=max_decisions,
        poll_timeout_s=2.0,
        idle_timeout_s=600.0,
        startup_timeout_s=120.0,
        completion_timeout_s=60.0,
        status_interval_s=5.0,
        redis_host=socket.getfqdn(),
        redis_port=redis_port,
        redis_bind=profile.redis_bind,
        redis_protected_mode=profile.redis_protected_mode,
        redis_namespace=f"academy-chemgraph-swarm:{args.run_id}",
        start_redis=not args.no_start_redis,
        mpiexec=profile.mpiexec,
        chemgraph_repo_root=Path(profile.repo_root).resolve(),
    )


def wait_redis(host: str, port: int, run_dir: Path) -> None:
    import redis

    deadline = time.time() + 30
    while True:
        try:
            redis.Redis(host=host, port=port).ping()
            return
        except Exception:
            if time.time() > deadline:
                log = run_dir / "redis.log"
                if log.exists():
                    print(log.read_text(errors="replace")[-4000:], file=sys.stderr)
                raise
            time.sleep(1)


def run_allocation(plan: AllocationPlan) -> int:
    """Start Redis if requested and run per-rank daemons under mpiexec."""
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    redis_proc: subprocess.Popen[bytes] | None = None
    if plan.start_redis:
        redis_server = shutil.which("redis-server")
        if redis_server is None:
            raise RuntimeError("redis-server is required unless --no-start-redis is set")
        redis_log = (plan.run_dir / "redis.log").open("ab")
        redis_proc = subprocess.Popen(
            [
                redis_server,
                "--bind",
                plan.redis_bind,
                "--port",
                str(plan.redis_port),
                "--protected-mode",
                plan.redis_protected_mode,
                "--save",
                "",
                "--appendonly",
                "no",
                "--daemonize",
                "no",
            ],
            stdout=redis_log,
            stderr=subprocess.STDOUT,
        )
        (plan.run_dir / "redis.pid").write_text(
            f"{redis_proc.pid}\n",
            encoding="utf-8",
        )
    try:
        wait_redis(plan.redis_host, plan.redis_port, plan.run_dir)
        daemon_args = [
            "--run-dir", str(plan.run_dir),
            "--run-token", plan.run_token,
            "--agent-count", str(plan.agent_count),
            "--campaign-config", str(plan.campaign_config),
            "--lm-config", str(plan.lm_config),
            "--max-decisions", str(plan.max_decisions),
            "--poll-timeout-s", str(plan.poll_timeout_s),
            "--idle-timeout-s", str(plan.idle_timeout_s),
            "--startup-timeout-s", str(plan.startup_timeout_s),
            "--completion-timeout-s", str(plan.completion_timeout_s),
            "--status-interval-s", str(plan.status_interval_s),
            "--redis-host", plan.redis_host,
            "--redis-port", str(plan.redis_port),
            "--redis-namespace", plan.redis_namespace,
            "--chemgraph-repo-root", str(plan.chemgraph_repo_root),
        ]
        cmd = [
            plan.mpiexec,
            "-n", str(plan.agent_count),
            "--ppn", str(plan.agents_per_node),
            sys.executable, "-m", "chemgraph.cli.main", "academy", "mpi-daemon", "--",
            *daemon_args,
        ]
        (plan.run_dir / "launch_command.txt").write_text(
            " ".join(cmd) + "\n",
            encoding="utf-8",
        )
        return subprocess.call(cmd)
    finally:
        if redis_proc is not None:
            redis_proc.terminate()
            try:
                redis_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                redis_proc.kill()
                redis_proc.wait()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = prepare_compute_launch(args)
    print(f"ChemGraph Academy run: {args.run_id}")
    print(f"  system: {load_system_profile(args.system).name}")
    print(f"  campaign: {args.campaign}")
    print(f"  run dir: {plan.run_dir}")
    print(f"  LM config: {plan.lm_config}")
    print(f"  agents: {plan.agent_count}, agents_per_node: {plan.agents_per_node}")
    return run_allocation(plan)


if __name__ == "__main__":
    raise SystemExit(main())
