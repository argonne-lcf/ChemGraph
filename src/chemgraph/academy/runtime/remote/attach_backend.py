"""Attach-mode backend: ssh straight to a compute node inside the
operator's already-running interactive PBS allocation and exec
``chemgraph academy spawn-site`` there.

Readiness signal: the daemon's existing per-rank
``placement.json`` write (via observability.write_status_snapshot)
materialises once an agent has registered on the exchange and
entered runtime. We poll for it instead of inventing a new
status-file protocol.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path

from chemgraph.academy.runtime.remote.site_spec import SiteSpec
from chemgraph.academy.runtime.remote.ssh_transport import ssh_quote, ssh_run


@dataclasses.dataclass
class AttachConfig:
    """Inputs the launcher resolves before the backend takes over."""

    site: SiteSpec
    run_id: str
    campaign: str  # campaign id or path passed to spawn-site
    bundle_root: str  # remote checkout root, e.g. /flare/ChemGraph/jinchu/ChemGraph
    env_script: str  # absolute path on the compute host, e.g. {bundle_root}/env.aurora.sh
    run_dir: str  # absolute path on the compute host (also visible to the dashboard side)
    exchange_type: str = "http"
    http_exchange_url: str | None = None
    extra_args: tuple[str, ...] = ()


def _build_remote_command(cfg: AttachConfig) -> str:
    """Render the bash one-liner that ssh runs on the compute host."""
    inner = [
        "chemgraph", "academy", "spawn-site", "--",
        "--system", cfg.site.name,
        "--run-id", cfg.run_id,
        "--campaign", cfg.campaign,
        "--run-dir", cfg.run_dir,
        "--agents", ",".join(cfg.site.agents),
        "--exchange-type", cfg.exchange_type,
    ]
    if cfg.http_exchange_url:
        inner += ["--http-exchange-url", cfg.http_exchange_url]
    inner += list(cfg.extra_args)
    inner_q = " ".join(ssh_quote(s) for s in inner)

    # Wrap in bash that sources env, cds to bundle, redirects the
    # spawn-site output to a per-attach log, and `exec`s so SIGTERM
    # from ssh-disconnect propagates to the python process instead of
    # being caught by the bash wrapper.
    log_path = f"{cfg.run_dir}/{cfg.site.name}.attach.log"
    script = (
        f"set -e; "
        f"mkdir -p {ssh_quote(cfg.run_dir)}; "
        f"source {ssh_quote(cfg.env_script)}; "
        f"cd {ssh_quote(cfg.bundle_root)}; "
        f"exec {inner_q} >> {ssh_quote(log_path)} 2>&1"
    )
    return f"bash -lc {ssh_quote(script)}"


def start(cfg: AttachConfig) -> subprocess.Popen[bytes]:
    """Fire off the remote spawn-site. Returns the local ssh Popen
    handle so the launcher can SIGTERM-cancel later. Does not block
    for readiness -- caller polls placement.json.

    ``-tt`` forces PTY allocation on the remote side so the python
    process has a controlling terminal that gets SIGHUP'd when the
    local ssh dies. Without this, Ctrl-C on the launcher kills the
    local ssh but leaves the remote python running inside the
    operator's allocation (the same orphan family as the old UAN
    relay bug, but at the SSH layer rather than the script layer).
    """
    assert cfg.site.compute_host, "attach mode requires compute_host"
    remote = _build_remote_command(cfg)
    argv = ["ssh", "-tt", cfg.site.compute_host, remote]
    return subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


async def wait_ready(
    cfg: AttachConfig,
    *,
    local_run_dir: Path,
    timeout_s: float = 300.0,
    poll_interval_s: float = 5.0,
) -> set[str]:
    """Wait until every agent in ``cfg.site.agents`` has registered.

    The daemon writes ``{run_dir}/placement.json`` via the existing
    write_status_snapshot path. Either a local mirror (rsync'd from
    the compute host) or a shared filesystem must populate it.

    For phase 1 we poll a local path because the operator already
    runs the dashboard launcher which rsync-mirrors. If neither
    mirror nor shared FS is present, falls back to ssh+cat.

    Returns the set of registered agent names on success; raises
    TimeoutError otherwise.
    """
    deadline = time.monotonic() + timeout_s
    want = set(cfg.site.agents)
    local_placement = local_run_dir / "placement.json"

    async def _read_local() -> set[str]:
        if not local_placement.exists():
            return set()
        try:
            data = json.loads(local_placement.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        agents = data.get("agents") if isinstance(data, dict) else None
        return set(agents.keys()) if isinstance(agents, dict) else set()

    async def _read_remote() -> set[str]:
        try:
            r = ssh_run(
                cfg.site.compute_host,  # type: ignore[arg-type]
                f"cat {ssh_quote(cfg.run_dir + '/placement.json')} 2>/dev/null || true",
                timeout_s=10,
                check=False,
            )
        except subprocess.SubprocessError:
            return set()
        if not r.stdout.strip():
            return set()
        try:
            data = json.loads(r.stdout)
        except json.JSONDecodeError:
            return set()
        agents = data.get("agents") if isinstance(data, dict) else None
        return set(agents.keys()) if isinstance(agents, dict) else set()

    while time.monotonic() < deadline:
        registered = await _read_local()
        if not (want & registered):
            registered = await _read_remote()
        if want.issubset(registered):
            return want
        await asyncio.sleep(poll_interval_s)
    raise TimeoutError(
        f"attach[{cfg.site.name}]: agents {sorted(want)} did not register "
        f"within {timeout_s:.0f}s",
    )


def stop(proc: subprocess.Popen[bytes], *, force: bool = False) -> None:
    """SIGTERM the ssh process (forwards to the remote bash, which
    via ``exec`` is replaced by the compute process -- so SIGTERM
    lands on python). Does NOT touch the operator's allocation.
    """
    if proc.poll() is not None:
        return
    sig = "kill" if force else "terminate"
    getattr(proc, sig)()


class AttachSiteBackend:
    """SiteBackend implementation for attach-mode."""

    def __init__(self, cfg: AttachConfig, local_run_dir: Path) -> None:
        self.cfg = cfg
        self.local_run_dir = local_run_dir
        self.site_name = cfg.site.name
        self._proc: subprocess.Popen[bytes] | None = None

    async def start(self) -> None:
        self._proc = start(self.cfg)

    async def wait_ready(
        self,
        *,
        local_run_dir: Path,
        timeout_s: float,
    ) -> set[str]:
        # ``local_run_dir`` from the orchestrator wins over the one
        # captured at construction time so the launcher can override.
        return await wait_ready(
            self.cfg,
            local_run_dir=local_run_dir,
            timeout_s=timeout_s,
        )

    async def stop(self, *, force: bool = False) -> None:
        if self._proc is not None:
            stop(self._proc, force=force)


if __name__ == "__main__":  # ponytail: command-rendering only, no live ssh
    cfg = AttachConfig(
        site=SiteSpec(
            name="aurora",
            mode="attach",
            agents=("alpha",),
            compute_host="x4505c5s0b0n0",
        ),
        run_id="run-008",
        campaign="federated-chat",
        bundle_root="/flare/ChemGraph/jinchu/ChemGraph",
        env_script="/flare/ChemGraph/jinchu/ChemGraph/env.aurora.sh",
        run_dir="/flare/ChemGraph/jinchu/runs/run-008",
        http_exchange_url=None,
    )
    cmd = _build_remote_command(cfg)
    assert "spawn-site" in cmd
    assert "--agents alpha" in cmd
    assert "--exchange-type http" in cmd
    assert "exec" in cmd
    assert "attach.log" in cmd
    print("attach_backend self-check ok")
    print(cmd)
