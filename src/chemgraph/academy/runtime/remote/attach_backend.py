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
import sys
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
    # Venv activate script absolute path. We source this rather than a
    # made-up env.{system}.sh; the existing manual workflow does
    # `source /flare/.../venvs/academy-swarm/bin/activate` to put
    # ``chemgraph`` on PATH. Caller usually derives this from
    # system_profile.venv_python.
    venv_activate: str
    run_dir: str  # absolute path on the compute host (also visible to the dashboard side)
    # Env vars to export inside the remote bash before invoking
    # spawn-site. Operators type these manually today (ALCF_PROJECT,
    # ALCF_USER, ALCF_SSH_USER, ARGO_USER, http_proxy, no_proxy, ...).
    # The launcher forwards them from the operator's local environment
    # so the remote shell sees the same values.
    remote_env: dict[str, str] = dataclasses.field(default_factory=dict)
    # ssh target for the login node (e.g. "jinchuli@aurora.alcf.anl.gov").
    # Empty string -> direct ssh to compute_host. ALCF (Aurora, Crux) blocks
    # direct laptop->compute ssh; the laptop has to ssh into the login node
    # first, which then ssh's to the compute node using its in-cluster
    # hostbased trust. Provide this for any HPC where compute nodes aren't
    # reachable from the public internet.
    login_host: str = ""
    exchange_type: str = "http"
    http_exchange_url: str | None = None
    extra_args: tuple[str, ...] = ()


def _build_remote_command(cfg: AttachConfig) -> str:
    """Render the bash one-liner that ssh runs on the compute host.

    Logging strategy: the log file is opened FIRST (via exec
    redirection on the shell itself) so even pre-source failures
    (bad bundle_root, missing env.sh, mkdir refusing) leave a
    diagnostic trail in {site}.attach.log. Without this any error
    before the spawn-site exec produces zero output anywhere
    visible to the operator -- you only see "ssh exited" with no
    explanation.
    """
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

    log_path = f"{cfg.run_dir}/{cfg.site.name}.attach.log"
    # Render the env-var exports the operator would type manually
    # (ALCF_*, ARGO_USER, http_proxy, etc). Each value is shell-quoted.
    env_pairs = list(cfg.remote_env.items())
    # PBS env reconstruction: nested ssh into a compute node does
    # NOT inherit PBS_* from the operator's qsub -I shell, so
    # mpiexec defaults to "this host only" and refuses multi-rank
    # launches. With pbs_jobid in hand we reconstruct PBS_JOBID and
    # PBS_NODEFILE from the deterministic path
    # ``/var/spool/pbs/aux/$PBS_JOBID`` (any process inside the
    # allocation can read it).
    if cfg.site.pbs_jobid:
        env_pairs.append(("PBS_JOBID", cfg.site.pbs_jobid))
        env_pairs.append(("PBS_NODEFILE", f"/var/spool/pbs/aux/{cfg.site.pbs_jobid}"))
    env_exports = "; ".join(
        f"export {k}={ssh_quote(v)}"
        for k, v in sorted(env_pairs)
        if v
    )
    env_block = (env_exports + "; ") if env_exports else ""

    # Three-stage script:
    # 1. mkdir the run_dir up front. Without an existing run_dir we
    #    can't open the log file -- chicken-and-egg.
    # 2. Re-exec self with stdout+stderr redirected to the log so
    #    every subsequent failure (source, cd, exec) is captured.
    # 3. ``set -x`` so the log shows what the shell actually tried.
    script = (
        f"mkdir -p {ssh_quote(cfg.run_dir)} 2>&1 || "
        f'  {{ echo "mkdir failed: cannot create {cfg.run_dir}" >&2; exit 1; }}; '
        f"exec >> {ssh_quote(log_path)} 2>&1; "
        f"set -ex; "
        f"echo \"attach.log opened at $(date -u +%FT%TZ) on $(hostname)\"; "
        f"{env_block}"
        f"source {ssh_quote(cfg.venv_activate)}; "
        f"cd {ssh_quote(cfg.bundle_root)}; "
        f"exec {inner_q}"
    )
    return f"bash -lc {ssh_quote(script)}"


def start(cfg: AttachConfig) -> subprocess.Popen[bytes]:
    """Fire off the remote spawn-site. Returns the local ssh Popen
    handle so the launcher can SIGTERM-cancel later. Does not block
    for readiness -- caller polls placement.json.

    Transport selection:
      login_host set -> nested ssh: laptop -> login -> compute.
        Required on ALCF where compute nodes only accept
        connections from inside the cluster (hostbased trust from
        login node, no laptop-side key option). We verified that
        ProxyJump ("ssh -J login compute") does NOT work on
        Aurora because the inner hop attempts publickey/hostbased
        auth from the laptop's perspective even though the bytes
        flow through the login node -- the compute's authorized_
        keys only trust the login node itself.
      login_host empty -> direct ssh to compute_host. Works on
        HPCs whose compute nodes are routable + accept laptop
        keys (e.g. some on-prem clusters).

    ``-tt`` forces PTY allocation on the FINAL hop so the python
    process has a controlling terminal that gets SIGHUP'd when the
    SSH chain closes. Without this, Ctrl-C kills the laptop ssh
    but the remote python survives as an orphan.

    SSH stderr is INHERITED (not DEVNULL'd) so the operator sees
    auth failures, host-unreachable, etc. directly on the launcher's
    terminal. The remote bash redirects its own output to the
    per-site attach.log on the compute host -- this stderr channel
    only catches ssh-layer problems.
    """
    assert cfg.site.compute_host, "attach mode requires compute_host"
    remote = _build_remote_command(cfg)
    # ServerAliveInterval keeps the SSH connection (and therefore
    # the controlling PTY allocated by -tt) warm during long stretches
    # of no output from the remote -- e.g. spawn-site's
    # wait_for_peers_alive can sit silent for minutes while the other
    # site's allocation comes up. Without keepalives, the PTY closes,
    # remote python takes SIGHUP, agent dies as ``signal 1``. The
    # values mirror what the dashboard launcher uses for its own
    # long-lived ssh tunnels.
    keepalive = [
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=999",
    ]
    if cfg.login_host:
        # Nested: ssh laptop->login, then on the login node run
        # `ssh -tt compute <remote>`. The inner ssh inherits the
        # login node's identity / hostbased trust to reach compute.
        # We quote the inner argv as a single bash command for the
        # outer ssh to execute. The keepalive flags need to be on
        # BOTH ssh invocations -- the inner one is what wraps the
        # python process's PTY.
        inner_keepalive = " ".join(keepalive)
        inner = (
            f"ssh -tt {inner_keepalive} "
            f"{ssh_quote(cfg.site.compute_host)} {ssh_quote(remote)}"
        )
        argv = ["ssh", "-tt", *keepalive, cfg.login_host, inner]
    else:
        argv = ["ssh", "-tt", *keepalive, cfg.site.compute_host, remote]
    return subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=None,  # inherit
        stdin=subprocess.DEVNULL,
    )


async def wait_ready(
    cfg: AttachConfig,
    *,
    local_run_dir: Path,
    timeout_s: float = 300.0,
    poll_interval_s: float = 5.0,
    proc: subprocess.Popen[bytes] | None = None,
    log_interval_s: float = 30.0,
) -> set[str]:
    """Wait until every agent in ``cfg.site.agents`` has registered.

    The daemon writes ``{run_dir}/placement.json`` via the existing
    write_status_snapshot path. Either a local mirror (rsync'd from
    the compute host) or a shared filesystem must populate it.

    Args:
        proc: the ssh Popen handle returned by start(). If provided,
            we check whether it has died and short-circuit with a
            useful error rather than waiting out the full timeout.
        log_interval_s: print a "still waiting" line at this cadence
            so the operator sees the launcher isn't hung.

    Returns the set of registered agent names on success; raises
    TimeoutError otherwise. Raises RuntimeError if the underlying
    ssh process exits before agents register.
    """
    deadline = time.monotonic() + timeout_s
    next_log = time.monotonic() + log_interval_s
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

    # Use the login node as the ssh target for read-only probes
    # (cat placement.json, tail attach.log). placement.json lives on
    # the SHARED home/project FS, which the login node can read
    # without bouncing through the compute host -- one less hop,
    # avoids re-prompting agent forwarding, and works regardless of
    # whether compute->login is reachable bidirectionally.
    probe_host = cfg.login_host or cfg.site.compute_host

    async def _read_remote() -> set[str]:
        try:
            r = ssh_run(
                probe_host,  # type: ignore[arg-type]
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

    async def _tail_attach_log(n: int = 30) -> str:
        try:
            r = ssh_run(
                probe_host,  # type: ignore[arg-type]
                f"tail -n {n} {ssh_quote(cfg.run_dir + '/' + cfg.site.name + '.attach.log')} 2>/dev/null || true",
                timeout_s=10,
                check=False,
            )
        except subprocess.SubprocessError:
            return ""
        return r.stdout

    while time.monotonic() < deadline:
        # Early exit if the ssh died -- no point polling for a file
        # nobody will ever write.
        if proc is not None and proc.poll() is not None:
            tail = await _tail_attach_log()
            raise RuntimeError(
                f"attach[{cfg.site.name}]: ssh exited with code "
                f"{proc.returncode} before agents registered. "
                f"Last lines of {cfg.run_dir}/{cfg.site.name}.attach.log:\n"
                f"{tail or '  (no log written -- bash failed before opening it)'}",
            )

        registered = await _read_local()
        if not (want & registered):
            registered = await _read_remote()
        if want.issubset(registered):
            return want

        if time.monotonic() >= next_log:
            missing = sorted(want - registered)
            print(
                f"[attach:{cfg.site.name}] waiting for {missing} to register "
                f"(elapsed {int(time.monotonic() - (deadline - timeout_s))}s "
                f"of {int(timeout_s)}s)",
                file=sys.stderr,
            )
            next_log = time.monotonic() + log_interval_s

        await asyncio.sleep(poll_interval_s)

    tail = await _tail_attach_log()
    raise TimeoutError(
        f"attach[{cfg.site.name}]: agents {sorted(want)} did not register "
        f"within {timeout_s:.0f}s. Last lines of "
        f"{cfg.run_dir}/{cfg.site.name}.attach.log:\n"
        f"{tail or '  (no log written)'}",
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
            proc=self._proc,
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
        venv_activate="/flare/ChemGraph/jinchu/venvs/academy-swarm/bin/activate",
        run_dir="/flare/ChemGraph/jinchu/runs/run-008",
        remote_env={"ARGO_USER": "jinchu.li", "http_proxy": "http://proxy.alcf.anl.gov:3128"},
        http_exchange_url=None,
    )
    cmd = _build_remote_command(cfg)
    assert "spawn-site" in cmd
    assert "--agents alpha" in cmd
    assert "--exchange-type http" in cmd
    assert "exec" in cmd
    assert "attach.log" in cmd
    assert "source" in cmd and "activate" in cmd
    assert "export ARGO_USER=" in cmd
    print("attach_backend self-check ok")
    print(cmd)
