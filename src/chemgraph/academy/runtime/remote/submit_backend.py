"""Submit-mode backend: render a PBS script, qsub it via ssh to the
login node, poll qstat for state transitions, and poll for daemon
placement.json once the job goes to R.

Why pbs script rendering lives inline (no separate job_script.py):
the template is ~20 lines and only one caller renders it. Splitting
would be theater. Promote if a second caller appears.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from chemgraph.academy.runtime.remote.site_spec import SiteSpec
from chemgraph.academy.runtime.remote.ssh_transport import scp_upload, ssh_quote, ssh_run


PbsState = Literal["Q", "R", "E", "F", "X", "H", "S", "T", "W", "U"]


async def _wait_placement_ready(
    *,
    site: SiteSpec,
    run_dir: str,
    login_host: str,
    local_run_dir: Path,
    diagnostic_log_name: str,
    terminated: Callable[[], str | None] | None = None,
    poll_interval_s: float = 5.0,
    log_interval_s: float = 30.0,
) -> set[str]:
    """Wait until every agent in ``site.agents`` has registered.

    Polls indefinitely; terminal conditions:
    - all wanted agents registered  -> returns their set
    - ``terminated()`` returns a non-None reason -> RuntimeError
      (submit-mode wraps qstat to detect PBS F/X/E)

    No dashboard-side deadline: PBS walltime is the authoritative ceiling.

    Registration signal: each daemon writes ONE file
    ``agent_status/<agent_name>.json`` under the run_dir. Filename
    presence is race-free (each daemon owns its filename), unlike a
    shared placement.json which two rank-0 daemons on a shared FS would
    clobber.
    """
    started_at = time.monotonic()
    next_log = started_at + log_interval_s
    want = set(site.agents)
    local_state_dir = local_run_dir / "agent_status"
    probe_host = login_host

    async def _read_local() -> set[str]:
        if not local_state_dir.is_dir():
            return set()
        try:
            return {p.stem for p in local_state_dir.glob("*.json") if p.is_file()}
        except OSError:
            return set()

    async def _read_remote() -> set[str]:
        try:
            r = ssh_run(
                probe_host,  # type: ignore[arg-type]
                f"ls -1 {ssh_quote(run_dir + '/agent_status')}/ 2>/dev/null "
                "| sed -n 's/\\.json$//p' || true",
                timeout_s=10,
                check=False,
            )
        except subprocess.SubprocessError:
            return set()
        return {line.strip() for line in r.stdout.splitlines() if line.strip()}

    async def _tail_log(n: int = 30) -> str:
        try:
            r = ssh_run(
                probe_host,  # type: ignore[arg-type]
                f"tail -n {n} {ssh_quote(run_dir + '/' + diagnostic_log_name)} 2>/dev/null || true",
                timeout_s=10,
                check=False,
            )
        except subprocess.SubprocessError:
            return ""
        return r.stdout

    while True:
        if terminated is not None:
            reason = terminated()
            if reason is not None:
                tail = await _tail_log()
                raise RuntimeError(
                    f"submit[{site.name}]: {reason} before agents "
                    f"registered. Last lines of {run_dir}/{diagnostic_log_name}:\n"
                    f"{tail or '  (no log written)'}",
                )

        registered = await _read_local()
        if not (want & registered):
            registered = await _read_remote()
        if want.issubset(registered):
            return want

        if time.monotonic() >= next_log:
            missing = sorted(want - registered)
            elapsed = int(time.monotonic() - started_at)
            print(
                f"[submit:{site.name}] waiting for {missing} to register "
                f"(elapsed {elapsed}s; PBS walltime is the ceiling)",
                file=sys.stderr,
            )
            next_log = time.monotonic() + log_interval_s

        await asyncio.sleep(poll_interval_s)


@dataclasses.dataclass
class SubmitConfig:
    site: SiteSpec
    run_id: str
    campaign: str
    login_host: str  # ssh target, e.g. "jinchuli@aurora.alcf.anl.gov"
    bundle_root: str
    env_script: str
    run_dir: str
    exchange_type: str = "http"
    http_exchange_url: str | None = None
    # PBS knobs that don't come from the --site flag.
    project: str | None = None
    filesystems: str | None = None
    extra_pbs_lines: tuple[str, ...] = ()
    # Extra argv appended verbatim to the rendered spawn-site
    # invocation. -- lets
    # the launcher pass through spawn-site flags it doesn't have a
    # dedicated --launcher-flag for (--agents-per-node, etc.).
    extra_spawn_args: tuple[str, ...] = ()
    # Env vars to export inside the PBS script before invoking
    # spawn-site.  PBS batch
    # jobs only inherit a tiny default env, so without an explicit
    # export here, system profiles that substitute ${ALCF_USER},
    # ${ALCF_PROJECT}, etc. fail to load with "unresolved environment
    # variables". The launcher populates this from the operator's
    # local shell.
    remote_env: dict[str, str] = dataclasses.field(default_factory=dict)
    # Optional operator-authored PBS script. When set, replaces the
    # built-in template body after ``${VAR}`` substitution. Supported
    # variables: ${PROJECT}, ${QUEUE}, ${WALLTIME}, ${NODES},
    # ${FILESYSTEMS}, ${RUN_DIR}, ${BUNDLE_ROOT}, ${ENV_SCRIPT},
    # ${ENV_EXPORTS}, ${SPAWN_INVOCATION}, ${SITE}. Set from the
    # canvas Launch editor via launch_defaults.per_site_overrides.
    pbs_script_template: str | None = None


def _spawn_invocation(cfg: SubmitConfig) -> str:
    site = cfg.site
    spawn_args = [
        "swarm", "spawn-site", "--",
        "--system", site.name,
        "--run-id", cfg.run_id,
        "--campaign", cfg.campaign,
        "--run-dir", cfg.run_dir,
        "--agents", ",".join(site.agents),
        "--exchange-type", cfg.exchange_type,
    ]
    if cfg.http_exchange_url:
        spawn_args += ["--http-exchange-url", cfg.http_exchange_url]
    spawn_args += list(cfg.extra_spawn_args)
    return " ".join(spawn_args)


def _env_export_block(cfg: SubmitConfig) -> str:
    """Bash lines that export each remote_env var. One per line."""
    lines: list[str] = []
    for k in sorted(cfg.remote_env):
        v = cfg.remote_env[k]
        if v:
            lines.append(f"export {k}={ssh_quote(v)}")
    return "\n".join(lines)


def _pbs_substitution_values(cfg: SubmitConfig) -> dict[str, str]:
    """Vars available for ${VAR} substitution in a user-authored PBS
    template. Kept small and load-bearing -- if the operator wants
    something we haven't exposed, they can hardcode it in their
    template (their template runs verbatim, they're an HPC user).
    """
    site = cfg.site
    project = site.project or cfg.project or ""
    filesystems = site.filesystems or cfg.filesystems or ""
    return {
        "SITE": site.name,
        "PROJECT": project,
        "QUEUE": site.queue or "",
        "WALLTIME": site.walltime or "",
        "NODES": str(site.nodes),
        "FILESYSTEMS": filesystems,
        "RUN_DIR": cfg.run_dir,
        "BUNDLE_ROOT": cfg.bundle_root,
        "ENV_SCRIPT": cfg.env_script,
        "ENV_EXPORTS": _env_export_block(cfg),
        "SPAWN_INVOCATION": _spawn_invocation(cfg),
        "RUN_ID": cfg.run_id,
        "CAMPAIGN": cfg.campaign,
    }


def render_pbs_script(cfg: SubmitConfig) -> str:
    site = cfg.site

    if cfg.pbs_script_template:
        # Operator-authored path: expand ${VAR}s from our derived table
        # and ship the result verbatim. string.Template rejects
        # unknown vars via KeyError so a typo like ${RUN_DR} surfaces
        # at qsub time instead of producing a broken script.
        import string
        try:
            return string.Template(cfg.pbs_script_template).substitute(
                _pbs_substitution_values(cfg),
            )
        except KeyError as exc:
            raise ValueError(
                f"submit[{site.name}]: PBS template references unknown "
                f"variable ${{{exc.args[0]}}}. Supported: "
                f"{sorted(_pbs_substitution_values(cfg))}",
            ) from exc

    # Built-in template path (no override).
    assert site.queue and site.walltime
    project = site.project or cfg.project
    if not project:
        raise ValueError(
            f"submit[{site.name}]: PBS -A project is required. "
            "Pass project=... on --site or set --project.",
        )
    filesystems = site.filesystems or cfg.filesystems

    pbs_lines = [
        "#!/bin/bash",
        f"#PBS -A {project}",
        f"#PBS -q {site.queue}",
        f"#PBS -l select={site.nodes},walltime={site.walltime}",
    ]
    if filesystems:
        pbs_lines.append(f"#PBS -l filesystems={filesystems}")
    pbs_lines += [
        "#PBS -j oe",
        f"#PBS -o {cfg.run_dir}/{site.name}.pbs.log",
        *cfg.extra_pbs_lines,
        "",
        "set -e",
        f"mkdir -p {cfg.run_dir}",
    ]
    # PBS batch jobs only inherit a tiny default env (no ALCF_USER,
    # ARGO_USER, http_proxy from the operator's login shell). System
    # profiles do ${ALCF_USER}/${ALCF_PROJECT} substitution at load
    # time -- without these exports the daemon dies with
    # "unresolved environment variables" before anything else runs.
    # Render BEFORE source so the env_script can also reference them
    # if it needs to.
    env_block = _env_export_block(cfg)
    if env_block:
        pbs_lines.append(env_block)
    pbs_lines += [
        f"source {cfg.env_script}",
        f"cd {cfg.bundle_root}",
        "",
    ]
    pbs_lines.append(_spawn_invocation(cfg))
    pbs_lines.append("")
    return "\n".join(pbs_lines)


def _job_id_path(cfg: SubmitConfig, local_run_dir: Path | None = None) -> Path | None:
    """Where to persist the qsub'd job id locally so a stop subcommand
    (or operator manual qdel) can find it after a Ctrl-C.

    Returns None if no usable local directory is available. We don't
    write to ``cfg.run_dir`` directly because that path lives on the
    compute host, not the operator's laptop.
    """
    if local_run_dir is None:
        return None
    try:
        local_run_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return local_run_dir / f"{cfg.site.name}.job_id"


def submit_job(
    cfg: SubmitConfig,
    *,
    local_run_dir: Path | None = None,
) -> str:
    """Render the PBS script, scp it to the login node, run qsub.

    Persists the returned job_id to ``<local_run_dir>/<site>.job_id``
    BEFORE returning so a Ctrl-C between qsub and the launcher's
    next await can't lose the id. Without this, an orphan PBS job
    could end up running on the HPC with no record of its id on
    the laptop.
    """
    text = render_pbs_script(cfg)
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".pbs",
        delete=False,
    ) as f:
        f.write(text)
        local_path = f.name
    remote_path = f"/tmp/chemgraph-{cfg.site.name}-{cfg.run_id}.pbs"
    try:
        # Pre-create run_dir on the login node BEFORE qsub. PBS opens
        # the `-o <path>` file at job-start time (not at script-run
        # time), so if run_dir does not exist yet, PBS silently fails
        # to write the log -- the job runs to completion but its
        # stdout is lost, producing a 0-byte ``<site>.pbs.log`` and
        # a "no log written" diagnostic from wait_ready. Seen on Crux
        # the first time a fresh run_id was launched. The script body
        # also does ``mkdir -p``, but that runs AFTER PBS has already
        # bound the output file -- too late. ssh + mkdir is cheap;
        # ControlMaster reuses the existing connection.
        ssh_run(
            cfg.login_host,
            f"mkdir -p {ssh_quote(cfg.run_dir)}",
            timeout_s=30,
        )
        # scp the script over. scp_upload reuses ControlMaster so the
        # file lands on the SAME login node the next ssh_run lands on.
        # Without this, aurora.alcf.anl.gov round-robin DNS puts the
        # scp and qsub on different UAN hosts, and qsub fails with
        # "script file: No such file or directory" because /tmp is
        # per-node. ssh_run below also uses ControlMaster, so both
        # calls pin to the same warmed socket.
        scp_res = scp_upload(local_path, cfg.login_host, remote_path)
        if scp_res.returncode != 0:
            raise RuntimeError(
                f"submit[{cfg.site.name}]: scp failed: "
                f"{scp_res.stderr.strip() or scp_res.stdout.strip()}"
            )
        r = ssh_run(
            cfg.login_host,
            f"qsub {ssh_quote(remote_path)}",
            timeout_s=60,
        )
    finally:
        Path(local_path).unlink(missing_ok=True)
    job_id = r.stdout.strip()
    if not job_id:
        raise RuntimeError(f"submit[{cfg.site.name}]: qsub returned no job id")
    # Persist BEFORE returning so the value survives a launcher Ctrl-C.
    record = _job_id_path(cfg, local_run_dir)
    if record is not None:
        try:
            record.write_text(job_id + "\n", encoding="utf-8")
        except OSError:
            pass  # not fatal -- the operator still sees it on stderr
    return job_id


def qstat_state(cfg: SubmitConfig, job_id: str) -> PbsState | None:
    """Return the one-letter PBS state, or None if the job isn't listed.

    Uses ``qstat -x -f -F json`` for stable parsing across PBS Pro
    versions (OpenPBS on Aurora; PBS Pro on Crux/Polaris both honor
    -F json).
    """
    try:
        r = ssh_run(
            cfg.login_host,
            f"qstat -x -f -F json {ssh_quote(job_id)}",
            timeout_s=30,
            check=False,
        )
    except subprocess.SubprocessError:
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    jobs = data.get("Jobs") if isinstance(data, dict) else None
    if not isinstance(jobs, dict):
        return None
    job = jobs.get(job_id) or next(iter(jobs.values()), None)
    if not isinstance(job, dict):
        return None
    state = job.get("job_state")
    return state if isinstance(state, str) and state else None


def _color_state(state: str | None) -> str:
    """Color a PBS job-state letter for human eyes. Defers to the
    launcher's ANSI helpers via a lazy import (avoids circular)."""
    from chemgraph.academy.runtime.remote.remote_launcher import (
        _green, _yellow, _red, _bold,
    )
    if state == "R":
        return _green(_bold(state))
    if state == "Q":
        return _yellow(state)
    if state in ("H", "S", "T", "W", "U"):
        return _yellow(state)
    if state in ("F", "E", "X"):
        return _red(_bold(state))
    return _bold(str(state))


async def wait_running(
    cfg: SubmitConfig,
    job_id: str,
    *,
    poll_interval_s: float = 15.0,
) -> None:
    """Block until ``job_id`` transitions to R.

    Polls indefinitely; PBS walltime is the only ceiling. Raises
    RuntimeError if the job reaches a terminal state (F/X/E) before
    R -- covers the "job dies in queue" edge case that would
    otherwise loop forever.
    """
    last_state: str | None = None
    while True:
        state = qstat_state(cfg, job_id)
        if state and state != last_state:
            print(
                f"[submit:{cfg.site.name}] job {job_id} state -> "
                f"{_color_state(state)}",
                file=sys.stderr,
            )
            last_state = state
        if state == "R":
            return
        if state in ("F", "X", "E"):
            raise RuntimeError(
                f"submit[{cfg.site.name}]: job {job_id} reached terminal "
                f"state {state} before running",
            )
        await asyncio.sleep(poll_interval_s)


def qdel_job(cfg: SubmitConfig, job_id: str) -> None:
    try:
        ssh_run(
            cfg.login_host,
            f"qdel {ssh_quote(job_id)}",
            timeout_s=30,
            check=False,
        )
    except subprocess.SubprocessError:
        pass


class SubmitSiteBackend:
    """SiteBackend impl for submit-mode."""

    def __init__(self, cfg: SubmitConfig, local_run_dir: Path | None = None) -> None:
        self.cfg = cfg
        self.site_name = cfg.site.name
        self.local_run_dir = local_run_dir
        self.job_id: str | None = None

    async def start(self) -> None:
        # Run blocking ssh in a thread so the orchestrator's event
        # loop keeps polling other sites' progress concurrently. The
        # thread persists job_id to disk before returning, so a Ctrl-C
        # arriving here can't strand a PBS job with no traceable id.
        self.job_id = await asyncio.to_thread(
            submit_job, self.cfg, local_run_dir=self.local_run_dir,
        )
        from chemgraph.academy.runtime.remote.remote_launcher import _cyan, _bold
        print(
            f"[submit:{self.cfg.site.name}] qsub -> {_cyan(_bold(self.job_id))}",
            file=sys.stderr,
        )

    async def wait_ready(
        self,
        *,
        local_run_dir: Path,
    ) -> set[str]:
        if not self.job_id:
            raise RuntimeError("submit backend: start() must precede wait_ready()")
        # No dashboard-side deadline: PBS walltime is the ceiling.
        # wait_running exits when job goes R; if it dies in queue we
        # raise. Post-R registration then polls indefinitely too,
        # short-circuiting only when qstat reports a terminal state.
        await wait_running(self.cfg, self.job_id)
        job_id = self.job_id
        cfg = self.cfg
        def _pbs_terminated() -> str | None:
            state = qstat_state(cfg, job_id)
            if state in ("F", "X", "E"):
                return f"job {job_id} reached PBS state {state}"
            return None
        return await _wait_placement_ready(
            site=self.cfg.site,
            run_dir=self.cfg.run_dir,
            login_host=self.cfg.login_host,
            local_run_dir=local_run_dir,
            diagnostic_log_name=f"{self.cfg.site.name}.pbs.log",
            terminated=_pbs_terminated,
        )

    async def stop(self, *, force: bool = False) -> None:
        if self.job_id:
            await asyncio.to_thread(qdel_job, self.cfg, self.job_id)


if __name__ == "__main__":  # ponytail: command-rendering self-check
    cfg = SubmitConfig(
        site=SiteSpec(
            name="aurora",
            agents=("alpha", "beta"),
            queue="debug",
            walltime="01:00:00",
            nodes=1,
            project="MYPROJ",
            filesystems="home:flare",
        ),
        run_id="run-008",
        campaign="federated-chat",
        login_host="jinchuli@aurora.alcf.anl.gov",
        bundle_root="/flare/MYPROJ/jinchu/ChemGraph",
        env_script="/flare/MYPROJ/jinchu/ChemGraph/env.aurora.sh",
        run_dir="/flare/MYPROJ/jinchu/runs/run-008",
        http_exchange_url=None,
    )
    text = render_pbs_script(cfg)
    assert "#PBS -A MYPROJ" in text
    assert "#PBS -q debug" in text
    assert "#PBS -l select=1,walltime=01:00:00" in text
    assert "#PBS -l filesystems=home:flare" in text
    assert "spawn-site" in text
    assert "--agents alpha,beta" in text
    assert "--exchange-type http" in text
    print("submit_backend self-check ok")
