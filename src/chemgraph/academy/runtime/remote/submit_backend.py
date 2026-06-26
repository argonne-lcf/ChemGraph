"""Submit-mode backend: render a PBS script, qsub it via ssh to the
login node, poll qstat for state transitions, and forward to attach-
mode's placement.json polling once the job goes to R.

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
from pathlib import Path
from typing import Literal

from chemgraph.academy.runtime.remote.attach_backend import wait_ready as _placement_wait_ready
from chemgraph.academy.runtime.remote.attach_backend import AttachConfig
from chemgraph.academy.runtime.remote.site_spec import SiteSpec
from chemgraph.academy.runtime.remote.ssh_transport import ssh_quote, ssh_run


PbsState = Literal["Q", "R", "E", "F", "X", "H", "S", "T", "W", "U"]


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
    # invocation. Same purpose as AttachConfig.extra_args -- lets
    # the launcher pass through spawn-site flags it doesn't have a
    # dedicated --launcher-flag for (--agents-per-node, etc.).
    extra_spawn_args: tuple[str, ...] = ()


def render_pbs_script(cfg: SubmitConfig) -> str:
    site = cfg.site
    assert site.mode == "submit"
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
        f"source {cfg.env_script}",
        f"cd {cfg.bundle_root}",
        "",
    ]

    spawn_args = [
        "chemgraph", "academy", "spawn-site", "--",
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
    # Shell-safe: the PBS script is a single-quoted bash file, no
    # interpolation, so we can simply join. Don't break the bash
    # tokenizer with values containing spaces, etc -- not expected here.
    pbs_lines.append(" ".join(spawn_args))
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
        # scp the script over. ControlMaster keeps this fast.
        subprocess.run(
            ["scp", local_path, f"{cfg.login_host}:{remote_path}"],
            check=True,
            capture_output=True,
            text=True,
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
    timeout_s: float = 1800.0,
    poll_interval_s: float = 15.0,
) -> None:
    """Block until ``job_id`` transitions to R. Raises TimeoutError otherwise.

    Note: queue waits can be long; default 30 min is a placeholder.
    Operator can bump via --queue-timeout-s (phase 3).
    """
    deadline = time.monotonic() + timeout_s
    last_state: str | None = None
    while time.monotonic() < deadline:
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
        if state in ("F", "X", "E") and state == "F":
            # Finished without ever going to R -- treat as failure.
            raise RuntimeError(
                f"submit[{cfg.site.name}]: job {job_id} finished without running",
            )
        await asyncio.sleep(poll_interval_s)
    raise TimeoutError(
        f"submit[{cfg.site.name}]: job {job_id} did not reach R within {timeout_s:.0f}s",
    )


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
        timeout_s: float,
    ) -> set[str]:
        if not self.job_id:
            raise RuntimeError("submit backend: start() must precede wait_ready()")
        # Split the timeout between queue-wait and post-R registration
        # wait. 80/20 is arbitrary but reasonable -- queues are the
        # long pole.
        queue_budget = max(60.0, timeout_s * 0.8)
        registration_budget = max(60.0, timeout_s - queue_budget)
        await wait_running(self.cfg, self.job_id, timeout_s=queue_budget)
        # Once R, reuse attach-mode's placement.json polling. Build an
        # AttachConfig-shaped view of the SubmitConfig so we don't
        # re-implement the polling loop. AttachConfig's ``venv_activate``
        # is what we'd source on the bash side -- in submit-mode that
        # role is filled by the PBS script's own ``source``, so this
        # value isn't actually used by wait_ready (only the run_dir
        # and login_host matter for the read-only probes). Pass the
        # SubmitConfig.env_script forward to keep the structural
        # shape consistent.
        as_attach = AttachConfig(
            site=self.cfg.site,
            run_id=self.cfg.run_id,
            campaign=self.cfg.campaign,
            bundle_root=self.cfg.bundle_root,
            venv_activate=self.cfg.env_script,
            run_dir=self.cfg.run_dir,
            login_host=self.cfg.login_host,
            exchange_type=self.cfg.exchange_type,
            http_exchange_url=self.cfg.http_exchange_url,
        )
        return await _placement_wait_ready(
            as_attach,
            local_run_dir=local_run_dir,
            timeout_s=registration_budget,
            # submit-mode writes spawn-site stdout/stderr to the PBS
            # job's -o destination (<site>.pbs.log), not attach.log.
            # Without this the error message would point at a
            # nonexistent file and claim "(no log written)" even
            # though the real log exists under a different name.
            diagnostic_log_name=f"{self.cfg.site.name}.pbs.log",
        )

    async def stop(self, *, force: bool = False) -> None:
        if self.job_id:
            await asyncio.to_thread(qdel_job, self.cfg, self.job_id)


if __name__ == "__main__":  # ponytail: command-rendering self-check
    cfg = SubmitConfig(
        site=SiteSpec(
            name="aurora",
            mode="submit",
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
