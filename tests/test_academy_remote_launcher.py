"""Phase-1 unit tests for the attach-mode remote launcher.

Pure-stdlib site_spec is tested directly; attach_backend's command
rendering is tested without any ssh.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# The site_spec module is stdlib-only and can import even without the
# academy extra. The attach_backend module imports ssh_transport (also
# stdlib) and site_spec. Both should import cleanly.
from chemgraph.academy.runtime.remote.site_spec import SiteSpec, parse_site


def test_parse_site_attach_basic() -> None:
    s = parse_site("aurora:attach=x4505;agents=alpha")
    assert s.name == "aurora"
    assert s.mode == "attach"
    assert s.compute_host == "x4505"
    assert s.agents == ("alpha",)


def test_parse_site_attach_multiple_agents_csv() -> None:
    s = parse_site("aurora:agents=alpha,beta,gamma;attach=x4505")
    assert s.agents == ("alpha", "beta", "gamma")
    assert s.compute_host == "x4505"


def test_parse_site_order_independent() -> None:
    a = parse_site("crux:attach=h1;agents=worker-a,worker-b")
    b = parse_site("crux:agents=worker-a,worker-b;attach=h1")
    assert a == b


@pytest.mark.parametrize(
    "bad",
    [
        "noseparator",
        "aurora:",
        "aurora:agents=",
        "aurora:attach=x",
        ":agents=a;attach=x",
        "aurora:attach=x;queue=debug;walltime=01:00:00;agents=a",
    ],
)
def test_parse_site_rejects_bad_input(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_site(bad)


def test_parse_site_submit_mode_basic() -> None:
    s = parse_site("aurora:queue=debug;walltime=01:00:00;agents=alpha")
    assert s.mode == "submit"
    assert s.queue == "debug"
    assert s.walltime == "01:00:00"
    assert s.nodes == 1
    assert s.project is None
    assert s.filesystems is None


def test_parse_site_submit_mode_all_keys() -> None:
    s = parse_site(
        "aurora:queue=prod;walltime=08:00:00;nodes=16;"
        "project=ChemGraph;filesystems=home:flare;agents=alpha,beta"
    )
    assert s.mode == "submit"
    assert s.nodes == 16
    assert s.project == "ChemGraph"
    assert s.filesystems == "home:flare"
    assert s.agents == ("alpha", "beta")


@pytest.mark.parametrize(
    "bad",
    [
        "aurora:queue=debug;agents=a",  # walltime missing
        "aurora:walltime=01:00:00;agents=a",  # queue missing
        "aurora:queue=q;walltime=1;nodes=notanint;agents=a",
    ],
)
def test_parse_site_submit_mode_rejects_incomplete(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_site(bad)


def test_submit_backend_renders_pbs_script() -> None:
    from chemgraph.academy.runtime.remote.submit_backend import (
        SubmitConfig,
        render_pbs_script,
    )

    cfg = SubmitConfig(
        site=SiteSpec(
            name="aurora",
            mode="submit",
            agents=("alpha", "beta"),
            queue="debug",
            walltime="01:00:00",
            nodes=2,
            project="MYPROJ",
            filesystems="home:flare",
        ),
        run_id="run-008",
        campaign="federated-chat",
        login_host="user@aurora.alcf.anl.gov",
        bundle_root="/flare/MYPROJ/u/ChemGraph",
        env_script="/flare/MYPROJ/u/ChemGraph/env.aurora.sh",
        run_dir="/flare/MYPROJ/u/runs/run-008",
        http_exchange_url="https://exchange.academy-agents.org/v1",
    )
    text = render_pbs_script(cfg)
    assert text.startswith("#!/bin/bash")
    assert "#PBS -A MYPROJ" in text
    assert "#PBS -q debug" in text
    assert "#PBS -l select=2,walltime=01:00:00" in text
    assert "#PBS -l filesystems=home:flare" in text
    assert "source /flare/MYPROJ/u/ChemGraph/env.aurora.sh" in text
    assert "chemgraph academy spawn-site" in text
    assert "--agents alpha,beta" in text
    assert "--exchange-type http" in text
    assert "exchange.academy-agents.org" in text


def test_submit_backend_omits_filesystems_when_unset() -> None:
    from chemgraph.academy.runtime.remote.submit_backend import (
        SubmitConfig,
        render_pbs_script,
    )

    cfg = SubmitConfig(
        site=SiteSpec(
            name="crux",
            mode="submit",
            agents=("alpha",),
            queue="debug",
            walltime="00:30:00",
            project="MYPROJ",
        ),
        run_id="r",
        campaign="federated-chat",
        login_host="user@crux.alcf.anl.gov",
        bundle_root="/lus/cg",
        env_script="/lus/cg/env.crux.sh",
        run_dir="/lus/runs/r",
    )
    text = render_pbs_script(cfg)
    assert "filesystems=" not in text


def test_submit_backend_requires_project() -> None:
    from chemgraph.academy.runtime.remote.submit_backend import (
        SubmitConfig,
        render_pbs_script,
    )

    cfg = SubmitConfig(
        site=SiteSpec(
            name="aurora",
            mode="submit",
            agents=("alpha",),
            queue="debug",
            walltime="01:00:00",
        ),
        run_id="r",
        campaign="federated-chat",
        login_host="u@aurora",
        bundle_root="/flare/cg",
        env_script="/flare/cg/env.aurora.sh",
        run_dir="/flare/runs/r",
    )
    with pytest.raises(ValueError, match="project"):
        render_pbs_script(cfg)


def test_submit_backend_persists_job_id_on_disk(tmp_path, monkeypatch) -> None:
    """Regression guard: submit_job writes the job_id to
    <local_run_dir>/<site>.job_id BEFORE returning, so a Ctrl-C
    arriving immediately after qsub (but before the launcher records
    self.job_id) doesn't strand a PBS job with no traceable id."""
    import subprocess as _sp
    from unittest.mock import patch

    from chemgraph.academy.runtime.remote.submit_backend import (
        SubmitConfig,
        submit_job,
    )

    cfg = SubmitConfig(
        site=SiteSpec(
            name="aurora",
            mode="submit",
            agents=("alpha",),
            queue="debug",
            walltime="01:00:00",
            project="MYPROJ",
        ),
        run_id="run-008",
        campaign="federated-chat",
        login_host="u@aurora",
        bundle_root="/flare/cg",
        env_script="/flare/cg/env.aurora.sh",
        run_dir="/flare/runs/run-008",
    )

    class _FakeCompleted:
        def __init__(self, stdout): self.stdout = stdout

    def fake_run(argv, **kw):
        if argv[0] == "scp":
            return _FakeCompleted("")
        # ssh_transport -> ssh ... qsub
        return _FakeCompleted("12345.aurora-pbs\n")

    monkeypatch.setattr(_sp, "run", fake_run)

    job_id = submit_job(cfg, local_run_dir=tmp_path)
    assert job_id == "12345.aurora-pbs"
    persisted = tmp_path / "aurora.job_id"
    assert persisted.exists(), "expected job_id to be persisted before return"
    assert persisted.read_text().strip() == "12345.aurora-pbs"


def test_submit_backend_wait_ready_constructs_attach_view_with_current_field_names() -> None:
    """Regression guard: SubmitSiteBackend.wait_ready() constructs
    an AttachConfig from its SubmitConfig to reuse attach-mode's
    placement.json polling. The field names need to track AttachConfig
    -- this was missed during the env_script -> venv_activate
    rename and only surfaced at runtime (TypeError after a real
    qsub had already landed)."""
    from unittest.mock import patch
    from chemgraph.academy.runtime.remote.submit_backend import (
        SubmitConfig,
        SubmitSiteBackend,
    )

    cfg = SubmitConfig(
        site=SiteSpec(
            name="aurora", mode="submit", agents=("a",),
            queue="debug", walltime="00:15:00", project="P",
        ),
        run_id="r", campaign="federated-chat",
        login_host="u@aurora", bundle_root="/flare/cg",
        env_script="/flare/cg/v/bin/activate",
        run_dir="/flare/runs/r",
    )
    backend = SubmitSiteBackend(cfg)
    backend.job_id = "fake-1"

    # Intercept wait_running so we don't actually poll qstat; bypass
    # straight to the placement.json polling path that builds the
    # AttachConfig.
    from chemgraph.academy.runtime.remote import submit_backend
    async def fake_wait_running(*a, **kw): return None
    async def fake_wait_ready(as_attach, **kw):
        # Touch each field that wait_ready actually reads, so a future
        # AttachConfig rename surfaces here instead of in production.
        _ = as_attach.site
        _ = as_attach.run_dir
        _ = as_attach.login_host
        return {"a"}

    import asyncio
    with patch.object(submit_backend, "wait_running", fake_wait_running), \
         patch.object(submit_backend, "_placement_wait_ready", fake_wait_ready):
        result = asyncio.run(
            backend.wait_ready(local_run_dir=Path("/tmp"), timeout_s=10.0)
        )
    assert result == {"a"}


def test_submit_backend_pbs_script_exports_remote_env() -> None:
    """Regression guard: PBS batch jobs only inherit a tiny default
    env, so system profiles that do ${ALCF_USER}/${ALCF_PROJECT}
    substitution fail to load. The rendered PBS script must export
    forwarded env vars BEFORE `source <env_script>`.

    Symptom without this: spawn-site dies inside the PBS job with
    'unresolved environment variables: ALCF_PROJECT, ALCF_USER, ...'
    -- caught live in submit-smoke-004."""
    from chemgraph.academy.runtime.remote.submit_backend import (
        SubmitConfig,
        render_pbs_script,
    )

    cfg = SubmitConfig(
        site=SiteSpec(
            name="aurora", mode="submit", agents=("a",),
            queue="debug", walltime="01:00:00", project="P",
        ),
        run_id="r", campaign="federated-chat",
        login_host="u@aurora", bundle_root="/flare/cg",
        env_script="/flare/cg/v/bin/activate",
        run_dir="/flare/runs/r",
        remote_env={
            "ALCF_PROJECT": "ChemGraph",
            "ALCF_USER": "jinchu",
            "ARGO_USER": "jinchu.li",
        },
    )
    text = render_pbs_script(cfg)
    assert "export ALCF_PROJECT=ChemGraph" in text
    assert "export ALCF_USER=jinchu" in text
    assert "export ARGO_USER=jinchu.li" in text
    # Order matters: exports must come BEFORE source so env_script
    # can reference them (and so spawn-site itself sees them).
    export_idx = text.index("export ALCF_PROJECT=")
    source_idx = text.index("source ")
    assert export_idx < source_idx, "exports must precede source <env_script>"


def test_submit_backend_per_site_project_overrides_global() -> None:
    """site.project (from --site flag) wins over SubmitConfig.project
    (from --project CLI). Lets a multi-site invocation use one global
    project for most sites and override per-site."""
    from chemgraph.academy.runtime.remote.submit_backend import (
        SubmitConfig,
        render_pbs_script,
    )

    cfg = SubmitConfig(
        site=SiteSpec(
            name="aurora",
            mode="submit",
            agents=("alpha",),
            queue="debug",
            walltime="01:00:00",
            project="SITE_PROJ",
        ),
        run_id="r",
        campaign="federated-chat",
        login_host="u@aurora",
        bundle_root="/flare/cg",
        env_script="/flare/cg/env.aurora.sh",
        run_dir="/flare/runs/r",
        project="GLOBAL_PROJ",
    )
    text = render_pbs_script(cfg)
    assert "#PBS -A SITE_PROJ" in text
    assert "GLOBAL_PROJ" not in text


# ---------------------------------------------------------------------------
# Phase 3: multi-site orchestration
# ---------------------------------------------------------------------------


import asyncio
import os


class _FakeBackend:
    """Test double implementing the SiteBackend Protocol."""

    def __init__(
        self,
        site_name: str,
        *,
        start_raises: BaseException | None = None,
        ready_raises: BaseException | None = None,
        ready_agents: set[str] | None = None,
        ready_delay_s: float = 0.0,
    ) -> None:
        self.site_name = site_name
        self.start_called = False
        self.stop_called = False
        self.stop_force = False
        self._start_raises = start_raises
        self._ready_raises = ready_raises
        self._ready_agents = ready_agents or {f"{site_name}-agent"}
        self._ready_delay_s = ready_delay_s

    async def start(self) -> None:
        self.start_called = True
        if self._start_raises is not None:
            raise self._start_raises

    async def wait_ready(self, *, local_run_dir, timeout_s):  # type: ignore[no-untyped-def]
        if self._ready_delay_s:
            await asyncio.sleep(self._ready_delay_s)
        if self._ready_raises is not None:
            raise self._ready_raises
        return set(self._ready_agents)

    async def stop(self, *, force: bool = False) -> None:
        self.stop_called = True
        self.stop_force = force


def _ns(**kw):
    import argparse
    defaults = dict(
        run_id="r1",
        campaign="federated-chat",
        site=[],
        bundle_root="/flare/cg",
        venv_activate=None,
        run_dir="/tmp/r1",
        local_run_dir="/tmp/r1",
        exchange_type="http",
        http_exchange_url=None,
        project=None,
        ready_timeout_s=5.0,
        auto_bootstrap=False,
        bootstrap_recipient=None,
        spawn_arg=[],
        skip_preflight=True,  # tests don't care about operator env
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_launch_all_sites_ready_returns_zero(monkeypatch):
    from chemgraph.academy.runtime.remote import remote_launcher

    backends = [_FakeBackend("aurora"), _FakeBackend("crux")]
    pairs = [
        (SiteSpec(name="aurora", mode="attach", agents=("a",), compute_host="h"), backends[0]),
        (SiteSpec(name="crux",   mode="attach", agents=("b",), compute_host="h"), backends[1]),
    ]
    monkeypatch.setattr(remote_launcher, "build_backends", lambda args: pairs)

    args = _ns(site=["aurora:attach=h;agents=a", "crux:attach=h;agents=b"])
    rc = asyncio.run(remote_launcher._launch(args))
    assert rc == 0
    assert all(b.start_called for b in backends)
    assert not any(b.stop_called for b in backends)


def test_launch_wait_ready_failure_stops_all_siblings(monkeypatch):
    from chemgraph.academy.runtime.remote import remote_launcher

    good = _FakeBackend("aurora", ready_delay_s=0.05)
    bad = _FakeBackend("crux", ready_raises=TimeoutError("no register"))
    pairs = [
        (SiteSpec(name="aurora", mode="attach", agents=("a",), compute_host="h"), good),
        (SiteSpec(name="crux",   mode="attach", agents=("b",), compute_host="h"), bad),
    ]
    monkeypatch.setattr(remote_launcher, "build_backends", lambda args: pairs)

    args = _ns(site=["aurora:attach=h;agents=a", "crux:attach=h;agents=b"])
    rc = asyncio.run(remote_launcher._launch(args))
    assert rc == 1
    # Both backends should have stop() called even though only one failed.
    assert good.stop_called
    assert bad.stop_called


def test_launch_start_failure_short_circuits_wait_ready(monkeypatch):
    from chemgraph.academy.runtime.remote import remote_launcher

    bad = _FakeBackend("aurora", start_raises=RuntimeError("scp failed"))
    good = _FakeBackend("crux")
    pairs = [
        (SiteSpec(name="aurora", mode="submit", agents=("a",), queue="q", walltime="01:00:00"), bad),
        (SiteSpec(name="crux",   mode="attach", agents=("b",), compute_host="h"), good),
    ]
    monkeypatch.setattr(remote_launcher, "build_backends", lambda args: pairs)

    args = _ns(site=["aurora:queue=q;walltime=01:00:00;agents=a", "crux:attach=h;agents=b"])
    rc = asyncio.run(remote_launcher._launch(args))
    assert rc == 1
    # Sibling should NOT have wait_ready called; phase 3's gather-start
    # short-circuits to teardown before wait_ready runs.
    assert bad.stop_called and good.stop_called


def test_launch_auto_bootstrap_runs_when_ready(monkeypatch):
    from chemgraph.academy.runtime.remote import remote_launcher

    backend = _FakeBackend("aurora")
    pairs = [
        (SiteSpec(name="aurora", mode="attach", agents=("a",), compute_host="h"), backend),
    ]
    monkeypatch.setattr(remote_launcher, "build_backends", lambda args: pairs)

    captured: list[list[str]] = []
    def fake_run_bootstrap(args):
        # Match the signature of _run_bootstrap. Capture for assertion.
        captured.append([
            "--campaign", args.campaign,
            "--run-id", args.run_id,
            "--exchange-type", args.exchange_type,
        ])
        return 0
    monkeypatch.setattr(remote_launcher, "_run_bootstrap", fake_run_bootstrap)

    args = _ns(
        site=["aurora:attach=h;agents=a"],
        auto_bootstrap=True,
    )
    rc = asyncio.run(remote_launcher._launch(args))
    assert rc == 0
    assert captured == [["--campaign", "federated-chat", "--run-id", "r1", "--exchange-type", "http"]]


def test_launch_auto_bootstrap_skipped_without_flag(monkeypatch):
    from chemgraph.academy.runtime.remote import remote_launcher

    backend = _FakeBackend("aurora")
    pairs = [
        (SiteSpec(name="aurora", mode="attach", agents=("a",), compute_host="h"), backend),
    ]
    monkeypatch.setattr(remote_launcher, "build_backends", lambda args: pairs)
    called = []
    monkeypatch.setattr(remote_launcher, "_run_bootstrap", lambda args: called.append(1) or 0)

    args = _ns(site=["aurora:attach=h;agents=a"], auto_bootstrap=False)
    rc = asyncio.run(remote_launcher._launch(args))
    assert rc == 0
    assert called == []


def test_launch_auto_bootstrap_propagates_recipient_override(monkeypatch):
    from chemgraph.academy.runtime.remote import remote_launcher

    backend = _FakeBackend("aurora")
    pairs = [
        (SiteSpec(name="aurora", mode="attach", agents=("a",), compute_host="h"), backend),
    ]
    monkeypatch.setattr(remote_launcher, "build_backends", lambda args: pairs)

    captured_argv: list[list[str]] = []
    def fake_bootstrap_main(argv):
        captured_argv.append(list(argv))
        return 0
    # Patch bootstrap.main where _run_bootstrap imports it from.
    import chemgraph.academy.runtime.bootstrap as bs
    monkeypatch.setattr(bs, "main", fake_bootstrap_main)

    args = _ns(
        site=["aurora:attach=h;agents=a"],
        auto_bootstrap=True,
        bootstrap_recipient="custom-receiver",
    )
    rc = asyncio.run(remote_launcher._launch(args))
    assert rc == 0
    assert captured_argv
    assert "--recipient" in captured_argv[0]
    assert "custom-receiver" in captured_argv[0]


def test_launch_bootstrap_failure_keeps_agents_running(monkeypatch):
    """If bootstrap fails, the launcher returns nonzero but does NOT
    tear down the agents -- operator can re-run bootstrap manually."""
    from chemgraph.academy.runtime.remote import remote_launcher

    backend = _FakeBackend("aurora")
    pairs = [
        (SiteSpec(name="aurora", mode="attach", agents=("a",), compute_host="h"), backend),
    ]
    monkeypatch.setattr(remote_launcher, "build_backends", lambda args: pairs)
    monkeypatch.setattr(remote_launcher, "_run_bootstrap", lambda args: 7)

    args = _ns(site=["aurora:attach=h;agents=a"], auto_bootstrap=True)
    rc = asyncio.run(remote_launcher._launch(args))
    assert rc == 7
    assert not backend.stop_called  # agents intentionally left running


def test_parse_args_rejects_missing_site():
    from chemgraph.academy.runtime.remote import remote_launcher

    with pytest.raises(SystemExit):
        remote_launcher.parse_args([
            "--run-id", "r1",
            "--campaign", "federated-chat",
            "--bundle-root", "/flare/cg",
        ])


def test_parse_args_accepts_multiple_sites():
    from chemgraph.academy.runtime.remote import remote_launcher

    args = remote_launcher.parse_args([
        "--run-id", "r1",
        "--campaign", "federated-chat",
        "--bundle-root", "/flare/cg",
        "--site", "aurora:attach=h1;agents=a",
        "--site", "crux:queue=debug;walltime=01:00:00;agents=b;project=MYPROJ",
        "--auto-bootstrap",
    ])
    assert len(args.site) == 2
    assert args.auto_bootstrap is True


# ---------------------------------------------------------------------------
# bundle_root per-site override (mixed filesystem HPCs)
# ---------------------------------------------------------------------------


def test_parse_site_accepts_bundle_root_override():
    """Operator can override the global --bundle-root per site to
    handle HPCs on different filesystems (e.g. Aurora /flare vs
    Crux /eagle)."""
    s = parse_site("crux:attach=h1;agents=a;bundle_root=/eagle/cg")
    assert s.bundle_root == "/eagle/cg"


def test_parse_site_bundle_root_default_none():
    s = parse_site("aurora:attach=h1;agents=a")
    assert s.bundle_root is None


def test_parse_site_bundle_root_works_in_submit_mode():
    s = parse_site(
        "crux:queue=debug;walltime=01:00:00;agents=a;project=P;bundle_root=/eagle/cg"
    )
    assert s.mode == "submit"
    assert s.bundle_root == "/eagle/cg"


def test_resolve_bundle_root_per_site_wins(monkeypatch):
    from chemgraph.academy.runtime.remote import remote_launcher

    args = remote_launcher.parse_args([
        "--run-id", "r1",
        "--campaign", "federated-chat",
        "--bundle-root", "/flare/cg",  # global default
        "--site", "crux:attach=h;agents=a;bundle_root=/eagle/cg",
    ])
    site = parse_site(args.site[0])
    assert remote_launcher._resolve_bundle_root(site, args) == "/eagle/cg"


def test_resolve_bundle_root_falls_back_to_global(monkeypatch):
    from chemgraph.academy.runtime.remote import remote_launcher

    args = remote_launcher.parse_args([
        "--run-id", "r1",
        "--campaign", "federated-chat",
        "--bundle-root", "/flare/cg",
        "--site", "aurora:attach=h;agents=a",
    ])
    site = parse_site(args.site[0])
    assert remote_launcher._resolve_bundle_root(site, args) == "/flare/cg"


def test_resolve_bundle_root_errors_when_neither_set():
    from chemgraph.academy.runtime.remote import remote_launcher

    args = remote_launcher.parse_args([
        "--run-id", "r1",
        "--campaign", "federated-chat",
        "--site", "aurora:attach=h;agents=a",
    ])
    site = parse_site(args.site[0])
    with pytest.raises(ValueError, match="bundle root"):
        remote_launcher._resolve_bundle_root(site, args)


def test_attach_wait_ready_short_circuits_on_ssh_death(monkeypatch, tmp_path):
    """If the underlying ssh dies before agents register, wait_ready
    should raise RuntimeError immediately (with the attach.log tail)
    rather than waiting out the full timeout."""
    import subprocess as _sp
    from chemgraph.academy.runtime.remote import attach_backend
    from chemgraph.academy.runtime.remote.attach_backend import (
        AttachConfig,
        wait_ready,
    )

    class _DeadProc:
        returncode = 255
        def poll(self): return 255

    cfg = AttachConfig(
        site=SiteSpec(
            name="aurora", mode="attach", agents=("alpha",),
            compute_host="x4505",
        ),
        run_id="r", campaign="federated-chat",
        bundle_root="/flare/cg",
        venv_activate="/flare/cg/venvs/academy-swarm/bin/activate",
        run_dir="/flare/runs/r",
    )

    # ssh_run is what _tail_attach_log calls; return empty.
    monkeypatch.setattr(
        attach_backend,
        "ssh_run",
        lambda *a, **kw: _sp.CompletedProcess(args=[], returncode=0, stdout="", stderr=""),
    )

    async def _go():
        await wait_ready(
            cfg, local_run_dir=tmp_path, timeout_s=5.0,
            poll_interval_s=0.01, proc=_DeadProc(),
        )

    with pytest.raises(RuntimeError, match="ssh exited"):
        asyncio.run(_go())


def test_attach_build_remote_command_redirects_log_early():
    """The remote bash must redirect stdout/stderr to the attach.log
    BEFORE attempting source/cd, so failures in those steps are
    captured. Regression guard: previously the log was opened only
    on the spawn-site exec line, so a missing env_script produced
    zero visible diagnostic."""
    from chemgraph.academy.runtime.remote.attach_backend import (
        AttachConfig,
        _build_remote_command,
    )

    cfg = AttachConfig(
        site=SiteSpec(
            name="aurora", mode="attach", agents=("alpha",),
            compute_host="x4505",
        ),
        run_id="r", campaign="federated-chat",
        bundle_root="/flare/cg",
        venv_activate="/flare/cg/venvs/academy-swarm/bin/activate",
        run_dir="/flare/runs/r",
    )
    cmd = _build_remote_command(cfg)
    # The `exec >> ...attach.log` redirect must appear before
    # `source` and `cd` in the script body so they get captured.
    log_redirect_idx = cmd.index("aurora.attach.log")
    source_idx = cmd.index("source")
    cd_idx = cmd.index("cd ")
    assert log_redirect_idx < source_idx, "log opened after source -- pre-source failures invisible"
    assert log_redirect_idx < cd_idx, "log opened after cd -- pre-cd failures invisible"


def test_launch_propagates_bundle_root_error_as_exit_2(monkeypatch):
    """If a site has no bundle_root resolution, build_backends raises
    ValueError which the launcher converts to exit code 2."""
    from chemgraph.academy.runtime.remote import remote_launcher

    # No global --bundle-root, no per-site bundle_root=.
    monkeypatch.setattr(
        remote_launcher,
        "build_backends",
        lambda args: (_ for _ in ()).throw(ValueError("bundle root missing")),
    )
    args = _ns(site=["crux:attach=h;agents=a"], bundle_root=None)
    rc = asyncio.run(remote_launcher._launch(args))
    assert rc == 2


def test_attach_backend_renders_spawn_site_command() -> None:
    from chemgraph.academy.runtime.remote.attach_backend import (
        AttachConfig,
        _build_remote_command,
    )

    cfg = AttachConfig(
        site=SiteSpec(
            name="aurora",
            mode="attach",
            agents=("alpha", "beta"),
            compute_host="x4505c5s0b0n0",
        ),
        run_id="run-008",
        campaign="federated-chat",
        bundle_root="/flare/ChemGraph/jinchu/ChemGraph",
        venv_activate="/flare/cg/venvs/academy-swarm/bin/activate",
        run_dir="/flare/ChemGraph/jinchu/runs/run-008",
        http_exchange_url="https://exchange.academy-agents.org/v1",
    )
    cmd = _build_remote_command(cfg)
    # Sources the venv activate (puts chemgraph on PATH), cds, execs
    # the right CLI with the right args.
    assert "venvs/academy-swarm/bin/activate" in cmd
    assert "spawn-site" in cmd
    assert "--system aurora" in cmd
    assert "--run-id run-008" in cmd
    assert "--agents alpha,beta" in cmd
    assert "--exchange-type http" in cmd
    assert "exchange.academy-agents.org" in cmd
    # exec is what lets SIGTERM propagate from the ssh-launched bash
    # down to the python process. Regression-guard for accidentally
    # dropping it during a refactor.
    assert "exec " in cmd
    # Log redirection to per-site attach.log so the launcher can tail
    # it after a boot timeout.
    assert "aurora.attach.log" in cmd


def test_attach_backend_nests_through_login_host_when_set() -> None:
    """Compute nodes on ALCF aren't reachable from outside; the
    laptop ssh's to the login node, which then ssh's to the compute
    node using its in-cluster hostbased trust."""
    import subprocess as _sp
    from unittest.mock import patch

    from chemgraph.academy.runtime.remote.attach_backend import (
        AttachConfig,
        start,
    )

    cfg = AttachConfig(
        site=SiteSpec(
            name="aurora", mode="attach", agents=("alpha",),
            compute_host="x4610c7s2b0n0",
        ),
        run_id="r", campaign="federated-chat",
        bundle_root="/flare/cg",
        venv_activate="/flare/cg/venvs/academy-swarm/bin/activate",
        run_dir="/flare/runs/r",
        login_host="jinchuli@aurora.alcf.anl.gov",
    )
    captured: list[list[str]] = []

    class _FakePopen:
        def __init__(self, argv, **kw):
            captured.append(list(argv))
        def poll(self): return None
        def terminate(self): pass
        def kill(self): pass

    with patch.object(_sp, "Popen", _FakePopen):
        start(cfg)
    argv = captured[0]
    # Outer ssh goes to the login host with -tt and ServerAlive
    # keepalive flags (anywhere before the host); login host is the
    # last positional arg before the body.
    assert argv[0] == "ssh"
    assert "-tt" in argv
    assert "jinchuli@aurora.alcf.anl.gov" in argv
    assert "ServerAliveInterval=30" in argv
    # Body is the last argv element -- the bash one-liner.
    body = argv[-1]
    assert "ssh -tt" in body
    assert "ServerAliveInterval=30" in body  # inner ssh also keepalives
    assert "x4610c7s2b0n0" in body
    # And the deepest level still has the spawn-site invocation.
    assert "spawn-site" in body


def test_attach_backend_emits_remote_env_exports() -> None:
    """The launcher used to require operators to manually export
    ALCF_USER, ARGO_USER, http_proxy etc inside the qsub -I shell
    before running spawn-site. Now we forward them from the
    operator's laptop shell so the remote bash sees the same
    values. Regression: bare invocation without exports causes
    spawn-site to fail at lm_config rejection of <argo-user>
    placeholder or with no proxy / wrong path components.
    """
    from chemgraph.academy.runtime.remote.attach_backend import (
        AttachConfig,
        _build_remote_command,
    )

    cfg = AttachConfig(
        site=SiteSpec(
            name="aurora", mode="attach", agents=("alpha",),
            compute_host="x4505",
        ),
        run_id="r", campaign="federated-chat",
        bundle_root="/flare/cg",
        venv_activate="/flare/cg/venvs/academy-swarm/bin/activate",
        run_dir="/flare/runs/r",
        remote_env={
            "ARGO_USER": "jinchu.li",
            "ALCF_USER": "jinchu",
            "http_proxy": "http://proxy.alcf.anl.gov:3128",
        },
    )
    cmd = _build_remote_command(cfg)
    assert "export ALCF_USER=jinchu" in cmd
    assert "export ARGO_USER=jinchu.li" in cmd
    assert "export http_proxy=" in cmd
    # Env exports must come BEFORE source activate (so that PATH
    # doesn't shadow them). The whole script is wrapped as a
    # quoted bash -lc payload so textual index() ordering only
    # works between events inside that same payload.
    export_idx = cmd.index("export ALCF_USER=")
    activate_idx = cmd.index("activate")
    assert export_idx < activate_idx, "exports should run before source activate"


def test_parse_site_accepts_pbs_jobid_override() -> None:
    """Operator can supply PBS_JOBID from inside their qsub -I
    shell so the launcher can reconstruct PBS_NODEFILE remotely.
    Nested ssh strips PBS env, which would otherwise make mpiexec
    inside spawn-site refuse multi-node placements."""
    s = parse_site("aurora:attach=x4505;agents=a;pbs_jobid=8568282.aurora")
    assert s.pbs_jobid == "8568282.aurora"


def test_attach_backend_exports_pbs_jobid_and_nodefile_when_provided() -> None:
    """When pbs_jobid is set, the remote bash must export both
    PBS_JOBID and the deterministic PBS_NODEFILE path before
    invoking spawn-site -- otherwise mpiexec can't see the
    multi-node allocation."""
    from chemgraph.academy.runtime.remote.attach_backend import (
        AttachConfig,
        _build_remote_command,
    )

    cfg = AttachConfig(
        site=SiteSpec(
            name="aurora", mode="attach", agents=("a",),
            compute_host="x4505",
            pbs_jobid="8568282.aurora-pbs",
        ),
        run_id="r", campaign="federated-chat",
        bundle_root="/flare/cg",
        venv_activate="/flare/cg/venvs/academy-swarm/bin/activate",
        run_dir="/flare/runs/r",
    )
    cmd = _build_remote_command(cfg)
    assert "export PBS_JOBID=8568282.aurora-pbs" in cmd
    # ssh_quote may wrap the path; assert both PBS_NODEFILE name + path appear
    assert "PBS_NODEFILE=" in cmd
    assert "/var/spool/pbs/aux/8568282.aurora-pbs" in cmd


def test_attach_backend_skips_pbs_exports_when_jobid_unset() -> None:
    """No pbs_jobid -> no PBS_* exports at all. Single-node attach
    runs don't need them; only multi-node runs that depend on
    mpiexec reading the nodefile."""
    from chemgraph.academy.runtime.remote.attach_backend import (
        AttachConfig,
        _build_remote_command,
    )

    cfg = AttachConfig(
        site=SiteSpec(
            name="aurora", mode="attach", agents=("a",),
            compute_host="x4505",
            pbs_jobid=None,
        ),
        run_id="r", campaign="federated-chat",
        bundle_root="/flare/cg",
        venv_activate="/flare/cg/venvs/academy-swarm/bin/activate",
        run_dir="/flare/runs/r",
    )
    cmd = _build_remote_command(cfg)
    assert "PBS_JOBID" not in cmd
    assert "PBS_NODEFILE" not in cmd


def test_attach_backend_appends_extra_spawn_args() -> None:
    """The launcher's --spawn-arg passthrough lets the operator pass
    spawn-site flags the launcher doesn't have dedicated --flag
    coverage for (e.g. --agents-per-node, --max-decisions).
    Repeatable + appended verbatim, preserving argv order."""
    from chemgraph.academy.runtime.remote.attach_backend import (
        AttachConfig,
        _build_remote_command,
    )

    cfg = AttachConfig(
        site=SiteSpec(
            name="aurora", mode="attach", agents=("alpha",),
            compute_host="x4505",
        ),
        run_id="r", campaign="federated-chat",
        bundle_root="/flare/cg",
        venv_activate="/flare/cg/venvs/academy-swarm/bin/activate",
        run_dir="/flare/runs/r",
        extra_args=("--agents-per-node", "2", "--max-decisions", "100"),
    )
    cmd = _build_remote_command(cfg)
    assert "--agents-per-node 2" in cmd
    assert "--max-decisions 100" in cmd
    # They must appear AFTER --exchange-type http (i.e. at the tail of
    # the spawn-site argv, not before) so the operator's overrides
    # win against the launcher's defaults.
    ex_idx = cmd.index("--exchange-type")
    apn_idx = cmd.index("--agents-per-node")
    assert ex_idx < apn_idx


def test_attach_backend_empty_remote_env_renders_no_export_block() -> None:
    """When the operator forgot to set env vars, we should not
    emit ``export = ;`` which is a syntax error. Empty remote_env
    is harmless (the operator just gets the usual failure from
    spawn-site about missing ARGO_USER)."""
    from chemgraph.academy.runtime.remote.attach_backend import (
        AttachConfig,
        _build_remote_command,
    )

    cfg = AttachConfig(
        site=SiteSpec(
            name="aurora", mode="attach", agents=("alpha",),
            compute_host="x4505",
        ),
        run_id="r", campaign="federated-chat",
        bundle_root="/flare/cg",
        venv_activate="/flare/cg/venvs/academy-swarm/bin/activate",
        run_dir="/flare/runs/r",
        remote_env={},
    )
    cmd = _build_remote_command(cfg)
    assert "export =" not in cmd  # no malformed export


def test_collect_remote_env_picks_up_alcf_vars(monkeypatch):
    """The launcher reads forwarded env vars from os.environ at
    launch time. Confirms the right names are in the list and
    empty values are filtered out."""
    from chemgraph.academy.runtime.remote import remote_launcher

    for k in remote_launcher._FORWARDED_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ALCF_USER", "jinchu")
    monkeypatch.setenv("ALCF_SSH_USER", "jinchuli")
    monkeypatch.setenv("ARGO_USER", "jinchu.li")
    monkeypatch.setenv("ALCF_PROJECT", "")  # empty -- must be dropped

    env = remote_launcher._collect_remote_env(exchange_type="http")
    assert env["ALCF_USER"] == "jinchu"
    assert env["ALCF_SSH_USER"] == "jinchuli"
    assert env["ARGO_USER"] == "jinchu.li"
    assert "ALCF_PROJECT" not in env  # empty was filtered


def test_collect_remote_env_injects_alcf_proxy_for_http(monkeypatch):
    """When exchange_type=http, the launcher injects the ALCF HTTP
    proxy + no_proxy list so compute ranks can reach
    exchange.academy-agents.org. Required because the operator's
    laptop usually doesn't have http_proxy set, so env forwarding
    alone leaves the remote shell with nothing to pass to mpiexec's
    --genv."""
    from chemgraph.academy.runtime.remote import remote_launcher

    for k in (*remote_launcher._FORWARDED_ENV_VARS,):
        monkeypatch.delenv(k, raising=False)

    env = remote_launcher._collect_remote_env(exchange_type="http")
    assert env["http_proxy"] == remote_launcher._ALCF_HTTP_PROXY
    assert env["HTTP_PROXY"] == remote_launcher._ALCF_HTTP_PROXY
    assert env["https_proxy"] == remote_launcher._ALCF_HTTP_PROXY
    # NO_PROXY intentionally NOT injected -- see _collect_remote_env
    # comment about shell-quote escaping across the ssh chain.
    assert "no_proxy" not in env
    assert "NO_PROXY" not in env


def test_collect_remote_env_skips_proxy_injection_for_redis(monkeypatch):
    """Non-http exchange types (redis, local, hybrid) don't need the
    proxy because compute ranks talk to a local Redis or in-process
    exchange. Skip the injection to avoid masking operator-set
    proxy values they intentionally turned off."""
    from chemgraph.academy.runtime.remote import remote_launcher

    for k in (*remote_launcher._FORWARDED_ENV_VARS,):
        monkeypatch.delenv(k, raising=False)

    env = remote_launcher._collect_remote_env(exchange_type="redis")
    assert "http_proxy" not in env
    assert "HTTP_PROXY" not in env


def test_ansi_helpers_noop_when_color_disabled(monkeypatch):
    """When stderr isn't a TTY OR NO_COLOR is set, ANSI helpers
    pass text through unchanged. Otherwise piping launcher output
    to a log file dumps escape codes everywhere."""
    from chemgraph.academy.runtime.remote import remote_launcher
    monkeypatch.setattr(remote_launcher, "_USE_COLOR", False)
    assert remote_launcher._green("ok") == "ok"
    assert remote_launcher._red("fail") == "fail"
    assert remote_launcher._bold("name") == "name"


def test_ansi_helpers_emit_codes_when_enabled(monkeypatch):
    from chemgraph.academy.runtime.remote import remote_launcher
    monkeypatch.setattr(remote_launcher, "_USE_COLOR", True)
    out = remote_launcher._green("ok")
    assert out.startswith("\033[")
    assert "ok" in out
    assert out.endswith("\033[0m")


def test_preflight_env_passes_when_all_required_set(monkeypatch):
    from chemgraph.academy.runtime.remote import remote_launcher
    import io

    for k in remote_launcher._FORWARDED_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ALCF_PROJECT", "ChemGraph")
    monkeypatch.setenv("ALCF_USER", "jinchu")
    monkeypatch.setenv("ARGO_USER", "jinchu.li")

    errs = remote_launcher._preflight_env(stderr=io.StringIO())
    assert errs == []


def test_preflight_env_reports_each_missing_required(monkeypatch):
    from chemgraph.academy.runtime.remote import remote_launcher
    import io

    for k in remote_launcher._FORWARDED_ENV_VARS:
        monkeypatch.delenv(k, raising=False)

    errs = remote_launcher._preflight_env(stderr=io.StringIO())
    # Three required vars: ALCF_PROJECT, ALCF_USER, ARGO_USER.
    assert len(errs) == 3
    joined = "\n".join(errs)
    assert "ALCF_PROJECT" in joined
    assert "ALCF_USER" in joined
    assert "ARGO_USER" in joined
    # Optional ALCF_SSH_USER is not in the error list.
    assert all("ALCF_SSH_USER" not in e for e in errs)


def test_preflight_env_optional_ssh_user_not_required(monkeypatch):
    from chemgraph.academy.runtime.remote import remote_launcher
    import io

    for k in remote_launcher._FORWARDED_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ALCF_PROJECT", "ChemGraph")
    monkeypatch.setenv("ALCF_USER", "jinchu")
    monkeypatch.setenv("ARGO_USER", "jinchu.li")
    # ALCF_SSH_USER intentionally NOT set -- should be fine.

    errs = remote_launcher._preflight_env(stderr=io.StringIO())
    assert errs == []


def test_preflight_error_messages_include_example(monkeypatch):
    """Error message should show the operator HOW to fix it, not
    just that it's missing. Regression guard: 'ARGO_USER not set'
    is useless; 'export ARGO_USER=jinchu.li' is actionable."""
    from chemgraph.academy.runtime.remote import remote_launcher
    import io

    for k in remote_launcher._FORWARDED_ENV_VARS:
        monkeypatch.delenv(k, raising=False)

    errs = remote_launcher._preflight_env(stderr=io.StringIO())
    for msg in errs:
        assert "export " in msg, f"missing fix-it suggestion: {msg}"


def test_launch_exits_2_on_preflight_failure(monkeypatch):
    """When preflight finds missing required env vars and the
    operator hasn't passed --skip-preflight, _launch returns 2
    BEFORE attempting any ssh or backend construction."""
    from chemgraph.academy.runtime.remote import remote_launcher

    for k in remote_launcher._FORWARDED_ENV_VARS:
        monkeypatch.delenv(k, raising=False)

    # build_backends should NOT be reached.
    called = []
    monkeypatch.setattr(
        remote_launcher,
        "build_backends",
        lambda args: called.append(1) or [],
    )

    args = _ns(site=["aurora:attach=h;agents=a"], skip_preflight=False)
    rc = asyncio.run(remote_launcher._launch(args))
    assert rc == 2
    assert called == []  # short-circuited before backend construction


def test_preflight_dashboard_passes_when_relay_file_present(monkeypatch):
    """Dashboard probe sshes the login node and checks profile.relay_host_file.
    Present + non-empty -> ok. Missing -> error pointing operator at
    --system dashboard fix."""
    import io
    from chemgraph.academy.runtime.remote import remote_launcher
    from chemgraph.academy.runtime.remote import ssh_transport
    import subprocess as _sp

    class _FakeProfile:
        remote_host = "u@aurora"
        relay_host_file = "/flare/cg/uan-relay-18186.host"

    monkeypatch.setattr(
        remote_launcher, "load_system_profile",
        lambda name: _FakeProfile(),
    )
    # ssh test+cat returns the relay host -> present.
    monkeypatch.setattr(
        ssh_transport, "ssh_run",
        lambda host, cmd, **kw: _sp.CompletedProcess(
            args=[], returncode=0, stdout="uan-0003\n", stderr=""),
    )

    pairs = [(
        SiteSpec(name="aurora", mode="attach", agents=("a",), compute_host="h"),
        _FakeBackend("aurora"),
    )]
    errs = remote_launcher._preflight_dashboard(pairs, stderr=io.StringIO())
    assert errs == []


def test_preflight_dashboard_fails_when_relay_file_missing(monkeypatch):
    import io
    from chemgraph.academy.runtime.remote import remote_launcher
    from chemgraph.academy.runtime.remote import ssh_transport
    import subprocess as _sp

    class _FakeProfile:
        remote_host = "u@crux"
        relay_host_file = "/eagle/cg/uan-relay-18187-crux.host"

    monkeypatch.setattr(
        remote_launcher, "load_system_profile",
        lambda name: _FakeProfile(),
    )
    monkeypatch.setattr(
        ssh_transport, "ssh_run",
        lambda host, cmd, **kw: _sp.CompletedProcess(
            args=[], returncode=0, stdout="MISSING\n", stderr=""),
    )

    pairs = [(
        SiteSpec(name="crux", mode="attach", agents=("a",), compute_host="h"),
        _FakeBackend("crux"),
    )]
    errs = remote_launcher._preflight_dashboard(pairs, stderr=io.StringIO())
    assert len(errs) == 1
    msg = errs[0]
    assert "crux" in msg
    assert "dashboard" in msg.lower()
    # Error must suggest the fix (--system list including crux).
    assert "--system" in msg
    assert "crux" in msg


def test_preflight_dashboard_lists_all_sites_in_fix_suggestion(monkeypatch):
    """The 'try this command' suggestion should include EVERY site
    in the launch invocation, not just the failing one. Operators
    typically re-run with the full list, not piecemeal."""
    import io
    from chemgraph.academy.runtime.remote import remote_launcher
    from chemgraph.academy.runtime.remote import ssh_transport
    import subprocess as _sp

    class _FakeAurora:
        remote_host = "u@aurora"
        relay_host_file = "/flare/cg/uan-relay-18186.host"
    class _FakeCrux:
        remote_host = "u@crux"
        relay_host_file = "/eagle/cg/uan-relay-18187-crux.host"

    profiles = {"aurora": _FakeAurora(), "crux": _FakeCrux()}
    monkeypatch.setattr(
        remote_launcher, "load_system_profile",
        lambda name: profiles[name],
    )

    # Aurora OK, Crux MISSING.
    def ssh_run(host, cmd, **kw):
        out = "uan-0003\n" if "uan-relay-18186" in cmd else "MISSING\n"
        return _sp.CompletedProcess(args=[], returncode=0, stdout=out, stderr="")
    monkeypatch.setattr(ssh_transport, "ssh_run", ssh_run)

    pairs = [
        (SiteSpec(name="aurora", mode="attach", agents=("a",), compute_host="h"), _FakeBackend("aurora")),
        (SiteSpec(name="crux",   mode="attach", agents=("b",), compute_host="h"), _FakeBackend("crux")),
    ]
    errs = remote_launcher._preflight_dashboard(pairs, stderr=io.StringIO())
    assert len(errs) == 1
    assert "aurora,crux" in errs[0]  # suggestion includes both


def test_preflight_dashboard_warns_but_continues_on_ssh_failure(monkeypatch):
    """If ssh itself fails (timeout, host unreachable) -- don't block;
    the launcher will give a clearer error later when it tries to
    use the unreachable host."""
    import io
    from chemgraph.academy.runtime.remote import remote_launcher
    from chemgraph.academy.runtime.remote import ssh_transport

    class _FakeProfile:
        remote_host = "u@aurora"
        relay_host_file = "/flare/cg/uan-relay-18186.host"
    monkeypatch.setattr(
        remote_launcher, "load_system_profile",
        lambda name: _FakeProfile(),
    )

    def raises(host, cmd, **kw):
        raise OSError("connection refused")
    monkeypatch.setattr(ssh_transport, "ssh_run", raises)

    pairs = [(
        SiteSpec(name="aurora", mode="attach", agents=("a",), compute_host="h"),
        _FakeBackend("aurora"),
    )]
    errs = remote_launcher._preflight_dashboard(pairs, stderr=io.StringIO())
    assert errs == []  # warn, don't fail


def test_launch_skip_preflight_bypasses_check(monkeypatch):
    """--skip-preflight lets the operator proceed even when env
    vars are missing -- escape hatch for the edge case where the
    remote bash inherits env from elsewhere."""
    from chemgraph.academy.runtime.remote import remote_launcher

    for k in remote_launcher._FORWARDED_ENV_VARS:
        monkeypatch.delenv(k, raising=False)

    backend = _FakeBackend("aurora")
    pairs = [
        (SiteSpec(name="aurora", mode="attach", agents=("a",), compute_host="h"), backend),
    ]
    monkeypatch.setattr(remote_launcher, "build_backends", lambda args: pairs)

    args = _ns(site=["aurora:attach=h;agents=a"], skip_preflight=True)
    rc = asyncio.run(remote_launcher._launch(args))
    assert rc == 0


def test_collect_remote_env_operator_proxy_wins(monkeypatch):
    """If the operator has http_proxy set in their laptop shell
    (e.g. corporate network), it must NOT be overwritten by the
    ALCF default. setdefault semantics."""
    from chemgraph.academy.runtime.remote import remote_launcher

    for k in remote_launcher._FORWARDED_ENV_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("http_proxy", "http://corp-proxy:8080")

    env = remote_launcher._collect_remote_env(exchange_type="http")
    assert env["http_proxy"] == "http://corp-proxy:8080"


def test_attach_backend_direct_ssh_when_login_host_empty() -> None:
    """Back-compat: with login_host empty (the default), the launcher
    falls back to a single ssh straight to the compute host. Useful
    for on-prem clusters where compute nodes are directly reachable."""
    import subprocess as _sp
    from unittest.mock import patch

    from chemgraph.academy.runtime.remote.attach_backend import (
        AttachConfig,
        start,
    )

    cfg = AttachConfig(
        site=SiteSpec(
            name="local", mode="attach", agents=("alpha",),
            compute_host="compute01.lab.example",
        ),
        run_id="r", campaign="federated-chat",
        bundle_root="/scratch/cg",
        venv_activate="/flare/cg/venvs/academy-swarm/bin/activate",
        run_dir="/scratch/runs/r",
        login_host="",
    )
    captured: list[list[str]] = []

    class _FakePopen:
        def __init__(self, argv, **kw):
            captured.append(list(argv))
        def poll(self): return None
        def terminate(self): pass
        def kill(self): pass

    with patch.object(_sp, "Popen", _FakePopen):
        start(cfg)
    argv = captured[0]
    # Direct ssh: first two are ssh -tt, target host appears, body is last.
    assert argv[0] == "ssh"
    assert "-tt" in argv
    assert "compute01.lab.example" in argv
    # The body is the remote bash directly; no inner ssh wrapping.
    assert "ssh -tt" not in argv[-1]


def test_attach_backend_uses_tt_for_signal_propagation() -> None:
    """Regression guard: ``ssh -tt`` is required so a Ctrl-C on the
    launcher delivers SIGHUP to the remote python via the SSH
    channel closing. Without -tt, the remote python keeps running
    after local ssh dies (orphan inside the operator's allocation).
    """
    import subprocess as _sp
    from unittest.mock import patch

    from chemgraph.academy.runtime.remote.attach_backend import (
        AttachConfig,
        start,
    )

    cfg = AttachConfig(
        site=SiteSpec(
            name="aurora",
            mode="attach",
            agents=("alpha",),
            compute_host="x4505",
        ),
        run_id="r",
        campaign="federated-chat",
        bundle_root="/flare/cg",
        venv_activate="/flare/cg/venvs/academy-swarm/bin/activate",
        run_dir="/flare/runs/r",
    )
    captured_argv: list[list[str]] = []

    class _FakePopen:
        def __init__(self, argv, **kw):
            captured_argv.append(list(argv))
        def poll(self): return None
        def terminate(self): pass
        def kill(self): pass

    with patch.object(_sp, "Popen", _FakePopen):
        start(cfg)
    assert captured_argv
    argv = captured_argv[0]
    assert argv[0] == "ssh"
    assert "-tt" in argv[:3], f"expected -tt early in ssh argv, got {argv[:5]}"


def test_attach_backend_omits_http_url_when_none() -> None:
    from chemgraph.academy.runtime.remote.attach_backend import (
        AttachConfig,
        _build_remote_command,
    )

    cfg = AttachConfig(
        site=SiteSpec(
            name="aurora",
            mode="attach",
            agents=("alpha",),
            compute_host="x4505",
        ),
        run_id="r",
        campaign="federated-chat",
        bundle_root="/flare/cg",
        venv_activate="/flare/cg/venvs/academy-swarm/bin/activate",
        run_dir="/flare/runs/r",
        http_exchange_url=None,
    )
    cmd = _build_remote_command(cfg)
    assert "--http-exchange-url" not in cmd
