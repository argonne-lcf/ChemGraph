"""Phase-1 unit tests for the attach-mode remote launcher.

Pure-stdlib site_spec is tested directly; attach_backend's command
rendering is tested without any ssh.
"""

from __future__ import annotations

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
        env_script="/flare/ChemGraph/jinchu/ChemGraph/env.aurora.sh",
        run_dir="/flare/ChemGraph/jinchu/runs/run-008",
        http_exchange_url="https://exchange.academy-agents.org/v1",
    )
    cmd = _build_remote_command(cfg)
    # Sources env, cds, execs the right CLI with the right args.
    assert "env.aurora.sh" in cmd
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
        env_script="/flare/cg/env.aurora.sh",
        run_dir="/flare/runs/r",
        http_exchange_url=None,
    )
    cmd = _build_remote_command(cfg)
    assert "--http-exchange-url" not in cmd
