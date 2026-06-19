from __future__ import annotations

from pathlib import Path

import pytest

# Skip when the optional 'academy' extra is absent; the runtime
# subpackage imports academy.* at module level.
pytest.importorskip("academy")

from chemgraph.academy.runtime import compute_launcher
from chemgraph.academy.runtime.compute_launcher import AllocationPlan


def _plan(tmp_path: Path) -> AllocationPlan:
    lm_config = tmp_path / "lm.json"
    campaign = tmp_path / "campaign.jsonc"
    lm_config.write_text("{}\n", encoding="utf-8")
    campaign.write_text("{}\n", encoding="utf-8")
    return AllocationPlan(
        run_dir=tmp_path,
        run_token="token-1",
        agent_count=3,
        agents_per_node=1,
        campaign_config=campaign,
        lm_config=lm_config,
        max_decisions=7,
        poll_timeout_s=2.0,
        idle_timeout_s=600.0,
        startup_timeout_s=120.0,
        completion_timeout_s=60.0,
        status_interval_s=5.0,
        redis_host="redis-host",
        redis_port=6392,
        redis_bind="0.0.0.0",
        redis_protected_mode="no",
        redis_namespace="ns",
        start_redis=False,
        mpiexec="mpiexec",
        chemgraph_repo_root=tmp_path / "ChemGraph",
    )


def test_run_allocation_builds_single_mpiexec_command(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(compute_launcher, "wait_redis", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        compute_launcher.subprocess,
        "call",
        lambda cmd: calls.append(cmd) or 0,
    )

    assert compute_launcher.run_allocation(_plan(tmp_path)) == 0

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:4] == ["mpiexec", "-n", "3", "--ppn"]
    assert "chemgraph.cli.main" in cmd
    assert "mpi-daemon" in cmd
    assert "--campaign-config" in cmd
    assert "--lm-config" in cmd
    assert "--exchange-type" in cmd
    assert "--chemgraph-repo-root" in cmd
    assert (tmp_path / "launch_command.txt").exists()


# ---------------------------------------------------------------------------
# Phase B.1: --exchange-type http + cross-HPC plumbing
# ---------------------------------------------------------------------------


def _plan_http(tmp_path: Path, *, http_exchange_url: str | None = None) -> AllocationPlan:
    base = _plan(tmp_path)
    import dataclasses
    return dataclasses.replace(
        base,
        exchange_type="http",
        http_exchange_url=http_exchange_url,
    )


def test_run_allocation_with_http_exchange_does_not_start_redis(
    tmp_path, monkeypatch,
) -> None:
    """When the exchange doesn't talk to Redis (``http``, ``local``),
    rank 0 must NOT start a redis-server subprocess. Otherwise compute
    nodes without redis-server installed fail at launch, and nodes with
    it pointlessly bind a port we never use."""
    started_subprocess: list[list[str]] = []

    def fake_popen(cmd, **kwargs):  # pragma: no cover - exercised via assert below
        started_subprocess.append(list(cmd))
        raise AssertionError(
            f"Popen should not be called for http exchange; got {cmd!r}",
        )

    monkeypatch.setattr(compute_launcher.subprocess, "Popen", fake_popen)
    # wait_redis is the other Redis-touching site; assert it's not called.
    def boom(*args, **kwargs):
        raise AssertionError("wait_redis should not run for http exchange")
    monkeypatch.setattr(compute_launcher, "wait_redis", boom)
    monkeypatch.setattr(
        compute_launcher.subprocess,
        "call",
        lambda cmd: 0,
    )

    plan = _plan_http(tmp_path)
    # start_redis is True by default; verify the http-exchange code path
    # still skips Redis. This is the "operator forgot --no-start-redis"
    # case, which used to fail loudly on nodes without redis-server.
    import dataclasses
    plan = dataclasses.replace(plan, start_redis=True)
    assert compute_launcher.run_allocation(plan) == 0
    assert started_subprocess == []


def test_run_allocation_forwards_http_exchange_url_when_set(
    tmp_path, monkeypatch,
) -> None:
    """``--http-exchange-url`` (operator override for a self-hosted
    exchange) must flow into the daemon's argv. Otherwise the daemon
    silently falls back to the hosted default."""
    calls: list[list[str]] = []
    monkeypatch.setattr(compute_launcher, "wait_redis", lambda *a, **k: None)
    monkeypatch.setattr(
        compute_launcher.subprocess,
        "call",
        lambda cmd: calls.append(cmd) or 0,
    )

    custom = "https://my-private-exchange.example.com/v1"
    plan = _plan_http(tmp_path, http_exchange_url=custom)
    assert compute_launcher.run_allocation(plan) == 0

    cmd = calls[0]
    assert "--http-exchange-url" in cmd
    assert custom in cmd
    # Sanity: also confirm --exchange-type http rode along.
    type_idx = cmd.index("--exchange-type")
    assert cmd[type_idx + 1] == "http"


def test_run_allocation_omits_http_exchange_url_flag_when_unset(
    tmp_path, monkeypatch,
) -> None:
    """When no override is given, the daemon argv must NOT carry an
    empty ``--http-exchange-url`` (which argparse would happily parse
    as a literal empty-string URL and pass to HttpExchangeFactory)."""
    calls: list[list[str]] = []
    monkeypatch.setattr(compute_launcher, "wait_redis", lambda *a, **k: None)
    monkeypatch.setattr(
        compute_launcher.subprocess,
        "call",
        lambda cmd: calls.append(cmd) or 0,
    )

    plan = _plan_http(tmp_path, http_exchange_url=None)
    assert compute_launcher.run_allocation(plan) == 0

    cmd = calls[0]
    assert "--http-exchange-url" not in cmd


# ---------------------------------------------------------------------------
# Phase B.1: agent subsetting + spawn-site --no-bootstrap forwarding
# ---------------------------------------------------------------------------


def _plan_subset(
    tmp_path: Path,
    *,
    agents: tuple[str, ...],
    skip_bootstrap: bool = True,
) -> AllocationPlan:
    """An AllocationPlan that mimics what ``spawn-site`` would build."""
    import dataclasses
    base = _plan(tmp_path)
    return dataclasses.replace(
        base,
        agent_count=len(agents),  # spawn-site derives count from slice
        agents=agents,
        skip_bootstrap=skip_bootstrap,
    )


def test_run_allocation_forwards_agents_flag_when_slice_given(
    tmp_path, monkeypatch,
) -> None:
    """When ``plan.agents`` is non-empty the daemon must receive
    ``--agents worker-a,worker-b``, otherwise it would launch the
    full campaign on every rank index and the rank-to-agent mapping
    would diverge across sites."""
    calls: list[list[str]] = []
    monkeypatch.setattr(compute_launcher, "wait_redis", lambda *a, **k: None)
    monkeypatch.setattr(
        compute_launcher.subprocess,
        "call",
        lambda cmd: calls.append(cmd) or 0,
    )

    plan = _plan_subset(tmp_path, agents=("worker-a", "worker-b"))
    assert compute_launcher.run_allocation(plan) == 0

    cmd = calls[0]
    assert "--agents" in cmd
    idx = cmd.index("--agents")
    assert cmd[idx + 1] == "worker-a,worker-b"
    # Slice length must drive mpiexec -n so rank ordering matches the
    # daemon's post-filter view of campaign.agents.
    assert cmd[: cmd.index("--ppn") + 2] == ["mpiexec", "-n", "2", "--ppn", "1"]


def test_run_allocation_omits_agents_flag_for_single_machine_runs(
    tmp_path, monkeypatch,
) -> None:
    """The single-machine ``run-compute`` flow leaves ``plan.agents``
    empty so the daemon falls back to its launch-everything default.
    A spurious ``--agents`` flag here would cause subsetting to fail
    closed (``filter_agents`` rejects unknown names)."""
    calls: list[list[str]] = []
    monkeypatch.setattr(compute_launcher, "wait_redis", lambda *a, **k: None)
    monkeypatch.setattr(
        compute_launcher.subprocess,
        "call",
        lambda cmd: calls.append(cmd) or 0,
    )

    assert compute_launcher.run_allocation(_plan(tmp_path)) == 0

    cmd = calls[0]
    assert "--agents" not in cmd


def test_run_allocation_forwards_no_bootstrap_when_requested(
    tmp_path, monkeypatch,
) -> None:
    """``spawn-site`` sets ``plan.skip_bootstrap=True`` because kickoff
    must be deferred until every federated site is up. The launcher
    must propagate this -- otherwise rank 0 dispatches the bootstrap
    locally and the campaign starts before remote agents have
    registered on the exchange."""
    calls: list[list[str]] = []
    monkeypatch.setattr(compute_launcher, "wait_redis", lambda *a, **k: None)
    monkeypatch.setattr(
        compute_launcher.subprocess,
        "call",
        lambda cmd: calls.append(cmd) or 0,
    )

    plan = _plan_subset(tmp_path, agents=("worker-a",), skip_bootstrap=True)
    assert compute_launcher.run_allocation(plan) == 0
    assert "--no-bootstrap" in calls[0]


def test_run_allocation_omits_no_bootstrap_for_single_machine_runs(
    tmp_path, monkeypatch,
) -> None:
    """``run-compute`` keeps its inline bootstrap so the
    single-machine UX doesn't regress -- the flag must be absent
    when ``plan.skip_bootstrap`` is False."""
    calls: list[list[str]] = []
    monkeypatch.setattr(compute_launcher, "wait_redis", lambda *a, **k: None)
    monkeypatch.setattr(
        compute_launcher.subprocess,
        "call",
        lambda cmd: calls.append(cmd) or 0,
    )

    assert compute_launcher.run_allocation(_plan(tmp_path)) == 0
    assert "--no-bootstrap" not in calls[0]


def test_prepare_compute_launch_derives_agent_count_from_agents(
    tmp_path, monkeypatch,
) -> None:
    """When ``--agents worker-a,worker-b`` is given the launcher must
    derive agent_count=2 from the slice length. An operator who also
    passes ``--agent-count`` that disagrees should hit a loud error
    -- silent precedence would let the two values drift, and the
    daemon's MPI -n would not equal its post-filter agent ordering."""
    import argparse
    args = argparse.Namespace(
        run_id="r", campaign="mace-ensemble-screening-20", run_dir=None,
        lm_base_url="http://stub:0/v1", relay_host=None, lm_model=None, lm_user=None,
        max_tokens=None, agents_per_node=None, max_decisions=None,
        redis_port=None, exchange_type="local", http_exchange_url=None,
        no_start_redis=True, system="aurora",
        agents="structure-agent-a,mace-agent",
        no_bootstrap=True,
        agent_count=None,
    )
    # Avoid the actual aurora profile load (we'd need ALCF_USER set,
    # the campaign template, etc). Stub the prep helpers that touch
    # the filesystem.
    monkeypatch.setattr(compute_launcher, "load_system_profile",
                        lambda name: _stub_profile(tmp_path))
    monkeypatch.setattr(compute_launcher, "_prepare_environment",
                        lambda profile, *, exchange_type: None)
    monkeypatch.setattr(compute_launcher, "_load_dashboard_metadata",
                        lambda run_dir: {})
    monkeypatch.setattr(compute_launcher, "_write_lm_config",
                        lambda **kw: tmp_path / "lm.json")
    monkeypatch.setattr(compute_launcher, "_export_workflow_lm_environment",
                        lambda lm_config: None)

    plan = compute_launcher.prepare_compute_launch(args)
    assert plan.agent_count == 2
    assert plan.agents == ("structure-agent-a", "mace-agent")
    assert plan.skip_bootstrap is True


def test_prepare_compute_launch_rejects_disagreeing_agent_count(
    tmp_path, monkeypatch,
) -> None:
    """Disagreeing ``--agent-count`` + ``--agents`` is a footgun:
    silent precedence would let the operator think they were
    launching 3 agents when only 2 ranks actually fire. Refuse loudly."""
    import argparse
    import pytest
    args = argparse.Namespace(
        run_id="r", campaign="mace-ensemble-screening-20", run_dir=None,
        lm_base_url="http://stub:0/v1", relay_host=None, lm_model=None, lm_user=None,
        max_tokens=None, agents_per_node=None, max_decisions=None,
        redis_port=None, exchange_type="local", http_exchange_url=None,
        no_start_redis=True, system="aurora",
        agents="structure-agent-a,mace-agent",
        no_bootstrap=True,
        agent_count=3,  # mismatched -- 2 names but operator says 3
    )
    monkeypatch.setattr(compute_launcher, "load_system_profile",
                        lambda name: _stub_profile(tmp_path))
    monkeypatch.setattr(compute_launcher, "_prepare_environment",
                        lambda profile, *, exchange_type: None)
    monkeypatch.setattr(compute_launcher, "_load_dashboard_metadata",
                        lambda run_dir: {})
    monkeypatch.setattr(compute_launcher, "_write_lm_config",
                        lambda **kw: tmp_path / "lm.json")
    monkeypatch.setattr(compute_launcher, "_export_workflow_lm_environment",
                        lambda lm_config: None)

    with pytest.raises(RuntimeError, match="contradicts --agents"):
        compute_launcher.prepare_compute_launch(args)


def _stub_profile(tmp_path: Path):
    """Minimal SystemProfile-shaped stub for prepare_compute_launch tests."""
    from chemgraph.academy.runtime.profiles.system import SystemProfile
    return SystemProfile(
        name="aurora",
        remote_host="jinchuli@aurora",
        remote_root=str(tmp_path),
        repo_root=str(tmp_path / "ChemGraph"),
        run_root=str(tmp_path / "runs"),
        relay_host_file=str(tmp_path / "relay.host"),
        relay_port=18186,
        venv_python=str(tmp_path / "venv/bin/python"),
        redis_bin_dir=str(tmp_path / "redis/bin"),
        redis_port=6392,
        redis_bind="0.0.0.0",
        redis_protected_mode="no",
        mpiexec="mpiexec",
        pythonpath_entries=[],
        path_entries=[],
        env={},
        unset_env=[],
        no_proxy="127.0.0.1,localhost",
    )
