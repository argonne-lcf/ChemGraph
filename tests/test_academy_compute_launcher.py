from __future__ import annotations

from pathlib import Path

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
