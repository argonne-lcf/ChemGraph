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
