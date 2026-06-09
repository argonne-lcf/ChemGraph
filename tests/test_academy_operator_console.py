from __future__ import annotations

from pathlib import Path

import pytest

from chemgraph.academy.runtime import operator_console
from chemgraph.academy.runtime.profiles.system import SystemProfile


def _profile(tmp_path: Path) -> SystemProfile:
    return SystemProfile(
        name="test-system",
        operator_host="user@example",
        remote_root="/remote/root",
        academy_repo_root="/remote/root/academy",
        repo_root="/remote/root/ChemGraph",
        run_root="/remote/root/runs",
        relay_host_file="/remote/root/relay.host",
        relay_port=18186,
        venv_python="/remote/root/venv/bin/python",
        redis_bin_dir="/remote/root/tools/redis/bin",
        redis_port=6392,
        redis_bind="0.0.0.0",
        redis_protected_mode="no",
        mpiexec="mpiexec",
        pythonpath_entries=[str(tmp_path)],
        no_proxy="127.0.0.1,localhost",
    )


def test_delete_existing_run_removes_remote_and_local(tmp_path, monkeypatch) -> None:
    local_run = tmp_path / "mirror" / "run-001"
    local_run.mkdir(parents=True)
    (local_run / "status.json").write_text("{}\n", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(
        operator_console,
        "_run",
        lambda command, **kwargs: calls.append(command),
    )

    operator_console._delete_existing_run(
        profile=_profile(tmp_path),
        host="user@example",
        ssh_opts=["-o", "BatchMode=yes"],
        run_id="run-001",
        local_run_dir=local_run,
    )

    assert not local_run.exists()
    assert calls
    assert calls[0][:4] == ["ssh", "-o", "BatchMode=yes", "user@example"]
    assert 'mv -- "$run_dir" "$trash_dir"' in calls[0][-1]
    assert 'rm -rf -- "$trash_dir"' in calls[0][-1]
    assert 'mkdir -p "$run_dir"' in calls[0][-1]


def test_delete_existing_run_rejects_unsafe_run_id(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="unsafe run id"):
        operator_console._delete_existing_run(
            profile=_profile(tmp_path),
            host="user@example",
            ssh_opts=[],
            run_id="../bad",
            local_run_dir=tmp_path / "mirror",
        )
