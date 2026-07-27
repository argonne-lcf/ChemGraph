from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

# Skip when the optional 'academy' extra is absent.
pytest.importorskip("academy")

from chemgraph.academy.runtime import dashboard_launcher
from chemgraph.academy.runtime.profiles.system import SubmitDefaults
from chemgraph.academy.runtime.profiles.system import SystemProfile


def _profile(tmp_path: Path) -> SystemProfile:
    return SystemProfile(
        name="test-system",
        remote_host="user@example",
        remote_root="/remote/root",
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
        pythonpath_entries=[str(tmp_path), "/remote/root/ChemGraph/src"],
        no_proxy="127.0.0.1,localhost",
        submit_defaults=SubmitDefaults(
            queue="debug",
            walltime="00:30:00",
            nodes=1,
            filesystems="home:eagle",
        ),
    )


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = {
        "run_id": "run-001",
        "system": "test-system",
        "lm_connect": "direct",
        "lm_base_url": "http://lm.example/v1",
        "remote_host": None,
        "ssh_control_path": str(tmp_path / "ssh-control"),
        "keep_ssh_master": False,
        "local_argo_host": "127.0.0.1",
        "local_argo_port": 18085,
        "reverse_port": 18185,
        "relay_port": None,
        "relay_python": None,
        "rsync_interval_s": 2.0,
        "local_mirror_root": str(tmp_path / "mirror"),
        "local_run_dir": None,
        "dashboard_host": "127.0.0.1",
        "dashboard_port": 8765,
        "local": False,
        "no_dashboard": True,
        "enable_launch_buttons": False,
        "bundle_root": None,
        "project": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_compute_wrapper_template_renders_profile_values(tmp_path) -> None:
    text = dashboard_launcher.wrapper(_profile(tmp_path))

    assert "%{" not in text
    assert '/remote/root/tools/redis/bin:/remote/root/bin:${PATH}' in text
    assert f'{tmp_path}:/remote/root/ChemGraph/src:${{PYTHONPATH:-}}' in text
    assert "/remote/root/venv/bin/python" in text


def test_dashboard_launcher_writes_remote_state(tmp_path, monkeypatch) -> None:
    local_run = tmp_path / "mirror" / "run-001"
    local_run.mkdir(parents=True)
    (local_run / "status.json").write_text("{}\n", encoding="utf-8")
    calls: list[dict] = []

    def fake_ssh(host, command, **kwargs):
        calls.append({"host": host, "command": command, **kwargs})
        return subprocess.CompletedProcess(["ssh"], 0, stdout="")

    monkeypatch.setattr(dashboard_launcher, "parse_args", lambda: _args(tmp_path))
    monkeypatch.setattr(dashboard_launcher, "load_system_profile", lambda _: _profile(tmp_path))
    monkeypatch.setattr(dashboard_launcher, "ssh", fake_ssh)
    monkeypatch.setattr(dashboard_launcher, "start_rsync", lambda *args, **kwargs: None)

    assert dashboard_launcher.main() == 0
    assert local_run.exists()

    wrapper_call = calls[1]
    assert wrapper_call["command"].endswith("chmod +x /remote/root/bin/swarm-run")
    assert "chemgraph.academy.runtime.compute_launcher" in wrapper_call["input_text"]

    metadata = json.loads(calls[2]["input_text"])
    assert metadata["run_id"] == "run-001"
    assert metadata["lm_base_url"] == "http://lm.example/v1"
    assert metadata["remote_run_dir"] == "/remote/root/runs/run-001"


def test_dashboard_launcher_generates_session_run_id(tmp_path, monkeypatch) -> None:
    calls: list[dict] = []

    def fake_ssh(host, command, **kwargs):
        calls.append({"host": host, "command": command, **kwargs})
        return subprocess.CompletedProcess(["ssh"], 0, stdout="")

    monkeypatch.setattr(
        dashboard_launcher,
        "parse_args",
        lambda: _args(tmp_path, run_id=None),
    )
    monkeypatch.setattr(dashboard_launcher, "load_system_profile", lambda _: _profile(tmp_path))
    monkeypatch.setattr(dashboard_launcher, "ssh", fake_ssh)
    monkeypatch.setattr(dashboard_launcher, "start_rsync", lambda *args, **kwargs: None)
    monkeypatch.setattr(dashboard_launcher.time, "strftime", lambda _: "session-2026-07-27-1200")

    assert dashboard_launcher.main() == 0
    metadata = json.loads(calls[2]["input_text"])
    assert metadata["run_id"] == "session-2026-07-27-1200"
