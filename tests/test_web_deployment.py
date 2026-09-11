"""Render deployment configuration without contacting a container or cluster."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("override", [None, "{}"])
def test_compose_keeps_provider_file_unless_json_is_explicit(override):
    docker = shutil.which("docker")
    if not docker:
        pytest.skip("Docker Compose is not installed")
    # Never import host credentials or a local .env into rendered test output.
    env = {key: os.environ[key] for key in ("PATH", "HOME") if key in os.environ}
    if override is not None:
        env["CHEMGRAPH_WEB_PROVIDERS"] = override
    result = subprocess.run(
        [
            docker,
            "compose",
            "--env-file",
            "/dev/null",
            "-f",
            "compose.web.yml",
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    api = json.loads(result.stdout)["services"]["web-api"]
    assert api["environment"].get("CHEMGRAPH_WEB_PROVIDERS") == override
    assert (
        api["environment"]["CHEMGRAPH_WEB_PROVIDERS_FILE"]
        == "/etc/chemgraph/providers.toml"
    )
    mount = next(
        v for v in api["volumes"] if v["target"] == "/etc/chemgraph/providers.toml"
    )
    assert mount["read_only"]
    assert mount["source"] == str(ROOT / "web-providers.example.toml")
    assert not mount.get("bind", {}).get("create_host_path", False)
    assert not api.get("ports")
