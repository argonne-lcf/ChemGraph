from __future__ import annotations

import shlex
import subprocess


def _default_control_path() -> str:
    # Matches the launcher default (dashboard_launcher.py) so plain
    # callers of ssh_run land on the same warmed master rather than
    # opening a new connection (which re-prompts for MFA every time).
    from pathlib import Path
    return str(Path.home() / ".ssh/cm-%r@%h:%p")


def _control_opts(control_path: str | None) -> list[str]:
    path = control_path or _default_control_path()
    return [
        "-o", f"ControlPath={path}",
        "-o", "ControlMaster=auto",
        "-o", "ControlPersist=yes",
    ]


def ssh_run(
    host: str,
    command: str,
    *,
    capture: bool = True,
    timeout_s: float | None = 30.0,
    check: bool = True,
    control_path: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` on ``host`` via ssh, reusing the shared
    ControlMaster socket so repeated calls do not re-prompt MFA.

    Defaults to ``~/.ssh/cm-%r@%h:%p`` (matches the dashboard-launcher's
    default). Override with ``control_path`` if the operator's config
    uses a different pattern.
    """
    argv = ["ssh", *_control_opts(control_path), host, command]
    return subprocess.run(
        argv,
        capture_output=capture,
        text=True,
        timeout=timeout_s,
        check=check,
    )


def scp_upload(
    local_path: str,
    host: str,
    remote_path: str,
    *,
    timeout_s: float | None = 60.0,
    control_path: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """scp with the same ControlMaster reuse as ssh_run.

    scp does NOT inherit ssh master settings from ~/.ssh/config the
    same way plain ssh does -- must pass -o ControlPath on the argv.
    Without this each save re-prompts for MFA.
    """
    argv = [
        "scp",
        *_control_opts(control_path),
        local_path,
        f"{host}:{remote_path}",
    ]
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def ssh_quote(s: str) -> str:
    """Quote for inclusion in a remote shell command."""
    return shlex.quote(s)


if __name__ == "__main__":  # ponytail: smoke against localhost if available
    try:
        r = ssh_run("localhost", "echo ok", timeout_s=5)
        assert r.stdout.strip() == "ok"
        print("ssh_transport self-check ok (localhost)")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        # No local sshd or no key set up; skip silently.
        print("ssh_transport self-check skipped (no local ssh)")
