from __future__ import annotations

import shlex
import subprocess


def ssh_run(
    host: str,
    command: str,
    *,
    capture: bool = True,
    timeout_s: float | None = 30.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` on ``host`` via ssh. Relies on operator's
    ``~/.ssh/config`` (ControlMaster, identity, etc).
    """
    argv = ["ssh", host, command]
    return subprocess.run(
        argv,
        capture_output=capture,
        text=True,
        timeout=timeout_s,
        check=check,
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
