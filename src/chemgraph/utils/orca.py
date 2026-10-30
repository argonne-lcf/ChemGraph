"""Operator-only ORCA discovery and bounded, shell-free execution."""

import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import time

from chemgraph.utils.executables import resolve_executable


def resolve_orca() -> str:
    """Resolve one raw executable name/path; explicit configuration never falls back."""
    if "ASE_ORCA_COMMAND" in os.environ:
        return resolve_executable("orca", "ASE_ORCA_COMMAND")
    for candidate in ("orca", "/opt/orca/orca", "/usr/local/orca/orca", "~/orca/orca"):
        try:
            return resolve_executable(candidate)
        except FileNotFoundError:
            pass
    raise FileNotFoundError(
        "Install licensed ORCA separately and set ASE_ORCA_COMMAND before importing ChemGraph."
    )


def orca_available() -> bool:
    try:
        resolve_orca()
        return True
    except (OSError, ValueError):
        return False


def timeout_setting(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    if not re.fullmatch(r"[0-9]+", value) or not 1 <= int(value) <= 86400:
        raise ValueError(f"{name} must be an integer in 1..86400 seconds")
    return int(value)


class OrcaBudget:
    """One monotonic deadline shared by the probe and every force evaluation."""

    def __init__(self):
        self.started = time.monotonic()
        self.evaluation_s = timeout_setting("CHEMGRAPH_ORCA_TIMEOUT_S", 3600)
        self.call_s = timeout_setting("CHEMGRAPH_ORCA_CALL_TIMEOUT_S", 7200)

    def remaining(self, limit=None):
        remaining = self.call_s - (time.monotonic() - self.started)
        if remaining <= 0:
            raise TimeoutError(
                "ORCA call deadline exceeded; see CHEMGRAPH_ORCA_CALL_TIMEOUT_S"
            )
        return min(remaining, self.evaluation_s if limit is None else limit)

    def metadata(self):
        return {
            "evaluation_timeout_s": self.evaluation_s,
            "call_timeout_s": self.call_s,
            "elapsed_s": time.monotonic() - self.started,
        }


def run_bounded(argv, directory, output, error, timeout):
    """Reap the process and kill its entire process group on timeout."""
    with open(output, "wb") as stdout, open(error, "wb") as stderr:
        with subprocess.Popen(
            argv, cwd=directory, stdout=stdout, stderr=stderr, start_new_session=True
        ) as process:
            try:
                return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # The process exited between wait's timeout and kill.
                process.wait()
                raise TimeoutError(
                    f"ORCA exceeded {timeout:g} seconds; diagnostic artifacts preserved"
                ) from exc


_IDENTITIES = {}


def identify_orca(executable: str, directory: Path, budget: OrcaBudget) -> str:
    stat = Path(executable).stat()
    key = (executable, stat.st_mtime_ns, stat.st_size)
    budget.remaining(10)
    if key in _IDENTITIES:
        return _IDENTITIES[key]
    probe = Path(tempfile.mkdtemp(prefix="identity-", dir=directory))
    output, error = probe / "probe.out", probe / "probe.err"
    run_bounded(
        [executable, "chemgraph-nonexistent-input.inp"],
        probe,
        output,
        error,
        budget.remaining(10),
    )
    match = re.search(
        r"Program Version\s+(6\.\d+\.\d+)", output.read_text(errors="replace")
    )
    if not match:
        raise ValueError(
            "ASE_ORCA_COMMAND must identify quantum-chemistry ORCA 6.x (Program Version header); check installation"
        )
    _IDENTITIES[key] = match[1]
    return match[1]
