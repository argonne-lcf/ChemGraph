"""Core subprocess runner for non-interactive Multiwfn analyses."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from chemgraph.schemas.multiwfn_schema import MultiwfnInputSchema, MultiwfnResult
from chemgraph.tools.ase_core import _resolve_existing_path

_OUTPUT_TAIL_CHARS = 8_000
_TERMINATE_GRACE_S = 5.0


def _resolve_multiwfn_executable() -> Path:
    """Resolve and validate the server-configured Multiwfn executable."""
    configured = os.environ.get("MULTIWFN_EXE")
    if not configured:
        raise ValueError(
            "MULTIWFN_EXE is not set. Set it to the path of the Multiwfn "
            "executable before starting ChemGraph."
        )

    candidate = Path(configured).expanduser()
    if candidate.is_absolute() or os.sep in configured:
        executable = candidate.resolve()
    else:
        found = shutil.which(configured)
        if found is None:
            raise FileNotFoundError(
                f"Multiwfn executable was not found on PATH: {configured}"
            )
        executable = Path(found).resolve()

    if not executable.is_file():
        raise FileNotFoundError(f"Multiwfn executable does not exist: {executable}")
    if not os.access(executable, os.X_OK):
        raise PermissionError(f"Multiwfn executable is not executable: {executable}")
    return executable


def _resolve_multiwfn_home(executable: Path) -> Path:
    """Return the configured Multiwfn distribution directory."""
    configured = os.environ.get("MULTIWFN_HOME")
    if configured is None:
        return executable.parent

    home = Path(configured).expanduser().resolve()
    if not home.is_dir():
        raise NotADirectoryError(f"MULTIWFN_HOME is not a directory: {home}")
    return home


def _read_tail(path: Path, max_chars: int = _OUTPUT_TAIL_CHARS) -> str:
    """Read a bounded diagnostic tail without loading a large output file."""
    if not path.is_file() or max_chars <= 0:
        return ""

    max_bytes = max_chars * 4
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        text = handle.read().decode("utf-8", errors="replace")
    return text[-max_chars:]


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate Multiwfn and any subprocesses it started."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return

    try:
        process.wait(timeout=_TERMINATE_GRACE_S)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _generated_artifacts(run_dir: Path, reserved: set[Path]) -> list[str]:
    """List files created by Multiwfn, excluding the runner's stream files."""
    return sorted(
        str(path.absolute())
        for path in run_dir.rglob("*")
        if (path.is_file() or path.is_symlink()) and path not in reserved
    )


def run_multiwfn_core(params: MultiwfnInputSchema) -> MultiwfnResult:
    """Run Multiwfn in documented batch mode and return execution metadata.

    The executable path is supplied through ``MULTIWFN_EXE`` rather than the
    agent-facing schema. Multiwfn receives one exact menu response per line on
    stdin, and all console output is captured in the invocation's run directory.
    """
    executable = _resolve_multiwfn_executable()
    multiwfn_home = _resolve_multiwfn_home(executable)

    raw_input = str(Path(params.input_file).expanduser())
    input_path = Path(_resolve_existing_path(raw_input)).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Multiwfn input file not found: {input_path}")

    log_root = Path(
        os.environ.get("CHEMGRAPH_LOG_DIR", Path.cwd() / "cg_logs")
    ).expanduser()
    log_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="multiwfn_", dir=log_root)).resolve()

    stdin_path = run_dir / "multiwfn.in"
    stdout_path = run_dir / "multiwfn_stdout.txt"
    stderr_path = run_dir / "multiwfn_stderr.txt"
    stdin_path.write_text("\n".join(params.menu_inputs) + "\n", encoding="utf-8")

    child_env = os.environ.copy()
    child_env["Multiwfnpath"] = str(multiwfn_home)

    started = time.monotonic()
    timed_out = False
    with (
        stdin_path.open("r", encoding="utf-8") as stdin_handle,
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
    ):
        process = subprocess.Popen(
            [str(executable), str(input_path)],
            cwd=run_dir,
            env=child_env,
            stdin=stdin_handle,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        try:
            process.wait(timeout=params.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)

    duration_s = time.monotonic() - started
    if timed_out:
        status = "timeout"
    elif process.returncode == 0:
        status = "success"
    else:
        status = "failure"

    reserved = {stdin_path, stdout_path, stderr_path}
    return {
        "status": status,
        "return_code": process.returncode,
        "duration_s": round(duration_s, 3),
        "executable": str(executable),
        "input_file": str(input_path),
        "run_directory": str(run_dir),
        "stdin_file": str(stdin_path),
        "stdout_file": str(stdout_path),
        "stderr_file": str(stderr_path),
        "artifacts": _generated_artifacts(run_dir, reserved),
        "stdout_tail": _read_tail(stdout_path),
        "stderr_tail": _read_tail(stderr_path),
    }
