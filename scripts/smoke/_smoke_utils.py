"""Shared helpers for the scripts/smoke/* test scripts.

A tiny PASS/FAIL reporter so every script has the same output shape and
exit code semantics. No external dependencies beyond the stdlib.
"""

from __future__ import annotations

import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path


class SmokeReporter:
    def __init__(self, title: str) -> None:
        self.title = title
        self.passed = 0
        self.failed = 0
        self._t0 = time.monotonic()
        print(f"\n=== {title} ===")

    @contextmanager
    def check(self, name: str):
        start = time.monotonic()
        try:
            yield
        except Exception as exc:
            elapsed = time.monotonic() - start
            self.failed += 1
            print(f"[FAIL] {name} ({elapsed:.1f}s): {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            elapsed = time.monotonic() - start
            self.passed += 1
            print(f"[PASS] {name} ({elapsed:.1f}s)")

    def summary_and_exit(self) -> None:
        total = self.passed + self.failed
        wall = time.monotonic() - self._t0
        print(
            f"\n--- {self.title}: {self.passed}/{total} passed, "
            f"{self.failed} failed ({wall:.1f}s total) ---"
        )
        sys.exit(0 if self.failed == 0 else 1)


def require_env(*names: str) -> dict[str, str]:
    """Return a {name: value} dict for the listed env vars, or exit 2 if any
    are missing. Use at the top of scripts that need credentials."""
    import os

    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        print(f"[SKIP] Missing required env vars: {', '.join(missing)}")
        print("       Export them and re-run.")
        sys.exit(2)
    return {n: os.environ[n] for n in names}


def water_xyz_path() -> Path:
    """Absolute path to the shared water.xyz fixture."""
    return Path(__file__).resolve().parent / "water.xyz"


# ── module-level helpers picklable across process / globus boundaries ──


def trivial_add(a: int, b: int) -> int:
    return a + b


def trivial_square(x: int) -> int:
    return x * x


def trivial_hostname() -> str:
    import socket

    return socket.gethostname()


def trivial_env_probe() -> dict:
    import os
    import sys

    info: dict = {
        "hostname": __import__("socket").gethostname(),
        "python": sys.version.split()[0],
        "pid": os.getpid(),
        "cwd": os.getcwd(),
    }
    try:
        info["sched_affinity"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        info["sched_affinity"] = None
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_devices"] = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        info["xpu_devices"] = (
            torch.xpu.device_count() if hasattr(torch, "xpu") and torch.xpu.is_available() else 0
        )
    except Exception as exc:
        info["torch_error"] = str(exc)
    return info
