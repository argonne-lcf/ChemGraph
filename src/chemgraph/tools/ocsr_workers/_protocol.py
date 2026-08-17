"""Shared line-protocol helpers for the local OCSR specialist workers.

STDLIB ONLY. This module is imported by worker scripts that run inside each
model's own conda env (MolNexTR: py3.8; MolScribe/DECIMER: py3.10), so it must
not depend on anything outside the standard library.

Protocol (one JSON object per line, ``\\n``-terminated, UTF-8):

  worker -> client, once at startup:
      {"event": "ready", "model": "...", "backend": "xpu|cuda|cpu", "load_s": 1.2}
  client -> worker, one per image:
      {"cmd": "infer", "id": "gen_000001", "image_path": "/tmp/....png"}
  worker -> client, one response per request (same id echoed):
      {"event": "result", "id": "gen_000001", "smiles": "CCO", "ok": true,
       "infer_s": 0.4}
      {"event": "result", "id": "gen_000001", "smiles": null, "ok": false,
       "error": "..."}
  client -> worker, to shut down:
      {"cmd": "quit"}

The single hard rule for a worker: **stdout carries ONLY protocol lines**. Any
model/library chatter must go to stderr, or it corrupts the stream. Use
``redirect_c_stdout_to_stderr`` around model load, and emit all protocol lines
via ``emit`` (which writes to the saved real stdout and flushes).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from typing import Any, Callable, Dict, Iterator, Optional


# The real stdout fd is captured at import time so protocol lines always reach
# the client even after we redirect Python-level sys.stdout during model load.
_REAL_STDOUT_FD = os.dup(1)
_real_stdout = os.fdopen(_REAL_STDOUT_FD, "w", buffering=1, encoding="utf-8")


def emit(obj: Dict[str, Any]) -> None:
    """Write one protocol JSON object to the real stdout and flush."""
    _real_stdout.write(json.dumps(obj) + "\n")
    _real_stdout.flush()


def emit_ready(model: str, backend: str, load_s: float) -> None:
    emit({"event": "ready", "model": model, "backend": backend, "load_s": round(load_s, 3)})


def emit_result(
    item_id: str,
    smiles: Optional[str],
    ok: bool,
    infer_s: float = 0.0,
    error: str = "",
) -> None:
    rec: Dict[str, Any] = {
        "event": "result",
        "id": item_id,
        "smiles": smiles,
        "ok": bool(ok),
        "infer_s": round(infer_s, 4),
    }
    if error:
        rec["error"] = str(error)[:500]
    emit(rec)


@contextlib.contextmanager
def redirect_c_stdout_to_stderr() -> Iterator[None]:
    """Redirect BOTH Python- and C-level stdout to stderr for the block.

    Model loads (torch/TF) print banners at the C level that ``sys.stdout``
    reassignment alone cannot capture; we dup stderr onto fd 1 so nothing lands
    on the protocol stream. The real stdout stays available via ``emit`` (it
    writes through the fd saved at import, not fd 1).
    """
    sys.stdout.flush()
    saved_fd = os.dup(1)
    try:
        os.dup2(2, 1)  # fd 1 (stdout) -> fd 2 (stderr)
        old_py_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            yield
        finally:
            sys.stdout = old_py_stdout
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


def log(msg: str) -> None:
    """Diagnostic line -> stderr (captured to the per-worker logfile)."""
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


def serve(model_name: str, infer_fn: Callable[[str], Optional[str]]) -> int:
    """Run the request loop until EOF or a ``quit`` command.

    ``infer_fn`` takes an image path and returns a SMILES string (or None). It
    MUST NOT be responsible for protocol I/O; this loop handles reading requests
    and emitting results, and it never lets a per-image exception kill the loop.
    Returns a process exit code.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            log(f"ignoring non-JSON request line: {line[:120]!r}")
            continue

        cmd = req.get("cmd")
        if cmd == "quit":
            log("received quit")
            return 0
        if cmd != "infer":
            log(f"unknown cmd: {cmd!r}")
            continue

        item_id = req.get("id", "")
        image_path = req.get("image_path", "")
        start = time.monotonic()
        try:
            smiles = infer_fn(image_path)
            emit_result(
                item_id,
                smiles or None,
                ok=bool(smiles),
                infer_s=time.monotonic() - start,
                error="" if smiles else "empty prediction",
            )
        except Exception as e:  # one bad image must not kill the worker
            emit_result(
                item_id,
                None,
                ok=False,
                infer_s=time.monotonic() - start,
                error=f"{type(e).__name__}: {e}",
            )
    return 0
