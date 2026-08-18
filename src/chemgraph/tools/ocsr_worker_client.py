"""Drive the specialist OCSR workers: spawn, talk, tear down.

Each specialist model lives in its own conda environment (see
:mod:`chemgraph.tools.ocsr_setup`) and runs as a subprocess speaking the JSON line
protocol in :mod:`chemgraph.tools.ocsr_workers._protocol`. This module is the parent
side of that conversation.

Workers are expensive to start (50-66 s for DECIMER, measured, dominated by loading
TensorFlow and the weights) and cheap to reuse (0.5-5 s per image). So one worker per
model is started on first use and kept, with an idle reaper closing it after a period
of inactivity rather than holding a multi-gigabyte process for the life of a CLI
session.

Ported from the OCSR benchmark's ``local_client._Worker``, which has driven thousands
of rows, with three deliberate changes: it takes image bytes rather than a
``BenchItem``, it returns a plain dict rather than a ``Prediction``, and the read loop
no longer blocks past its deadline (see :meth:`_Worker._read_until`).
"""

from __future__ import annotations

import atexit
import contextlib
import io
import json
import logging
import os
import selectors
import subprocess
import tempfile
import threading
import time
from importlib import resources

logger = logging.getLogger(__name__)

# Long enough for a cold TensorFlow import plus weights off a shared filesystem;
# DECIMER measured 66 s on an idle machine and can be slower under load.
DEFAULT_STARTUP_TIMEOUT_S = 300.0
DEFAULT_INFER_TIMEOUT_S = 120.0
# Interactive sessions should not sit on ~2 GB and a torch process indefinitely.
_MAX_LOG_BYTES = 5 * 1024 * 1024

# Longest single protocol line we will buffer. Messages are small JSON objects, so
# anything past this means the worker is emitting a stream with no newline in it, and
# waiting for one would grow the parent's memory until the deadline (measured: about
# 1.8 GB over the 300 s startup budget).
_MAX_LINE_BYTES = 1024 * 1024


def _die_with_parent() -> None:
    """Ask the kernel to SIGTERM this child when its parent dies.

    ``atexit`` and the ``BaseException`` handler both need the parent to keep
    running, so neither covers SIGKILL, an OOM kill, or a job teardown. A worker
    already blocked on stdin sees EOF and exits on its own, but one still loading
    its model is not reading yet, and that window is up to 66 s during which several GB
    would be stranded on a shared node with nothing to reap it.

    Linux only, and best effort: any failure here must not stop the spawn.
    """
    with contextlib.suppress(Exception):
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG

DEFAULT_IDLE_TIMEOUT_S = 600.0

_WORKER_PACKAGE = "chemgraph.tools.ocsr_workers"

# Set once, so the whole ocsr_workers directory is on disk for the life of the process
# rather than per spawn. The workers do `sys.path.insert(0, dirname(__file__))` and then
# `import _protocol`, so the script and its sibling must be extracted together; pulling
# out a single file would leave the import dangling under a zipimport.
_workers_dir: str | None = None
_workers_stack = contextlib.ExitStack()
atexit.register(_workers_stack.close)


def workers_dir() -> str:
    """Filesystem path to the packaged worker scripts, materializing them if needed.

    Copies the whole directory rather than calling ``as_file`` on it. Two reasons,
    both invisible under an editable install and fatal under a wheel: ``as_file`` on
    a *directory* traversable raises ``IsADirectoryError`` from a zip, and extracting
    a single file would leave ``_protocol.py`` behind, which the workers import by
    sitting next to it.
    """
    global _workers_dir
    if _workers_dir is None:
        pkg = resources.files(_WORKER_PACKAGE)
        direct = getattr(pkg, "_paths", None) or pkg
        if isinstance(direct, os.PathLike) or os.path.isdir(str(pkg)):
            _workers_dir = str(pkg)          # normal filesystem install
        else:
            out = _workers_stack.enter_context(tempfile.TemporaryDirectory(
                prefix="chemgraph_ocsr_workers_"))
            for entry in pkg.iterdir():
                if entry.is_file():
                    with open(os.path.join(out, entry.name), "wb") as fh:
                        fh.write(entry.read_bytes())
            _workers_dir = out
    return _workers_dir


def worker_script(model: str) -> str:
    """Path to one model's `*_infer.py`."""
    return os.path.join(workers_dir(), f"{model}_infer.py")


class WorkerError(RuntimeError):
    """A worker could not be started, or died, or did not answer in time."""


class _Worker:
    """One model's subprocess: start it, ask it for SMILES, shut it down."""

    def __init__(self, model: str, cfg: dict):
        self.model = model
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.last_used = time.monotonic()
        self._log_fh = None
        self._buf = ""   # partial line carried across reads, see _read_until

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        script = worker_script(self.model)
        if not os.path.exists(script):
            raise WorkerError(f"no worker script for {self.model!r} at {script}")

        python_bin = os.path.expanduser(os.path.expandvars(self.cfg["python_bin"]))
        if not os.path.exists(python_bin):
            raise WorkerError(
                f"{self.model} is not installed ({python_bin} does not exist). "
                f"Install it with: python -m chemgraph.tools.ocsr_setup {self.model}"
            )

        cmd = [python_bin, script, "--device", str(self.cfg.get("device", "cpu"))]
        weights = self.cfg.get("weights_path")
        if weights:
            cmd += ["--weights", os.path.expanduser(os.path.expandvars(weights))]
        # OCSRGlyph sets its own torch thread count and would ignore the environment,
        # so it takes the cap on the command line instead.
        threads = int(self.cfg.get("threads", 4))
        if self.model == "ocsrglyph":
            cmd += ["--threads", str(threads)]

        log_dir = os.path.expanduser("~/.chemgraph/ocsr_logs")
        os.makedirs(log_dir, exist_ok=True)
        # A real logfile, never DEVNULL: a worker that dies during import writes its
        # traceback to stderr, and without this the parent only sees a timeout.
        #
        # Truncate at spawn if it has grown past the cap. This bounds what a chatty
        # worker accumulates across sessions; it does not stop one already running,
        # which would need the parent to relay every line. The log exists only to
        # recover an import traceback, so a few MB is ample.
        log_path = os.path.join(log_dir, f"{self.model}.log")
        with contextlib.suppress(OSError):
            if os.path.getsize(log_path) > _MAX_LOG_BYTES:
                os.truncate(log_path, 0)
        self._log_fh = open(log_path, "a")

        env = {
            **os.environ,
            # The worker envs are conda envs, not venvs, so ~/.local/lib/pythonX.Y
            # takes priority over their own site-packages. That silently broke
            # MolScribe once: a user-site torch shadowed the env's and torchvision
            # failed to import inside the subprocess, where nobody could see it.
            "PYTHONNOUSERSITE": "1",
            # Four workers at library defaults is 224 threads on a 104-core shared
            # machine. Cap them.
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
            "TF_NUM_INTRAOP_THREADS": str(threads),
        }

        logger.info("starting OCSR worker %s (first call loads the model)", self.model)
        # stdout is deliberately NOT text-wrapped: _read_until does its own framing
        # with os.read, and a TextIOWrapper in front of it would hide bytes from the
        # selector. stdin is wrapped for convenience since we only ever write to it.
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log_fh, bufsize=0, env=env,
            preexec_fn=_die_with_parent,
        )
        self.stdin = io.TextIOWrapper(self.proc.stdin, encoding="utf-8", write_through=True)
        timeout = float(self.cfg.get("startup_timeout_s", DEFAULT_STARTUP_TIMEOUT_S))
        try:
            msg = self._read_until("ready", timeout)
        except BaseException:
            # A worker that never says `ready` is not registered with the client, so
            # nothing downstream would ever reap it. Left alone that leaks a ~2 GB
            # process per failed startup, outliving the interpreter.
            self.close()
            raise
        logger.info("OCSR worker %s ready in %.1fs", self.model, msg.get("load_s", 0.0))

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def close(self) -> None:
        """Ask the worker to quit, then insist.

        The escalation is not defensive padding. A TensorFlow worker can take well
        over ten seconds to unwind after acknowledging `quit`, so a bare `wait()`
        either hangs the caller or, if the caller gives up first, leaks a multi-
        gigabyte process.
        """
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                with contextlib.suppress(Exception):
                    self.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                    self.stdin.flush()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
        finally:
            if self._log_fh is not None:
                with contextlib.suppress(Exception):
                    self._log_fh.close()
                self._log_fh = None
            self.proc = None

    # -- protocol ----------------------------------------------------------

    def _read_until(self, event: str, timeout_s: float, req_id: str | None = None) -> dict:
        """Read protocol lines until `event` arrives, honouring a real deadline.

        Two failure modes have to be handled at once, and fixing one naively breaks
        the other.

        The benchmark's version looped `while time.monotonic() < deadline:` around a
        bare `readline()`, which blocks indefinitely on a pipe. The deadline was only
        consulted *between* lines, so a worker that went quiet hung the caller for
        ever. That never surfaced there because the failures seen were crashes, where
        EOF makes `readline` return immediately.

        The obvious fix, selecting on the pipe before each `readline()`, is also
        wrong: it selects on the *file descriptor* while reading through a buffered
        `TextIOWrapper`. When a worker emits chatter and its result in one write, both
        land in the buffer on the first read, the fd then has nothing left, and select
        blocks until the deadline with the answer already in hand. MolNexTR does
        exactly this (it prints `chiral_center_ids` before its result), so the bug
        would have hit a real model while passing every test on DECIMER, whose stdout
        is clean.

        So: no buffered reader. Read raw bytes with `os.read` and split lines here,
        keeping a leftover buffer across iterations. Non-protocol lines are skipped;
        the models print to stdout despite the worker's redirection.
        """
        fd = self.proc.stdout.fileno()
        sel = selectors.DefaultSelector()
        sel.register(fd, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout_s
        pending: list[str] = []
        try:
            while True:
                while pending:
                    line = pending.pop(0)
                    try:
                        msg = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("event") == "error":
                        raise WorkerError(
                            f"{self.model}: {msg.get('error', 'unknown error')}"
                        )
                    if msg.get("event") == event and (
                        req_id is None or msg.get("id") == req_id
                    ):
                        return msg

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WorkerError(
                        f"{self.model} did not send {event!r} within {timeout_s:.0f}s "
                        f"(see ~/.chemgraph/ocsr_logs/{self.model}.log)"
                    )
                if not sel.select(timeout=remaining):
                    continue
                chunk = os.read(fd, 65536)
                if not chunk:
                    raise WorkerError(
                        f"{self.model} exited while waiting for {event!r} "
                        f"(see ~/.chemgraph/ocsr_logs/{self.model}.log)"
                    )
                self._buf += chunk.decode("utf-8", errors="replace")
                *complete, self._buf = self._buf.split("\n")
                if len(self._buf) > _MAX_LINE_BYTES:
                    raise WorkerError(
                        f"{self.model} sent {len(self._buf)} bytes with no newline; "
                        f"the protocol is line-based, so this worker is not speaking it"
                    )
                pending.extend(complete)
        finally:
            sel.close()

    def infer(self, req_id: str, image_path: str, timeout_s: float) -> dict:
        """Send one image, return the worker's raw result dict."""
        if not self.alive():
            raise WorkerError(f"{self.model} is not running")
        self.stdin.write(
            json.dumps({"cmd": "infer", "id": req_id, "image_path": image_path}) + "\n"
        )
        self.stdin.flush()
        self.last_used = time.monotonic()
        try:
            return self._read_until("result", timeout_s, req_id=req_id)
        finally:
            # Stamp on completion too, not only on send: a long inference should
            # count as "used" when it finishes, or the reaper sees a stale timestamp
            # the moment the call returns.
            self.last_used = time.monotonic()


class OCSRWorkerClient:
    """Keeps one worker per model alive, and reaps them when idle."""

    def __init__(self, config: dict, idle_timeout_s: float = DEFAULT_IDLE_TIMEOUT_S):
        self.config = config
        self.idle_timeout_s = idle_timeout_s
        self._workers: dict[str, _Worker] = {}
        self._spawn_lock = threading.Lock()
        self._tmp_dir = tempfile.mkdtemp(prefix="chemgraph_ocsr_")
        os.chmod(self._tmp_dir, 0o700)
        atexit.register(self.close)

    def _get(self, model: str) -> _Worker:
        with self._spawn_lock:
            self._reap_idle()
            w = self._workers.get(model)
            if w is not None and w.alive():
                return w
            if model not in self.config:
                raise WorkerError(
                    f"no configuration for {model!r}. Known: {sorted(self.config)}"
                )
            w = _Worker(model, self.config[model])
            w.start()
            self._workers[model] = w
            return w

    def _reap_idle(self) -> None:
        """Close workers that have gone unused. Never one that is mid-inference.

        ``last_used`` is stamped when a request is *sent*, so a call that runs longer
        than the idle timeout would otherwise look idle and be closed out from under
        itself. The per-worker lock is the authority on whether one is busy: take it
        without blocking, and skip the worker if we cannot.
        """
        now = time.monotonic()
        for name, w in list(self._workers.items()):
            if not (w.alive() and (now - w.last_used) > self.idle_timeout_s):
                continue
            if not w.lock.acquire(blocking=False):
                continue  # in flight, and its own timeout governs it
            try:
                logger.info("closing idle OCSR worker %s", name)
                w.close()
                self._workers.pop(name, None)
            finally:
                w.lock.release()

    def predict(self, model: str, image_bytes: bytes, timeout_s: float | None = None) -> dict:
        """Run one image through one model. Never raises.

        Takes bytes rather than a path so the file the worker opens is the one the
        caller validated, closing the window where a path could be swapped in
        between. The protocol needs a filename, so the bytes are written to a
        private temp file for the duration of the call.

        Returns ``{"ok", "smiles", "error", "infer_s", "cold_start"}``.
        """
        cfg = self.config.get(model, {})
        if timeout_s is None:
            timeout_s = float(cfg.get("timeout_s", DEFAULT_INFER_TIMEOUT_S))

        was_running = model in self._workers and self._workers[model].alive()
        path = os.path.join(self._tmp_dir, f"{model}_{os.getpid()}_{time.monotonic_ns()}.png")
        t0 = time.monotonic()
        try:
            with open(path, "wb") as fh:
                fh.write(image_bytes)
            worker = self._get(model)
            with worker.lock:
                msg = worker.infer(str(time.monotonic_ns()), path, timeout_s)
            return {
                "ok": bool(msg.get("ok")),
                "smiles": msg.get("smiles"),
                "error": msg.get("error", "") or "",
                "infer_s": float(msg.get("infer_s", 0.0)),
                "cold_start": not was_running,
            }
        except BaseException as e:
            # BaseException, not Exception: a KeyboardInterrupt during a 60 s model
            # load must still tear the worker down rather than leave it orphaned.
            self._drop(model)
            if isinstance(e, KeyboardInterrupt):
                raise
            return {"ok": False, "smiles": None, "error": f"{type(e).__name__}: {e}",
                    "infer_s": time.monotonic() - t0, "cold_start": not was_running}
        finally:
            with contextlib.suppress(OSError):
                os.unlink(path)

    def _drop(self, model: str) -> None:
        w = self._workers.pop(model, None)
        if w is not None:
            with contextlib.suppress(Exception):
                w.close()

    def close(self) -> None:
        for name in list(self._workers):
            self._drop(name)
        with contextlib.suppress(Exception):
            import shutil

            shutil.rmtree(self._tmp_dir, ignore_errors=True)
