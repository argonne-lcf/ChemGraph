"""In-process subprocess wrapper that drives the launcher from the
dashboard's HTTP surface.

Backend for the per-HPC launch buttons and the operator inject-
message panel. Subprocesses ``swarm launch`` and ``swarm inject``
(the same CLIs the operator runs from a terminal), tails their
stderr line-by-line, parses state transitions, and exposes a JSON
snapshot the frontend polls.

Decision-doc: see plan.private-local.md. Tldr: option A (subprocess
+ stderr parsing) is the smallest delta. Future PR can swap to
launcher --json output without touching the dashboard.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

# ANSI escape stripper so colored launcher output doesn't leak into the
# parsed state strings. Operator-facing text in the UI strips colors;
# raw stderr in the sidebar can keep them if we want fancy rendering
# later.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# State patterns we recognise from launcher stderr. Anything matching
# advances the per-site state machine; anything else just gets appended
# to the per-site log buffer.
#
# Examples the launcher emits today (see remote_launcher.py /
# submit_backend.py):
#   [preflight] ok    dashboard aurora relay=...
#   [submit:aurora] qsub -> 12345.aurora-pbs-...
#   [submit:aurora] job 12345.aurora-pbs-... state -> Q
#   [submit:aurora] job 12345.aurora-pbs-... state -> R
#   [attach:aurora] waiting for [...] to register (elapsed Ns of Ns)
#   [launch] ready: aurora -> ['agent-aurora']
#   ok: sent to <recipient> (message_id=<id>)   [from swarm inject]
_QSUB_RE = re.compile(r"\[submit:(?P<site>[\w-]+)\] qsub -> (?P<job_id>\S+)")
_STATE_RE = re.compile(r"\[submit:(?P<site>[\w-]+)\] job \S+ state -> (?P<state>\w+)")
_READY_RE = re.compile(r"\[launch\] ready: (?P<site>[\w-]+) -> ")
_INJECT_OK = re.compile(r"^ok: sent to ")
_INJECT_FAIL = re.compile(r"^inject failed:")
_WAIT_RE = re.compile(r"\[attach:(?P<site>[\w-]+)\] waiting ")
# Errors that name a specific site -- mark that site failed:
#   [launch] wait_ready failed: attach[crux]: agents [...] did not register
#   [launch] wait_ready failed: submit[crux]: job 12345 finished without running
#   [launch] start failed for crux: <exc>
#   attach[crux]: ssh exited with code N before agents registered
#   submit[crux]: job 12345 finished without running
_SITE_FAIL_RE = re.compile(
    r"(?:attach|submit)\[(?P<site>[\w-]+)\][:\]]"
)
# Generic "wait_ready failed" or "start failed" without a parsable site
# also indicates something went wrong; we don't know WHICH site so we
# leave per-site state alone but raise a global "launcher_error" so the
# UI can warn.
_GENERIC_FAIL_RE = re.compile(
    r"\[launch\] (?:wait_ready|start) failed|\[launch\] interrupted"
)

SiteState = Literal[
    "idle", "submitting", "queued", "running", "ready", "failed",
]


@dataclass
class _SiteSnapshot:
    site: str
    state: SiteState = "idle"
    pbs_state: str | None = None   # raw PBS state letter (Q, R, F, ...)
    job_id: str | None = None
    last_event: str | None = None
    # True while a launcher subprocess covering this site is live.
    # Frontend uses this to disable only that site's Launch button
    # while another site's launch is still queuing / warming up.
    running: bool = False


@dataclass
class _RunSnapshot:
    sites: dict[str, _SiteSnapshot] = field(default_factory=dict)
    inject: Literal["idle", "in_flight", "dispatched", "failed"] = "idle"
    # Last ~200 lines of launcher stderr (from all live subprocesses,
    # interleaved), with ANSI stripped.
    log_lines: list[str] = field(default_factory=list)
    # Sequence number that increments with every state update so the
    # frontend can long-poll efficiently (skip if seq unchanged).
    seq: int = 0


_LOG_RING_SIZE = 200


class LaunchController:
    """Owns the per-site subprocesses + shared state snapshot. One
    instance per dashboard process. Thread-safe; the HTTP request
    threads poke at the snapshot while per-subprocess stderr-pump
    threads update it.

    Concurrency model: each canvas Launch button spawns its own
    ``swarm launch`` subprocess covering that site (or the
    set passed in the /api/launch body). Sites launch independently
    -- one can be queuing while another is being clicked. Refusal is
    per-site: a second Launch aurora click while aurora is still
    live returns 409, but Launch crux stays enabled the whole time.

    inject-message is still one-shot cross-site (no site key).
    """

    def __init__(self, sites: tuple[str, ...] = ()) -> None:
        self._lock = threading.Lock()
        self._snapshot = _RunSnapshot(
            sites={s: _SiteSnapshot(site=s) for s in sites},
        )
        # key: sorted-tuple of sites owned by that subprocess.
        self._procs: dict[tuple[str, ...], subprocess.Popen[str]] = {}
        self._inject_proc: subprocess.Popen[str] | None = None

    def set_sites(self, sites: tuple[str, ...]) -> None:
        """Ensure the snapshot has an entry for each currently-relevant
        site. Preserves running sites; only resets idle/finished ones
        that the operator has re-included.

        Called at the start of each /api/launch so the snapshot map
        matches whatever the canvas selected. Sites currently mid-run
        keep their state.
        """
        with self._lock:
            for s in sites:
                cur = self._snapshot.sites.get(s)
                if cur is None or not cur.running:
                    self._snapshot.sites[s] = _SiteSnapshot(site=s)
            self._snapshot.seq += 1

    # ------------------------------------------------------------------
    # Read API (called by HTTP GET /api/launch-status)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            snap = self._snapshot
            any_running = any(s.running for s in snap.sites.values())
            return {
                "seq": snap.seq,
                "inject": snap.inject,
                # Kept for back-compat with any UI code still reading
                # this field; per-site .running is the source of truth.
                "launcher_running": any_running,
                "launcher_exit_code": None,
                "sites": {
                    name: {
                        "state": s.state,
                        "pbs_state": s.pbs_state,
                        "job_id": s.job_id,
                        "last_event": s.last_event,
                        "running": s.running,
                    }
                    for name, s in snap.sites.items()
                },
                "log_lines": list(snap.log_lines),
            }

    # ------------------------------------------------------------------
    # Write API (called by HTTP POST /api/launch and /api/inject-message)
    # ------------------------------------------------------------------

    def launch(self, argv: list[str], sites: tuple[str, ...]) -> tuple[bool, str]:
        """Spawn ``swarm launch`` for the given site set.

        Refuses if any requested site is already mid-launch (a re-
        click is a no-op) but leaves other sites' buttons free. Each
        subprocess owns its own stderr pump and updates only its own
        sites' snapshot slots.
        """
        key = tuple(sorted(sites))
        with self._lock:
            for s in sites:
                cur = self._snapshot.sites.get(s)
                if cur and cur.running:
                    return False, f"{s} is already launching"
            # Reset just the sites we're about to launch.
            for s in sites:
                self._snapshot.sites[s] = _SiteSnapshot(
                    site=s, state="submitting", running=True,
                )
            self._snapshot.seq += 1

        full_argv = [
            sys.executable, "-m", "chemgraph.academy.cli",
            "launch", "--", *argv,
        ]
        self._spawn(full_argv, kind="launch", key=key, owned_sites=set(sites))
        return True, " ".join(shlex.quote(x) for x in full_argv)

    def dispatch_message(self, argv: list[str]) -> tuple[bool, str]:
        """Spawn ``swarm inject`` (operator message to an agent).
        One at a time; refuses if another inject is still in flight.
        Site launches run concurrently with this.
        """
        with self._lock:
            if self._inject_proc is not None and self._inject_proc.poll() is None:
                return False, "an inject-message is already in flight"
            self._snapshot.inject = "in_flight"
            self._snapshot.seq += 1

        full_argv = [
            sys.executable, "-m", "chemgraph.academy.cli",
            "inject", "--", *argv,
        ]
        self._spawn(full_argv, kind="inject", key=None, owned_sites=set())
        return True, " ".join(shlex.quote(x) for x in full_argv)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn(
        self,
        argv: list[str],
        *,
        kind: str,
        key: tuple[str, ...] | None,
        owned_sites: set[str],
    ) -> None:
        # NO_COLOR so we don't have to strip ANSI on every line; the raw
        # log buffer stays cleaner too. Operators who want colored CLI
        # output still get it when running launcher in a real terminal.
        env = dict(os.environ)
        env["NO_COLOR"] = "1"
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,  # line-buffered
        )
        if kind == "launch" and key is not None:
            self._procs[key] = proc
        elif kind == "inject":
            self._inject_proc = proc
        threading.Thread(
            target=self._pump_stderr,
            args=(proc, kind, key, owned_sites),
            daemon=True,
        ).start()

    def _pump_stderr(
        self,
        proc: subprocess.Popen[str],
        kind: str,
        key: tuple[str, ...] | None,
        owned_sites: set[str],
    ) -> None:
        try:
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                line = _ANSI.sub("", raw_line.rstrip("\n"))
                if not line:
                    continue
                with self._lock:
                    # Prefix so interleaved logs from concurrent site
                    # subprocesses are distinguishable in the UI.
                    tag = f"[{'+'.join(sorted(owned_sites))}] " if owned_sites else ""
                    self._record_log_line(tag + line)
                    self._apply_state_transition(line, kind, owned_sites)
                    self._snapshot.seq += 1
        finally:
            proc.wait()
            with self._lock:
                if kind == "launch" and key is not None:
                    self._procs.pop(key, None)
                    for s in owned_sites:
                        cur = self._snapshot.sites.get(s)
                        if cur is not None:
                            cur.running = False
                            # Safety net: if the subprocess exited
                            # non-zero, any owned site still in a
                            # transient state didn't reach 'ready' and
                            # we never saw a per-site failure line.
                            # Mark it failed by exit-code so the badge
                            # doesn't get stuck.
                            if proc.returncode != 0 and cur.state not in (
                                "ready", "failed", "idle",
                            ):
                                cur.state = "failed"
                                cur.last_event = (
                                    f"launcher exited code {proc.returncode}"
                                )
                elif kind == "inject":
                    self._inject_proc = None
                    if self._snapshot.inject == "in_flight":
                        self._snapshot.inject = (
                            "dispatched" if proc.returncode == 0 else "failed"
                        )
                self._snapshot.seq += 1

    def _record_log_line(self, line: str) -> None:
        """Hold _lock when calling. Appends + caps the ring buffer."""
        self._snapshot.log_lines.append(line)
        if len(self._snapshot.log_lines) > _LOG_RING_SIZE:
            self._snapshot.log_lines = self._snapshot.log_lines[-_LOG_RING_SIZE:]

    def _apply_state_transition(
        self,
        line: str,
        kind: str,
        owned_sites: set[str],
    ) -> None:
        """Hold _lock when calling. Mutates self._snapshot in place.

        ``owned_sites`` bounds which snapshot rows this subprocess is
        allowed to touch -- concurrent per-site launches must not
        overwrite each other's state via a stray regex match.
        """
        snap = self._snapshot

        def _ours(site: str) -> bool:
            # inject subprocess has no owned sites but its regex
            # matches (INJECT_OK/FAIL) don't reference site rows.
            # Launch subprocesses may only touch sites they own.
            return not owned_sites or site in owned_sites

        m = _QSUB_RE.search(line)
        if m:
            site = m.group("site")
            if site in snap.sites and _ours(site):
                snap.sites[site].job_id = m.group("job_id")
                snap.sites[site].state = "queued"
                snap.sites[site].last_event = "qsub'd"
            return

        m = _STATE_RE.search(line)
        if m:
            site, state_letter = m.group("site"), m.group("state")
            if site in snap.sites and _ours(site):
                snap.sites[site].pbs_state = state_letter
                if state_letter == "R":
                    snap.sites[site].state = "running"
                    snap.sites[site].last_event = "running"
                elif state_letter in ("F", "E", "X"):
                    snap.sites[site].state = "failed"
                    snap.sites[site].last_event = f"PBS state {state_letter}"
                elif state_letter == "Q":
                    snap.sites[site].last_event = "queued"
            return

        m = _READY_RE.search(line)
        if m:
            site = m.group("site")
            if site in snap.sites and _ours(site):
                snap.sites[site].state = "ready"
                snap.sites[site].last_event = "ready"
            return

        m = _WAIT_RE.search(line)
        if m:
            site = m.group("site")
            if site in snap.sites and _ours(site):
                snap.sites[site].last_event = "waiting for register"
            return

        if _INJECT_OK.search(line):
            snap.inject = "dispatched"
            return

        if _INJECT_FAIL.search(line):
            snap.inject = "failed"
            return

        # Site-specific failure: match attach[SITE]: / submit[SITE]:
        # patterns that appear in launcher error messages (wait_ready
        # failed, ssh exited, job finished without running, ...).
        # Avoid downgrading sites that ALREADY reached ready -- the
        # campaign-LM may fail mid-run AFTER the launcher's job (get
        # agents registered) already succeeded.
        m = _SITE_FAIL_RE.search(line)
        if m:
            site = m.group("site")
            if site in snap.sites and _ours(site) and snap.sites[site].state != "ready":
                # Only flip on lines that read like failures (avoid
                # innocuous mentions like "[attach:aurora] waiting ...")
                if any(
                    keyword in line
                    for keyword in (
                        "exited", "failed", "not register", "without running",
                        "interrupted",
                    )
                ):
                    snap.sites[site].state = "failed"
                    # Keep the full line so the UI can show it on hover
                    # and the PBS-log link can be discovered next to it.
                    snap.sites[site].last_event = line
            return

        if _GENERIC_FAIL_RE.search(line):
            # No site captured -- leave per-site state alone. The safety
            # net in _pump_stderr (on subprocess exit with non-zero) will
            # mark any still-pending sites failed by exit code.
            return

    # ------------------------------------------------------------------
    # Stop / cleanup (called when dashboard shuts down)
    # ------------------------------------------------------------------

    def stop(self) -> None:
        procs: list[subprocess.Popen[str]] = list(self._procs.values())
        if self._inject_proc is not None:
            procs.append(self._inject_proc)
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":  # ponytail: argv rendering self-check, no live subprocess
    c = LaunchController(sites=("aurora", "crux"))
    snap = c.snapshot()
    assert snap["sites"]["aurora"]["state"] == "idle"
    assert snap["sites"]["crux"]["state"] == "idle"
    assert snap["inject"] == "idle"
    assert snap["seq"] == 0
    print("launch_handler self-check ok")
