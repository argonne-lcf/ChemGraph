"""Durable per-session run manifest for cross-allocation resume.

Records each executed cost-bearing tool call (name, args, result-file path,
wall_time) and the pending next step on the shared filesystem under
``CHEMGRAPH_LOG_DIR``. Written incrementally with an atomic tmp+replace (the same
idiom as :class:`chemgraph.execution.job_tracker.JobTracker`), so a completed
step survives a walltime kill even though the LangGraph checkpointer is in-memory
and the SessionStore saves only at turn end.

This is deliberately independent of the SessionStore message layer, which drops
empty-content tool-call messages and has no column for tool arguments -- exactly
the information a resumed agent needs to continue without recomputing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


class RunManifest:
    """A durable, incrementally-written record of a session's executed steps."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._data: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "steps": [],
            "pending_next_step": None,
            "status": "running",
        }
        if self._path.is_file():
            try:
                loaded = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                # UnicodeDecodeError (a ValueError) covers a file corrupted with
                # non-UTF-8 bytes: the writer always emits pure ASCII, so this only
                # fires on external tampering, but the resume path must still not
                # crash on it.
                logger.warning("Could not load manifest %s: %s", self._path, exc)
            else:
                # Guard against valid-but-wrong-shape JSON (tampering, a schema
                # drift, or a truncation that still parses): every reader assumes
                # ``_data["steps"]`` is a list, so a bad shape must not reach
                # them. Fall back to the fresh default so the
                # resume this file exists to enable still proceeds.
                if self._is_valid_shape(loaded):
                    self._data = loaded
                else:
                    logger.warning(
                        "Ignoring manifest %s: unexpected shape or schema "
                        "version; starting fresh",
                        self._path,
                    )

    @staticmethod
    def _is_valid_shape(data: Any) -> bool:
        """True if *data* is a manifest this version can safely read."""
        return (
            isinstance(data, dict)
            and data.get("schema_version") == _SCHEMA_VERSION
            and isinstance(data.get("steps"), list)
        )

    # ------------------------------------------------------------------
    # persistence (atomic tmp + replace, mirrors JobTracker._save)
    # ------------------------------------------------------------------
    def _flush(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(self._data, indent=2, default=str), encoding="utf-8"
            )
            tmp.replace(self._path)
        except (TypeError, ValueError, OSError) as exc:
            # Best-effort metadata, never crash the run. Widened beyond OSError to
            # match JobTracker._save: json.dumps raises TypeError/ValueError on
            # content default=str cannot coerce (e.g. non-str dict keys), and that
            # must not propagate out of a write API called mid-tool-call.
            logger.warning("Could not persist manifest %s: %s", self._path, exc)
            try:
                tmp.unlink()
            except OSError:
                pass

    # ------------------------------------------------------------------
    # write API
    # ------------------------------------------------------------------
    def record_step_start(self, tool: str, args: dict) -> int:
        """Append a step in ``running`` state and return its 1-based index."""
        idx = len(self._data["steps"]) + 1
        self._data["steps"].append(
            {
                "index": idx,
                "tool": tool,
                "args": _jsonable(args),
                "status": "running",
                "result_file": None,
                "wall_time": None,
            }
        )
        self._flush()
        return idx

    def record_step_end(
        self,
        idx: int,
        *,
        result_file: Optional[str] = None,
        wall_time: Optional[float] = None,
        status: str = "done",
    ) -> None:
        """Mark step *idx* finished with its result-file path and wall time."""
        for step in self._data["steps"]:
            if isinstance(step, dict) and step.get("index") == idx:
                step.update(
                    result_file=result_file, wall_time=wall_time, status=status
                )
                break
        self._flush()

    def set_pending(self, tool: str, args: dict, reason: str = "") -> None:
        """Record the un-executed next step so a resume knows where to continue."""
        self._data["pending_next_step"] = {
            "tool": tool,
            "args": _jsonable(args),
            "reason": reason,
        }
        self._flush()

    def set_status(self, status: str) -> None:
        self._data["status"] = status
        self._flush()

    def clear_pending(self) -> None:
        """Drop the pending-next-step marker (one flush)."""
        self._data["pending_next_step"] = None
        self._flush()

    def mark_running(self) -> None:
        """Reset to a clean in-progress state: clear pending + status='running'.

        Called after a genuinely-completed (non-capped, non-error) step so a
        successful resume does not keep rendering a stale PENDING block or a
        'capped' status left over from an earlier cap. One flush for both.
        """
        self._data["pending_next_step"] = None
        self._data["status"] = "running"
        self._flush()

    # ------------------------------------------------------------------
    # read / render
    # ------------------------------------------------------------------
    def render_for_context(self) -> str:
        """Render a compact, untruncated block to append to a resume prompt.

        Reads every step defensively: ``_is_valid_shape`` guarantees ``steps`` is
        a list, but not that each element is a well-formed dict. A malformed step
        (a null, or one missing ``status``/``index``) is skipped so it
        cannot raise, since this method runs unguarded on the resume path and
        a crash here would defeat the resume this manifest exists to enable.
        """
        done = [
            s
            for s in self._data["steps"]
            if isinstance(s, dict) and s.get("status") == "done"
        ]
        lines = ["=== Run manifest (completed work - do NOT recompute) ==="]
        for s in done:
            raw = s.get("args")
            a = raw if isinstance(raw, dict) else {}
            lines.append(
                f"[{s.get('index', '?')}] {s.get('tool', '?')} "
                f"driver={a.get('driver', '?')} calc={_calc_name(a)} "
                f"-> result={s.get('result_file', '?')} "
                f"(wall_time={s.get('wall_time', '?')}s)"
            )
        pend = self._data.get("pending_next_step")
        if isinstance(pend, dict):
            raw = pend.get("args")
            a = raw if isinstance(raw, dict) else {}
            lines.append("=== PENDING NEXT STEP (start here) ===")
            lines.append(
                f"{pend.get('tool', '?')} driver={a.get('driver', '?')} "
                f"input={a.get('input_structure_file', '?')}"
                + (f"  ({pend.get('reason')})" if pend.get("reason") else "")
            )
        return "\n".join(lines)

    @property
    def status(self) -> str:
        return self._data.get("status", "running")

    @classmethod
    def for_session(
        cls, session_store, session_id: str
    ) -> Optional["RunManifest"]:
        """Load the manifest for a session via its stored ``log_dir``."""
        try:
            sess = session_store.get_session(session_id)
        except Exception:
            return None
        if not sess or not getattr(sess, "log_dir", None):
            return None
        p = Path(sess.log_dir) / "run_manifest.json"
        return cls(p) if p.is_file() else None


def _calc_name(args: dict) -> str:
    """Extract a calculator label from a run_ase args dict (dict or model)."""
    c = args.get("calculator")
    if isinstance(c, dict):
        return c.get("calculator_type", "?")
    return getattr(c, "calculator_type", "?") if c is not None else "?"


def _jsonable(value: Any) -> Any:
    """Normalize tool args to JSON-native types before persisting.

    Tool args may nest Pydantic models (e.g. an ``ASEInputSchema`` calculator).
    ``json.dumps(..., default=str)`` would stringify those to an unparseable repr,
    so a resumed process would reload ``calc=?`` and lose the very args the
    manifest exists to preserve. Convert models to dicts here so the on-disk form
    round-trips and ``_calc_name`` still resolves after reload.
    """
    if hasattr(value, "model_dump"):  # pydantic v2
        return _jsonable(value.model_dump())
    # pydantic v1: require both a callable ``.dict`` and ``__fields__`` so a plain
    # object that merely owns a ``.dict`` attribute (e.g. a namespace whose
    # ``dict`` is a real mapping) is not mistaken for a model and called.
    if (
        not isinstance(value, dict)
        and callable(getattr(value, "dict", None))
        and hasattr(value, "__fields__")
    ):
        return _jsonable(value.dict())
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
