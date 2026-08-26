"""Per-exchange artifact attribution for the ChemGraph Streamlit UI.

All queries in a chat share one log directory, so "newest file in the
directory" heuristics attribute artifacts from later queries (or stale
files from unrelated runs) to earlier exchanges.  Instead, the UI
snapshots the log directory before each agent run and records exactly
which files the run created or modified.  That per-exchange file list is
stored on the conversation entry and appended to a JSON manifest inside
the log directory so restored sessions keep the correct attribution.

Every function is Streamlit-free so it can be unit-tested without a
running Streamlit runtime.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "ui_artifacts.json"

# Classification buckets, in the order the UI renders them.
STRUCTURES = "structures"
IR_PLOTS = "ir_plots"
IR_SPECTRA = "ir_spectra"
IR_PEAKS = "ir_peaks"
FREQUENCY_TABLES = "frequency_tables"
MODE_TRAJECTORIES = "mode_trajectories"
TRAJECTORIES = "trajectories"
REPORTS = "reports"
IMAGES = "images"
DATA = "data"
OTHER = "other"


def snapshot_mtimes(log_dir: Optional[str]) -> dict[str, float]:
    """Return ``{relative_path: mtime}`` for every file under *log_dir*.

    Parameters
    ----------
    log_dir : str, optional
        Chat log directory.  ``None`` or a missing directory yields ``{}``.

    Returns
    -------
    dict[str, float]
        Relative file paths mapped to modification times.
    """
    if not log_dir or not os.path.isdir(log_dir):
        return {}
    base = Path(log_dir)
    snapshot: dict[str, float] = {}
    for path in base.rglob("*"):
        if not path.is_file() or path.name == MANIFEST_FILENAME:
            continue
        try:
            snapshot[path.relative_to(base).as_posix()] = path.stat().st_mtime
        except OSError:
            continue
    return snapshot


def collect_new_files(
    log_dir: Optional[str], before: dict[str, float]
) -> list[str]:
    """Return files created or modified since the *before* snapshot.

    Parameters
    ----------
    log_dir : str, optional
        Chat log directory.
    before : dict[str, float]
        Snapshot taken with :func:`snapshot_mtimes` before the run.

    Returns
    -------
    list[str]
        Relative paths of new/changed files, oldest first.
    """
    after = snapshot_mtimes(log_dir)
    changed = [
        rel
        for rel, mtime in after.items()
        if rel not in before or mtime != before[rel]
    ]
    changed.sort(key=lambda rel: after[rel])
    return changed


def classify_artifacts(files: list[str]) -> dict[str, list[str]]:
    """Group artifact paths by kind, preserving input (chronological) order.

    Parameters
    ----------
    files : list[str]
        Relative artifact paths from :func:`collect_new_files`.

    Returns
    -------
    dict[str, list[str]]
        Non-empty buckets keyed by the module-level kind constants.
    """
    kinds: dict[str, list[str]] = {}

    def _add(kind: str, rel: str) -> None:
        kinds.setdefault(kind, []).append(rel)

    for rel in files:
        name = Path(rel).name.lower()
        if name.endswith(".xyz"):
            _add(STRUCTURES, rel)
        elif name.endswith(".png") and name.startswith("ir_spectrum"):
            _add(IR_PLOTS, rel)
        elif name.endswith(".csv") and name.startswith("ir_spectrum"):
            _add(IR_SPECTRA, rel)
        elif name.endswith(".csv") and name.startswith("ir_peaks"):
            _add(IR_PEAKS, rel)
        elif name.endswith(".csv") and name.startswith("frequencies"):
            _add(FREQUENCY_TABLES, rel)
        elif fnmatch.fnmatch(name, "*_vib.*.traj"):
            _add(MODE_TRAJECTORIES, rel)
        elif name.endswith("_opt.traj"):
            # Only optimization trajectories are rendered as convergence plots;
            # matching the ``<stem>_opt.traj`` name the optimizer writes keeps an
            # unrelated .traj (e.g. an uploaded MD run) from being mislabeled.
            _add(TRAJECTORIES, rel)
        elif name.endswith(".traj"):
            _add(OTHER, rel)
        elif name.endswith((".html", ".htm")):
            _add(REPORTS, rel)
        elif name.endswith(".png"):
            _add(IMAGES, rel)
        elif name.endswith((".json", ".csv")):
            _add(DATA, rel)
        else:
            _add(OTHER, rel)
    return kinds


def append_manifest_entry(
    log_dir: Optional[str],
    query: str,
    files: list[str],
    attachments: Optional[list[str]] = None,
) -> None:
    """Append one exchange's artifact list to the log-dir manifest.

    Best-effort: any I/O failure is logged and swallowed so persistence
    problems never break the chat flow.

    Parameters
    ----------
    log_dir : str, optional
        Chat log directory.
    query : str
        User query that produced the artifacts.
    files : list[str]
        Relative artifact paths for the exchange.
    attachments : list[str], optional
        Display names of files the user attached to the query.
    """
    if not log_dir:
        return
    entries = load_manifest(log_dir)
    record: dict = {"query": query, "files": list(files)}
    if attachments:
        record["attachments"] = list(attachments)
    entries.append(record)
    try:
        os.makedirs(log_dir, exist_ok=True)
        manifest_path = Path(log_dir) / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps({"exchanges": entries}, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("Failed to write artifact manifest: %s", exc)


def load_manifest(log_dir: Optional[str]) -> list[dict]:
    """Return the manifest exchange list for *log_dir* (``[]`` on any problem).

    Parameters
    ----------
    log_dir : str, optional
        Chat log directory.

    Returns
    -------
    list[dict]
        Entries of the form ``{"query": str, "files": [str, ...]}``.
    """
    if not log_dir:
        return []
    manifest_path = Path(log_dir) / MANIFEST_FILENAME
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    exchanges = data.get("exchanges") if isinstance(data, dict) else None
    if not isinstance(exchanges, list):
        return []
    return [
        entry
        for entry in exchanges
        if isinstance(entry, dict) and isinstance(entry.get("files"), list)
    ]


def attach_artifacts_to_history(history: list[dict], log_dir: Optional[str]) -> None:
    """Attach manifest artifact lists to restored conversation entries.

    Entries and manifest records are appended in the same order, so they
    are matched positionally with the query text as a safety check; on the
    first mismatch the remaining entries are left untouched and fall back
    to legacy directory-based rendering.

    Parameters
    ----------
    history : list[dict]
        Conversation-history entries rebuilt from a stored session.
    log_dir : str, optional
        Session log directory containing the manifest.
    """
    manifest = load_manifest(log_dir)
    for entry, record in zip(history, manifest):
        if entry.get("query") != record.get("query"):
            break
        entry["artifacts"] = list(record.get("files", []))
        if record.get("attachments"):
            entry["attachments"] = list(record["attachments"])
