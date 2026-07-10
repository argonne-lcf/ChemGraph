"""Small JSON/JSONL artifact helpers for academy_sim demos."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temporary file and replace the target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding='utf-8',
    )
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def emit_event(
    run_dir: Path,
    *,
    event: str,
    run_id: str,
    graph: str,
    payload: dict[str, Any],
) -> None:
    """Write a lightweight event record for demo inspection."""

    append_jsonl(
        run_dir / 'events.jsonl',
        {
            'event': event,
            'graph': graph,
            'payload': payload,
            'run_id': run_id,
            'time': time.time(),
        },
    )


def write_status(
    run_dir: Path,
    *,
    run_id: str,
    graph: str,
    status: dict[str, Any],
) -> None:
    """Write per-graph status for demos."""

    write_json_atomic(
        run_dir / 'status' / f'{graph}.json',
        {
            'graph': graph,
            'run_id': run_id,
            'status': status,
            'updated_at': time.time(),
        },
    )
