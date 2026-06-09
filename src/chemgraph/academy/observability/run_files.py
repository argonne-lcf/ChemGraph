from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

__all__ = [
    'append_jsonl',
    'read_json_file',
    'read_jsonl',
    'write_json',
    'write_json_atomic',
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
        fp.write('\n')


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.{uuid.uuid4()}.tmp')
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    tmp.replace(path)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as fp:
        fp.write(json.dumps(payload, sort_keys=True))
        fp.write('\n')


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding='utf-8') as fp:
        for line in fp:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def read_json_file(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return default
    return payload if isinstance(payload, dict) else default
