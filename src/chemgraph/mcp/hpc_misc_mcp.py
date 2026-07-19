"""FastMCP tools for generic HPC run-artifact inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP


mcp = FastMCP(
    name="ChemGraph HPC Misc Tools",
    instructions="""
        You expose small, generic tools for inspecting files produced by HPC
        calculations. These tools do not run chemistry; they help agents inspect
        run artifacts without relying on simulation-specific readers.
    """,
)


@mcp.tool(
    name="inspect_json",
    description=(
        "Inspect a JSON file, a directory of JSON files, or a missing expected "
        "JSON path. Returns compact summaries and nearby JSON files when the "
        "requested path is absent."
    ),
)
def inspect_json(
    path: str,
    glob_pattern: str = "*.json",
    max_files: int = 20,
    max_preview_chars: int = 1200,
    recursive: bool = False,
) -> dict[str, Any]:
    """Inspect JSON artifacts without assuming one fixed output-file layout."""
    target = Path(path).expanduser()
    # A small model may pass a bare name for a JSON file a sibling tool wrote
    # into CHEMGRAPH_LOG_DIR. If the raw path is not a file, resolve it against
    # the log dir before falling back to the directory / nearby-files logic.
    if not target.is_file():
        from chemgraph.tools.ase_core import _resolve_existing_path

        resolved = Path(_resolve_existing_path(str(target))).expanduser()
        if resolved.is_file():
            target = resolved
    if target.is_file():
        return {
            "status": "ok",
            "kind": "file",
            "path": str(target),
            "json": _load_json_summary(
                target,
                max_preview_chars=max_preview_chars,
            ),
        }

    if target.is_dir():
        files = _json_files(
            target,
            glob_pattern=glob_pattern,
            max_files=max_files,
            recursive=recursive,
        )
        return {
            "status": "ok",
            "kind": "directory",
            "path": str(target),
            "glob_pattern": glob_pattern,
            "recursive": recursive,
            "file_count_returned": len(files),
            "files": [
                {
                    "path": str(file),
                    "json": _load_json_summary(
                        file,
                        max_preview_chars=max_preview_chars,
                    ),
                }
                for file in files
            ],
        }

    parent = target.parent
    nearby = (
        _json_files(
            parent,
            glob_pattern=glob_pattern,
            max_files=max_files,
            recursive=False,
        )
        if parent.is_dir()
        else []
    )
    return {
        "status": "not_found",
        "kind": "missing",
        "path": str(target),
        "parent_exists": parent.is_dir(),
        "nearby_json_files": [str(file) for file in nearby],
    }


def _json_files(
    directory: Path,
    *,
    glob_pattern: str,
    max_files: int,
    recursive: bool,
) -> list[Path]:
    if max_files < 1:
        return []
    iterator = directory.rglob(glob_pattern) if recursive else directory.glob(glob_pattern)
    return sorted(path for path in iterator if path.is_file())[:max_files]


def _load_json_summary(path: Path, *, max_preview_chars: int) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report file/read/parse failure.
        return {
            "status": "error",
            "error": repr(exc),
        }
    return {
        "status": "ok",
        "summary": _summarize_json(value),
        "preview": _json_preview(value, max_chars=max_preview_chars),
    }


def _summarize_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        summary: dict[str, Any] = {
            "type": "object",
            "keys": sorted(str(key) for key in value.keys())[:40],
        }
        for key in ("status", "energy", "energy_ev", "driver", "model"):
            if key in value:
                summary[key] = value[key]
        for key in ("results", "failures", "errors"):
            nested = value.get(key)
            if isinstance(nested, list):
                summary[f"{key}_count"] = len(nested)
        return summary
    if isinstance(value, list):
        return {
            "type": "array",
            "length": len(value),
            "first_item": _summarize_json(value[0]) if value else None,
        }
    return {
        "type": type(value).__name__,
        "value": value,
    }


def _json_preview(value: Any, *, max_chars: int) -> Any:
    try:
        text = json.dumps(value, sort_keys=True)
    except TypeError:
        text = repr(value)
    if len(text) <= max_chars:
        return value
    return {
        "truncated": True,
        "chars": len(text),
        "text": text[:max_chars],
    }


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    run_mcp_server(mcp, default_port=9020)
