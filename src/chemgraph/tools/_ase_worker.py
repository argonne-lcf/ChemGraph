"""Private subprocess entry point for isolated ASE calculator execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.tools.ase_core import _run_ase_core_in_process


def _json_default(value):
    """Convert NumPy scalars and paths used in minimal ASE result payloads."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> int:
    """Read an ASE request from stdin and write its result to the given path."""
    if len(sys.argv) != 2:
        return 2

    result_path = Path(sys.argv[1])
    try:
        params = ASEInputSchema.model_validate_json(sys.stdin.read())
        result = _run_ase_core_in_process(params)
    except Exception as exc:
        result = {
            "status": "failure",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

    result_path.write_text(
        json.dumps(result, default=_json_default), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
