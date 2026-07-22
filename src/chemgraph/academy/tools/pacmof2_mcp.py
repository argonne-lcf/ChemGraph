"""PACMOF2 partial-charge assignment as an MCP server.

Demo scaffolding: thin FastMCP wrapper around the upstream ``pacmof2``
pip package (snurr-group). When ChemGraph ships an official PACMOF2
MCP server, swap the campaign's ``mcp_servers[].command`` to point at
it and delete this file -- the tool name (``pacmof2_assign_charges``)
and schema are stable, so no campaign change beyond the command path.

Wraps the ``pacmof2.get_charges`` function so the runner can call it
via a plain ``tool_call`` node. One tool:

  pacmof2_assign_charges(cif_path, output_dir=None, identifier='_pacmof',
                         net_charge=None) -> dict

The function returns a dict with the resulting CIF path so downstream
workflow states can pick it up via ``store: {charges_cif: "$.result.charges_cif"}``.

CLI shape matches mofforge/ChemGraph: ``--transport streamable_http
--host --port``. Launch via::

    python -m swarm.tools.pacmof2_mcp \
        --transport streamable_http --host 127.0.0.1 --port <PORT>

First call downloads the HuggingFace-hosted PACMOF2 model (~500 MB) to
the user's cache. On Crux compute nodes with no external network,
pre-populate the cache once from a login node.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("chemgraph.academy.tools.pacmof2")

mcp = FastMCP(
    name="pacmof2",
    instructions=(
        "PACMOF2 (partial charges for MOFs). Runs on CPU; typical wall "
        "time is a few seconds per MOF once the model is cached. "
        "Input: an absolute CIF path. Output: a new CIF with partial "
        "charges assigned to each atom (path returned as charges_cif)."
    ),
)


_CACHED_GET_CHARGES = None


def _load_get_charges():
    """Import pacmof2.get_charges without triggering pacmof2/__init__.py's
    circular-import bug.

    pacmof2/__init__.py does ``from pacmof2.pacmof2 import ...`` while
    pacmof2/pacmof2.py does ``from . import models`` and models/__init__
    re-enters the still-loading parent. Instead we:
      1. Pre-import pacmof2.models under a synthetic parent so the
         nested imports resolve.
      2. exec pacmof2/pacmof2.py's file directly.
      3. Pull get_charges out of the resulting module.
    """
    global _CACHED_GET_CHARGES
    if _CACHED_GET_CHARGES is not None:
        return _CACHED_GET_CHARGES

    import importlib
    import importlib.util
    import sys

    # Step 1: locate the pacmof2 package on disk without importing it.
    # The pacmof2 repo layout has the package inside a same-named parent
    # dir (repo/pacmof2/pacmof2/pacmof2.py), so find_spec's search
    # location may point at the outer dir. Try both.
    spec = importlib.util.find_spec("pacmof2")
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError("pacmof2 package not installed")
    search_root = Path(spec.submodule_search_locations[0])
    candidates = [
        search_root / "pacmof2.py",
        search_root / "pacmof2" / "pacmof2.py",
    ]
    pacmof2_py = next((p for p in candidates if p.exists()), None)
    if pacmof2_py is None:
        raise ImportError(
            f"pacmof2/pacmof2.py not found; searched {[str(p) for p in candidates]}"
        )
    pkg_dir = pacmof2_py.parent

    # Step 2: register a minimal pacmof2 parent in sys.modules that the
    # submodule's `from . import models` can walk. Load models manually
    # since it doesn't have the circular-import problem.
    if "pacmof2" not in sys.modules:
        import types
        parent = types.ModuleType("pacmof2")
        parent.__path__ = [str(pkg_dir)]
        parent.__file__ = str(pkg_dir / "__init__.py")
        sys.modules["pacmof2"] = parent

        # Load models as pacmof2.models. Its own __init__ shouldn't
        # trigger further circulars since it's a leaf under the parent.
        models_spec = importlib.util.spec_from_file_location(
            "pacmof2.models",
            pkg_dir / "models" / "__init__.py",
            submodule_search_locations=[str(pkg_dir / "models")],
        )
        models_mod = importlib.util.module_from_spec(models_spec)
        sys.modules["pacmof2.models"] = models_mod
        models_spec.loader.exec_module(models_mod)
        parent.models = models_mod

    # Step 3: exec pacmof2.pacmof2 as a real submodule.
    if "pacmof2.pacmof2" not in sys.modules:
        submod_spec = importlib.util.spec_from_file_location(
            "pacmof2.pacmof2", pacmof2_py,
        )
        submod = importlib.util.module_from_spec(submod_spec)
        sys.modules["pacmof2.pacmof2"] = submod
        submod_spec.loader.exec_module(submod)

    _CACHED_GET_CHARGES = sys.modules["pacmof2.pacmof2"].get_charges
    return _CACHED_GET_CHARGES


def _resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        p = Path(output_dir)
    else:
        p = Path(os.environ.get("PACMOF2_OUTPUT_DIR", "."))
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


@mcp.tool(
    name="pacmof2_assign_charges",
    description=(
        "Assign PACMOF2 partial charges to every atom in a MOF CIF. "
        "Returns the path of the new CIF written to output_dir. "
        "Set net_charge for ionic MOFs; omit for neutral."
    ),
)
def pacmof2_assign_charges(
    cif_path: str,
    output_dir: str | None = None,
    identifier: str = "_pacmof",
    net_charge: int | None = None,
) -> dict:
    """Run PACMOF2 on one CIF, return path of the charge-assigned CIF."""
    src = Path(cif_path)
    if not src.is_absolute():
        raise ValueError(f"cif_path must be absolute; got {cif_path!r}")
    if not src.exists():
        raise FileNotFoundError(f"cif not found: {cif_path}")

    out_dir = _resolve_output_dir(output_dir)

    # pacmof2's __init__ has a real circular-import bug: it does
    # `from pacmof2.pacmof2 import ...` while pacmof2.pacmof2 does
    # `from . import models`, and models/__init__ re-enters the
    # partially-loaded parent. Bypass entirely by loading the
    # pacmof2.py file directly with importlib. The function itself
    # only uses top-level pacmof2.models (which imports cleanly on
    # its own), so this works.
    get_charges = _load_get_charges()

    if net_charge is None:
        get_charges(str(src), str(out_dir), identifier=identifier)
    else:
        get_charges(str(src), str(out_dir), identifier=identifier, net_charge=net_charge)

    # pacmof2 writes <stem><identifier>.cif into output_dir.
    charges_cif = out_dir / f"{src.stem}{identifier}.cif"
    if not charges_cif.exists():
        raise RuntimeError(
            f"pacmof2 did not produce expected file {charges_cif}. "
            f"output_dir contents: {sorted(p.name for p in out_dir.iterdir())}"
        )
    return {
        "status": "ok",
        "input_cif": str(src),
        "charges_cif": str(charges_cif),
        "identifier": identifier,
        "net_charge": net_charge,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PACMOF2 MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http", "streamable_http"],
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9012)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
