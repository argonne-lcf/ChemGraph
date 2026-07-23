"""Milestone 1 verification: mofforge <-> ChemGraph local integration.

Runs the full edge-side chain on a laptop, with no LLM required:

    1. Load mofforge's MCP tools through ChemGraph's MCP client
       (proves the two packages are wire-compatible over stdio).
    2. Build a MOF with the pormake backend (dia topology, N109 node + E41 edge).
    3. Hand the resulting CIF to ChemGraph's own ``run_ase`` for a MACE
       single-point energy.
    4. Validate the structure with ``mofforge_validate``.

This is the deterministic counterpart to the agent-driven run described in
``README.md`` (which exercises the same tools through an LLM).

Prerequisites
-------------
    pip install -e .                                    # ChemGraph
    pip install -e "/path/to/mofforge[mcp,chem,build]"  # mofforge + pormake

Environment
-----------
    MOFFORGE_LOG_DIR    base dir for mofforge output (CIFs); also forwarded to
                        the stdio MCP subprocess by chemgraph.cli.mcp_utils.
    CHEMGRAPH_LOG_DIR   base dir for ChemGraph output (energy JSON).

Run
---
    export MOFFORGE_LOG_DIR=/tmp/mofforge_out
    export CHEMGRAPH_LOG_DIR=/tmp/mofforge_out
    python scripts/mofforge_example/verify_local_integration.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _pormake_bb_dir() -> Path:
    """Locate pormake's shipped building-block (xyz) directory."""
    import pormake as pm

    return Path(pm.__file__).parent / "database" / "bbs"


def step_1_load_mcp_tools() -> None:
    """Load mofforge's MCP tools through ChemGraph's MCP client (stdio)."""
    from chemgraph.cli.mcp_utils import load_mcp_tools_from_config

    tools = load_mcp_tools_from_config(
        command="mofforge-mcp --transport stdio",
        server_name="mofforge",
        verbose=False,
    )
    assert tools, "mofforge MCP server returned no tools"
    names = sorted(t.name for t in tools)
    assert "mofforge_build" in names, names
    assert "mofforge_validate" in names, names
    print(f"[1] Loaded {len(names)} mofforge MCP tools: {names}")


def step_2_build_mof() -> str:
    """Build a dia-topology MOF with pormake; return the CIF path."""
    from mofforge.mcp import _impl

    bbs = _pormake_bb_dir()
    node, edge = str(bbs / "N109.xyz"), str(bbs / "E41.xyz")

    # dia is 4-connected -> matches N109 (4 connection points). pcu is
    # 6-connected and would fail with this node.
    res = _impl.build_impl(
        topology="dia",
        backend="pormake",
        node_files=[node],
        edge_files=[edge],
        output_dir="build_test",
    )
    assert res.get("success"), f"build failed: {res}"
    cif = res["output_paths"][0]
    print(f"[2] Built {res['atoms']}-atom MOF (dia/pormake) -> {cif}")
    return cif


def step_3_mace_energy(cif: str) -> float:
    """Compute a MACE single-point energy on the built CIF via ChemGraph."""
    from chemgraph.schemas.ase_input import ASEInputSchema
    from chemgraph.tools.ase_core import run_ase_core

    params = ASEInputSchema(
        input_structure_file=cif,
        output_results_file="mof_energy.json",
        driver="energy",
        calculator={"calculator_type": "mace_mp", "model": "small", "device": "cpu"},
    )
    out = run_ase_core(params)
    assert out.get("status") == "success", f"run_ase failed: {out}"
    energy = out["single_point_energy"]
    print(f"[3] MACE single-point energy: {energy:.4f} {out['unit']}")
    return energy


def step_4_validate(cif: str) -> None:
    """Validate the structure with mofforge."""
    from mofforge.mcp import _impl

    val = _impl.validate_impl(cif)
    assert val.get("success"), f"validate errored: {val}"
    # A raw, unrelaxed pormake placement is expected to report clashes; we only
    # assert the tool ran and returned a verdict.
    print(f"[4] Validation ran (is_valid={val.get('is_valid')})")


def main() -> None:
    if not os.environ.get("MOFFORGE_LOG_DIR"):
        os.environ["MOFFORGE_LOG_DIR"] = "/tmp/mofforge_out"
    if not os.environ.get("CHEMGRAPH_LOG_DIR"):
        os.environ["CHEMGRAPH_LOG_DIR"] = os.environ["MOFFORGE_LOG_DIR"]

    step_1_load_mcp_tools()
    cif = step_2_build_mof()
    step_3_mace_energy(cif)
    step_4_validate(cif)
    print("\nOK: mofforge -> ChemGraph local integration verified.")
    print(json.dumps({"cif": cif, "log_dir": os.environ["MOFFORGE_LOG_DIR"]}, indent=2))


if __name__ == "__main__":
    main()
