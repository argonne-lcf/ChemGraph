"""Verify the local mofforge -> ChemGraph FairChem/UMA integration.

Runs the full edge-side chain on a laptop, with no LLM required:

    1. Load mofforge's MCP tools through ChemGraph's MCP client
       (proves the two packages are wire-compatible over stdio).
    2. Build a MOF with the pormake backend (dia topology, N109 node + E41 edge).
    3. Hand the resulting CIF to the ``run_fairchem_single`` MCP tool for an
       UMA single-point energy.
    4. Validate the structure with ``mofforge_validate``.

This is the deterministic counterpart to the agent-driven run described in
``README.md`` (which exercises the same tools through an LLM).

Prerequisites
-------------
    pip install -e .                                    # ChemGraph
    pip install -e "/path/to/mofforge[mcp,chem,build]"  # mofforge + pormake
    pip install -e ".[uma]"                             # FairChem environment

Environment
-----------
    MOFFORGE_LOG_DIR  base directory for mofforge output CIFs.
    CHEMGRAPH_LOG_DIR base directory for the UMA result JSON.
    FAIRCHEM_PYTHON   Python executable with FairChem/UMA installed. Defaults
                      to the interpreter running this script.
    HF_TOKEN          Hugging Face token with access to the gated UMA model.

Run
---
    export MOFFORGE_LOG_DIR=/tmp/mofforge_out
    export CHEMGRAPH_LOG_DIR=/tmp/mofforge_out
    export FAIRCHEM_PYTHON=/path/to/fairchem-env/bin/python
    python scripts/mofforge_example/verify_local_integration.py --device cpu
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_FAIRCHEM_MODULE = "chemgraph.mcp.fairchem_mcp_hpc"
_FAIRCHEM_MODEL = "uma-s-1p1"
_FAIRCHEM_TASK = "odac"

_ENV_NAMES = {
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PYTHONPATH",
    "XDG_CACHE_HOME",
}
_ENV_PREFIXES = (
    "CHEMGRAPH_",
    "HF_",
    "CUDA_",
    "ZE_",
    "OMP_",
)


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


def _resolve_python(value: str | None) -> str:
    candidate = value or os.environ.get("FAIRCHEM_PYTHON") or sys.executable
    resolved = shutil.which(candidate)
    if resolved is None:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            resolved = str(path)
    if resolved is None:
        raise ValueError(f"FairChem Python executable not found: {candidate!r}")
    return str(Path(resolved).resolve())


def _fairchem_environment() -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if name in _ENV_NAMES or name.startswith(_ENV_PREFIXES)
    }
    env["CHEMGRAPH_EXECUTION_BACKEND"] = "local"
    env["COMPUTE_SYSTEM"] = "local"
    return env


def _result_dict(result: Any) -> dict[str, Any]:
    """Normalize the LangChain MCP adapter's structured/text result."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            return parsed
    if isinstance(result, list):
        for block in result:
            text = block.get("text") if isinstance(block, dict) else None
            if text:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
    raise TypeError(f"Unexpected run_fairchem_single result: {result!r}")


async def _invoke_run_fairchem(
    tools: list[Any],
    *,
    cif: str,
    output_file: str,
    device: str,
) -> dict[str, Any]:
    by_name = {tool.name: tool for tool in tools}
    if "run_fairchem_single" not in by_name:
        raise RuntimeError(
            "FairChem MCP server did not advertise run_fairchem_single"
        )
    result = await by_name["run_fairchem_single"].ainvoke(
        {
            "params": {
                "input_structure_file": str(Path(cif).resolve()),
                "output_result_file": str(Path(output_file).resolve()),
                "driver": "energy",
                "model_name": _FAIRCHEM_MODEL,
                "task_name": _FAIRCHEM_TASK,
                "device": device,
            }
        }
    )
    return _result_dict(result)


async def step_3_uma_energy(
    cif: str,
    *,
    fairchem_python: str,
    device: str,
) -> float:
    """Compute an UMA energy through the real FairChem MCP tool."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    output_file = str(
        Path(os.environ["CHEMGRAPH_LOG_DIR"]).resolve() / "mof_uma_energy.json"
    )
    client = MultiServerMCPClient(
        {
            "fairchem": {
                "transport": "stdio",
                "command": fairchem_python,
                "args": [
                    "-u",
                    "-m",
                    _FAIRCHEM_MODULE,
                    "--transport",
                    "stdio",
                ],
                "env": _fairchem_environment(),
            }
        }
    )
    async with contextlib.AsyncExitStack() as stack:
        session = await stack.enter_async_context(client.session("fairchem"))
        tools = await load_mcp_tools(session)
        out = await _invoke_run_fairchem(
            tools,
            cif=cif,
            output_file=output_file,
            device=device,
        )

    assert out.get("status") == "success", f"run_fairchem_single failed: {out}"
    energy = out["single_point_energy"]
    print(
        f"[3] UMA ({_FAIRCHEM_MODEL}/{_FAIRCHEM_TASK}) single-point energy: "
        f"{energy:.4f} {out['unit']}"
    )
    return energy


def step_4_validate(cif: str) -> None:
    """Validate the structure with mofforge."""
    from mofforge.mcp import _impl

    val = _impl.validate_impl(cif)
    assert val.get("success"), f"validate errored: {val}"
    # A raw, unrelaxed pormake placement is expected to report clashes; we only
    # assert the tool ran and returned a verdict.
    print(f"[4] Validation ran (is_valid={val.get('is_valid')})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fairchem-python",
        default=None,
        help="Python with FairChem installed; defaults to FAIRCHEM_PYTHON.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="UMA inference device (default: cpu).",
    )
    return parser


async def amain(args: argparse.Namespace) -> None:
    if not os.environ.get("MOFFORGE_LOG_DIR"):
        os.environ["MOFFORGE_LOG_DIR"] = "/tmp/mofforge_out"
    if not os.environ.get("CHEMGRAPH_LOG_DIR"):
        os.environ["CHEMGRAPH_LOG_DIR"] = os.environ["MOFFORGE_LOG_DIR"]

    step_1_load_mcp_tools()
    cif = step_2_build_mof()
    await step_3_uma_energy(
        cif,
        fairchem_python=_resolve_python(args.fairchem_python),
        device=args.device,
    )
    step_4_validate(cif)
    print("\nOK: mofforge -> ChemGraph local integration verified.")
    print(json.dumps({"cif": cif, "log_dir": os.environ["MOFFORGE_LOG_DIR"]}, indent=2))


def main() -> None:
    asyncio.run(amain(build_parser().parse_args()))


if __name__ == "__main__":
    main()
