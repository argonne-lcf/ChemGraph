#!/usr/bin/env python
"""Run native ChemGraph H2O screening through the maintained gRASPA MCP server."""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import time

REFERENCE = "/lus/flare/projects/IQC/thang/ChemGraph_parsl/weak_scaling_rerun/random_sampling/512_nodes/cif_files"


def pressure(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Expected a finite nonnegative pressure")
    return number


def finite_positive(value):
    number = pressure(value)
    if number == 0:
        raise argparse.ArgumentTypeError("Expected a positive value")
    return number


def prepare(args):
    """Validate a new workload, or load the frozen manifest without rediscovery."""
    root = Path(args.output_dir).expanduser().resolve()
    settings = {
        "conditions": [{"temperature": args.ads_temp, "pressure": args.ads_pressure},
                       {"temperature": args.des_temp, "pressure": args.des_pressure}],
        "n_cycles_per_phase": args.n_cycles, "simulation_timeout": args.simulation_timeout,
    }
    if settings["conditions"][0] == settings["conditions"][1]:
        raise ValueError("Adsorption and desorption conditions must differ")
    if args.resume:
        saved = json.loads((root / "screening.json").read_text())
        if any(saved["manifest"][key] != value for key, value in settings.items()):
            raise ValueError("Resume requires the original simulation settings")
        return root, saved["sources"], saved["manifest"]
    if (root / "workflow.json").exists() or (root / "screening.json").exists():
        raise ValueError("Use a fresh output directory or --resume")
    if args.cifs:
        sources = [Path(path).expanduser().resolve() for path in args.cifs]
    else:
        directory = Path(args.input_dir).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Not a CIF directory: {directory}")
        sources = [path.resolve() for path in sorted(directory.iterdir())
                   if path.is_file() and path.suffix.lower() == ".cif"]
    if args.limit:
        sources = sources[:args.limit]
    if not sources or any(not path.is_file() or path.suffix.lower() != ".cif" for path in sources):
        raise ValueError("Inputs must contain existing CIF files")
    manifest = {
        **settings, "structures": len(sources), "simulations": 2 * len(sources),
        "output_directory": str(root), "top_fraction": 0.2,
        "workers_per_node": os.environ.get("CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE", "9"),
    }
    return root, [str(path) for path in sources], manifest


def prepare_query(root, sources, manifest, resume):
    """Stage references only; graph discovery resolves links to original sources."""
    from chemgraph.tools.graspa_analysis import write_json

    if resume:
        return (root / "request.txt").read_text()
    root.mkdir(parents=True, exist_ok=True)
    inputs = root / "inputs"
    inputs.mkdir()
    for index, source in enumerate(sources):
        (inputs / f"{index:06d}.cif").symlink_to(source)
    query = (
        f"Screen all CIFs in the shared input directory {json.dumps(str(inputs))} for water harvesting. "
        "Create exactly one ensemble task with H2O and both conditions. "
        f"Use these conditions in order: adsorption, desorption: {json.dumps(manifest['conditions'])}. "
        f"Use {manifest['n_cycles_per_phase']} cycles EACH for initialization and production. "
        f"Set timeout_seconds to {json.dumps(manifest['simulation_timeout'])}. "
        f"Use output_directory {json.dumps(str(root / 'simulations'))}. "
        "Rank adsorption minus desorption uptake and select the top 20% of complete successful structures. "
        "Use the directory reference directly; do not enumerate files or split the dataset."
    )
    write_json(root / "screening.json", {"manifest": manifest, "sources": sources})
    (root / "request.txt").write_text(query)
    return query


def request_contract(root, sources, manifest):
    """Freeze CLI settings independently of the model's interpretation."""
    from chemgraph.schemas.graspa_workflow import GraspaRequestContract

    return GraspaRequestContract(
        request={"input_structures": str(root / "inputs"), "adsorbate": "H2O",
                 "conditions": manifest["conditions"], "n_cycles": manifest["n_cycles_per_phase"],
                 "timeout_seconds": manifest["simulation_timeout"], "output_directory": str(root / "simulations")},
        sources=sources,
        analysis={"adsorption": manifest["conditions"][0], "desorption": manifest["conditions"][1],
                  "top_fraction": manifest["top_fraction"]},
    )


async def run(args, root, sources, manifest):
    from chemgraph.agent.llm_agent import ChemGraph
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    query = prepare_query(root, sources, manifest, args.resume)
    client = MultiServerMCPClient({"graspa": {
        "transport": "streamable_http", "url": args.mcp_url,
        "timeout": 30.0, "sse_read_timeout": args.wait_timeout,
    }})
    async with client.session("graspa") as session:
        tools = await load_mcp_tools(session)
        required = {"run_graspa_ensemble", "check_job_status", "get_job_results"}
        missing = required - {tool.name for tool in tools}
        if missing:
            raise ValueError(f"Use graspa_mcp_hpc; missing tools: {sorted(missing)}")
        agent = ChemGraph(
            model_name=args.model, base_url=args.base_url, workflow_type="graspa_mcp", tools=tools,
            graspa_options={"run_directory": str(root), "poll_interval_seconds": args.poll_interval,
                            "wait_timeout_seconds": args.wait_timeout, "resume": args.resume,
                            "request_contract": request_contract(root, sources, manifest)},
            return_option="state", enable_memory=False, log_dir=str(root), recursion_limit=12,
        )
        state = await agent.run(query)
        print(json.dumps(state["analysis"], indent=2), flush=True)
        return 0 if state["workflow_status"] == "completed" else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input-dir")
    source.add_argument("--cifs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mcp-url", default=os.environ.get("GRASPA_MCP_URL", "http://127.0.0.1:9001/mcp/"))
    parser.add_argument("--ads-temp", type=finite_positive, default=298.0)
    parser.add_argument("--des-temp", type=finite_positive, default=298.0)
    parser.add_argument("--ads-pressure", type=pressure, default=960.0)
    parser.add_argument("--des-pressure", type=pressure, default=320.0)
    parser.add_argument("--n-cycles", type=int, default=2_000_000)
    parser.add_argument("--limit", type=int, default=int(os.environ.get("CG_LIMIT", "0")), help="First N sorted CIFs; 0 uses all")
    parser.add_argument("--simulation-timeout", type=finite_positive)
    parser.add_argument("--wait-timeout", type=finite_positive, default=9900)
    parser.add_argument("--poll-interval", type=finite_positive, default=30)
    parser.add_argument("--model", default="alcf:openai/gpt-oss-120b")
    parser.add_argument("--base-url", default=os.environ.get("CG_BASE_URL") or None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print counts; no server, LLM, or writes")
    args = parser.parse_args(argv)
    if args.n_cycles <= 0 or args.limit < 0:
        parser.error("--n-cycles must be positive and --limit must be nonnegative")
    if not args.input_dir and not args.cifs:
        args.input_dir = REFERENCE
    return args


def main():
    args = parse_args()
    root, sources, manifest = prepare(args)
    print(json.dumps(manifest, indent=2), flush=True)
    if args.dry_run:
        return 0
    started, status = time.monotonic(), 1
    try:
        status = asyncio.run(run(args, root, sources, manifest))
        return status
    finally:
        if root.is_dir():
            from chemgraph.tools.graspa_analysis import write_json
            write_json(root / "timing.json", {"wall_seconds": time.monotonic() - started, "exit_status": status})


if __name__ == "__main__":
    raise SystemExit(main())
