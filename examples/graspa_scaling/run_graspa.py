#!/usr/bin/env python
"""Run native ChemGraph H2O screening through the maintained gRASPA MCP server."""

import argparse
import asyncio
from collections import Counter
from contextlib import asynccontextmanager
import json
import math
import os
from pathlib import Path
import time
import uuid

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
    """Validate the selected workload without invoking an agent or writing files."""
    root = Path(args.output_dir).expanduser().resolve()
    settings = {
        "conditions": [{"temperature": args.ads_temp, "pressure": args.ads_pressure},
                       {"temperature": args.des_temp, "pressure": args.des_pressure}],
        "n_cycles_per_phase": args.n_cycles, "simulation_timeout": args.simulation_timeout,
    }
    if settings["conditions"][0] == settings["conditions"][1]:
        raise ValueError("Adsorption and desorption conditions must differ")
    if (root / "workflow.json").exists() or (root / "screening.json").exists():
        raise ValueError("Use a fresh output directory")
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


def prepare_query(root, sources, manifest):
    """Stage references only; server discovery resolves original source paths."""
    from chemgraph.tools.graspa_analysis import write_json

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
        f"Aggregate returned terminal JSONL files into {json.dumps(str(root / 'results.csv'))}. "
        "Use the directory reference directly; do not enumerate files or split the dataset."
    )
    write_json(root / "screening.json", {"manifest": manifest, "sources": sources})
    (root / "request.txt").write_text(query)
    return query


def result_interceptor(root, exports):
    """Save MCP result payloads without taking over tool selection or polling."""
    from mcp.types import CallToolResult, TextContent
    from chemgraph.tools.graspa_analysis import write_records

    async def save_result(request, handler):
        result = await handler(request)
        if not isinstance(result, CallToolResult) or result.isError:
            return result
        if request.name not in {"run_graspa_ensemble", "get_job_results"}:
            return result
        payload = result.structuredContent
        if payload is None:
            try:
                payload = json.loads("\n".join(block.text for block in result.content if block.type == "text"))
            except (ValueError, TypeError):
                return result
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]
        if not isinstance(payload, dict) or "error" in payload:
            return result
        key = payload.get("batch_id") or request.args.get("batch_id") or uuid.uuid4().hex
        if payload.get("status") == "submitted":
            exports[key] = payload
            return result
        rows = payload.get("results")
        if not isinstance(rows, list):
            return result
        directory = root / "tool_results"
        directory.mkdir(exist_ok=True)
        path = directory / f"{uuid.uuid4().hex}.jsonl"
        write_records(path, rows)
        summary = {key: value for key, value in payload.items() if key != "results"}
        summary.update(records_path=str(path), total_records=len(rows),
                       failed_records=sum(row.get("status") != "success" for row in rows))
        exports[key] = summary
        # Replace both representations so checkpoints also contain only paths/counts.
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(summary))],
                              structuredContent=summary)

    return save_result


def check_outcome(exports, sources, manifest):
    """Check the observed workload after execution; ranking stays in the tools."""
    from chemgraph.tools.graspa_analysis import read_records

    terminal = {"completed", "partial", "failed"}
    if not exports or any(item.get("status") not in terminal or not item.get("records_path")
                          for item in exports.values()):
        return {"status": "incomplete", "message": "Not all submitted work has terminal results"}
    records = [row for item in exports.values() for row in read_records(item["records_path"])]
    expected = Counter((source, c["temperature"], c["pressure"])
                       for source in sources for c in manifest["conditions"])
    actual = Counter((row["input_structure_file"], row["temperature_in_K"], row["pressure_in_Pa"])
                     for row in records)
    failed = sum(row["status"] != "success" for row in records)
    status = "incomplete" if actual != expected else "failed" if failed else "completed"
    return {"status": status, "total_records": len(records), "failed_records": failed,
            "records_paths": [item["records_path"] for item in exports.values()]}


@asynccontextmanager
async def ready_session(client, timeout):
    """Retry startup only; keep the ready session open for the entire run."""
    deadline = asyncio.get_running_loop().time() + timeout
    required = {"run_graspa_ensemble", "check_job_status", "get_job_results"}
    print(f"Waiting for MCP (startup timeout: {timeout:g}s)", flush=True)
    while True:
        connected = False
        try:
            async with asyncio.timeout_at(deadline) as startup:
                async with client.session("graspa") as session:
                    names = {tool.name for tool in (await session.list_tools()).tools}
                    connected = True
                    startup.reschedule(None)
                    missing = required - names
                    if missing:
                        raise ValueError(f"Use graspa_mcp_hpc; missing tools: {sorted(missing)}")
                    print("MCP ready", flush=True)
                    yield session
                    return
        except Exception as exc:
            if connected:
                raise
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"MCP did not become ready within {timeout:g}s") from exc
            print(f"Waiting for MCP: {type(exc).__name__}", flush=True)
            await asyncio.sleep(min(1, remaining))


async def run(args, root, sources, manifest):
    from progress import ProgressLogger
    from chemgraph.agent.llm_agent import ChemGraph
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langchain_core.tools import StructuredTool
    from chemgraph.mcp.data_analysis_mcp import aggregate_simulation_results, rank_mofs_performance
    from chemgraph.tools.graspa_analysis import write_json

    exports = {}
    client = MultiServerMCPClient({"graspa": {
        "transport": "streamable_http", "url": args.mcp_url,
        "timeout": 30.0, "sse_read_timeout": args.wait_timeout,
    }})
    async with ready_session(client, args.startup_timeout) as session:
        tools = await load_mcp_tools(session, tool_interceptors=[result_interceptor(root, exports)])
        query = prepare_query(root, sources, manifest)
        agent = ChemGraph(
            model_name=args.model, base_url=args.base_url, workflow_type="graspa_mcp", tools=tools,
            data_tools=[StructuredTool.from_function(function) for function in
                        (aggregate_simulation_results, rank_mofs_performance)],
            return_option="state", enable_memory=False, log_dir=str(root), recursion_limit=args.recursion_limit,
        )
        state = await agent.run(query, config={"callbacks": [ProgressLogger()]})
        response = state["messages"][-1]
        (root / "response.txt").write_text(str(response.get("content", "")) + "\n")
        outcome = check_outcome(exports, sources, manifest)
        ranking_done = any(message.get("type") == "tool" and message.get("name") == "rank_mofs_performance"
                           and str(message.get("content", "")).startswith("Analysis Complete")
                           for message in state["messages"])
        if outcome["status"] == "completed" and not ranking_done:
            outcome.update(status="incomplete", message="The analyst did not complete the ranking tool")
        write_json(root / "outcome.json", outcome)
        print(json.dumps(outcome, indent=2), flush=True)
        return 0 if outcome["status"] == "completed" else 1


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
    parser.add_argument("--startup-timeout", type=finite_positive, default=300,
                        help="Seconds to wait for the MCP handshake and required tools")
    parser.add_argument("--wait-timeout", type=finite_positive, default=9900)
    parser.add_argument("--recursion-limit", type=int, default=100)
    parser.add_argument("--model", default="alcf:openai/gpt-oss-120b")
    parser.add_argument("--base-url", default=os.environ.get("CG_BASE_URL") or None)
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print counts; no server, LLM, or writes")
    args = parser.parse_args(argv)
    if args.n_cycles <= 0 or args.limit < 0 or args.recursion_limit <= 0:
        parser.error("--n-cycles and --recursion-limit must be positive; --limit must be nonnegative")
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
