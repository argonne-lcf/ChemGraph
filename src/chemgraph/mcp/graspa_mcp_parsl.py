import os
from pathlib import Path
import asyncio
import json

import parsl
from parsl import python_app

import uvicorn
from mcp.server.fastmcp import FastMCP

from chemgraph.tools.graspa_tools import run_graspa_core
from chemgraph.schemas.graspa_schema import (
    graspa_input_schema,
    graspa_input_schema_ensemble,
)


@python_app
def run_graspa_parsl_app(job: dict):
    """
    Parsl python_app wrapper that runs a single graspa simulation.

    Parameters
    ----------
    job : dict
        Dictionary compatible with `run_graspa_core`'s expected input
    """
    if isinstance(job, dict):
        params = graspa_input_schema(**job)
    elif isinstance(job, graspa_input_schema):
        params = job
    else:
        raise TypeError(
            f"run_graspa_parsl_app expected dict or run_graspa_parsl_app, got {type(job)}"
        )

    return run_graspa_core(params)


def load_parsl_config(system_name: str):
    """
    Dynamically imports and returns the Parsl config based on the system name.
    """
    system_name = system_name.lower()
    run_dir = os.getcwd()

    print(f"Initializing Parsl for system: {system_name}")

    if system_name == "polaris":
        from chemgraph.configs.polaris_parsl import get_polaris_config

        return get_polaris_config(run_dir=run_dir)

    elif system_name == "aurora":
        from chemgraph.configs.aurora_parsl import get_aurora_config

        return get_aurora_config(run_dir=run_dir)

    else:
        raise ValueError(
            f"Unknown system specified: '{system_name}'. Supported: polaris, aurora, local"
        )


# Load Parsl Config
target_system = os.getenv("COMPUTE_SYSTEM", "polaris")
parsl.load(load_parsl_config(target_system))

# Start MCP server
mcp = FastMCP(
    name="Chemistry Tools MCP",
    instructions=(
        "You expose tools for running graspa simulations and reading their results. "
        "The available tools are:\n"
        "1. run_graspa_single: run a single graspa calculation using the specified input schema.\n"
        "2. run_graspa_ensemble: run graspa calculations over all structures in a directory using Parsl.\n"
        "Guidelines:\n"
        "• Use each tool only when its input schema matches the user request.\n"
        "• Do not guess numerical values; report tool errors exactly as they occur.\n"
        "• Keep responses compact — full results are written to the output files defined in the schemas.\n"
        "• When returning paths, use absolute paths.\n"
        "• Energies are in eV and wall times are in seconds."
    ),
)

"""
@mcp.tool(
    name="run_graspa_single",
    description="Run a single graspa calculation for one input file.",
)
def run_graspa_single(graspa_input_schema: graspa_input_schema):
    return run_graspa_core(graspa_input_schema)
"""


@mcp.tool(
    name="run_graspa_ensemble",
    description="Run an ensemble of graspa calculations for multiple input files.",
)
async def run_graspa_ensemble(
    graspa_input_schema_ensemble: graspa_input_schema_ensemble,
):
    """
    Run an ensemble of graspa calculations over all structure files in a directory
    using Parsl for parallel execution.

    Parameters
    ----------
    graspa_input_schema_ensemble : graspa_input_schema_ensemble
        Input parameters for the ensemble of gRASPA calculations.
    """
    input_source = graspa_input_schema_ensemble.input_structures
    structure_files: list[Path] = []
    output_dir: Path = Path.cwd()  # Default fallback

    if isinstance(input_source, list):
        structure_files = [Path(p) for p in input_source]
        missing = [p for p in structure_files if not p.exists()]
        if missing:
            raise ValueError(f"The following input files are missing: {missing}")
        if structure_files:
            output_dir = structure_files[0].parent
    else:
        input_dir = Path(input_source)
        if not input_dir.is_dir():
            raise ValueError(f"'{input_dir}' is not a valid directory.")

        structure_files = sorted([p for p in input_dir.iterdir() if p.suffix == ".cif"])
        output_dir = input_dir
    if not structure_files:
        raise ValueError("No structure files found to simulate.")

    # Base output file name
    base_output = Path(graspa_input_schema_ensemble.output_result_file).resolve()
    base_suffix = base_output.suffix if base_output.suffix else ".log"
    base_stem = base_output.stem

    pending_tasks = []

    for struct_path in structure_files:
        mof_name = struct_path.stem
        for condition in graspa_input_schema_ensemble.conditions:
            per_struct_output = base_output.with_name(
                f"{struct_path.stem}_{base_stem}{base_suffix}"
            )
            job = {
                "input_structure_file": str(struct_path),
                "output_result_file": str(per_struct_output),
                "temperature": condition.temperature,
                "pressure": condition.pressure,
                "adsorbate": graspa_input_schema_ensemble.adsorbate,
                "n_cycles": graspa_input_schema_ensemble.n_cycles,
            }

            fut = run_graspa_parsl_app(job)
            task_meta = {
                "structure": mof_name,
                "temperature": condition.temperature,
                "pressure": condition.pressure,
            }
            pending_tasks.append((task_meta, fut))

    async def wait_for_task(struct_name, parsl_future):
        try:
            # Wrap the Parsl/Concurrent future so it becomes Awaitable
            res = await asyncio.wrap_future(parsl_future)
            return res
        except Exception as e:
            return {
                "structure": struct_name,
                "status": "failure",
                "error_type": type(e).__name__,
                "message": str(e),
            }

    # This is the magic step that lets Agent 2 start while Agent 1 waits.
    results = await asyncio.gather(
        *(wait_for_task(name, fut) for name, fut in pending_tasks)
    )
    summary_log_path = output_dir / "simulation_results.jsonl"

    success_count = 0
    with open(summary_log_path, "a") as f:
        for res in results:
            if res.get("status") == "success":
                success_count += 1
            f.write(json.dumps(res) + "\n")

    return (
        f"Ensemble execution completed. Ran {len(results)} tasks "
        f"({success_count} successful). "
        f"Detailed results appended to '{summary_log_path}'."
    )


# Start MCP server
app = mcp.streamable_http_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9001)
