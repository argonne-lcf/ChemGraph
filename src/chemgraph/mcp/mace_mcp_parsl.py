import os
from pathlib import Path
import json

import parsl
from parsl import python_app

import uvicorn
from mcp.server.fastmcp import FastMCP

from parsl.config import Config
from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.utils import get_all_checkpoints

from chemgraph.tools.parsl_tools import (
    run_mace_core,
    mace_input_schema,
    mace_input_schema_ensemble,
)


@python_app
def run_mace_parsl_app(job: dict):
    """
    Parsl python_app wrapper that runs a single MACE simulation.

    Parameters
    ----------
    job : dict
        Dictionary compatible with `run_mace_core`'s expected input
        (same keys as mace_input_schema).

    Returns
    -------
    dict
        The result of `run_mace_core(job)`.
    """
    from chemgraph.tools.parsl_tools import mace_input_schema, run_mace_core

    if isinstance(job, dict):
        params = mace_input_schema(**job)
    elif isinstance(job, mace_input_schema):
        params = job
    else:
        raise TypeError(
            f"run_mace_parsl_app expected dict or mace_input_schema, got {type(job)}"
        )

    return run_mace_core(params)


mcp = FastMCP(
    name="Chemistry Tools MCP",
    instructions=(
        "You expose tools for running MACE simulations and reading their results. "
        "The available tools are:\n"
        "1. run_mace_single: run a single MACE calculation using the specified input schema.\n"
        "2. run_mace_ensemble: run MACE calculations over all structures in a directory using Parsl.\n"
        "3. extract_output_json: load simulation results from a JSON file.\n\n"
        "Guidelines:\n"
        "• Use each tool only when its input schema matches the user request.\n"
        "• Do not guess numerical values; report tool errors exactly as they occur.\n"
        "• Keep responses compact — full results are written to the output files defined in the schemas.\n"
        "• When returning paths, use absolute paths.\n"
        "• Energies are in eV and wall times are in seconds."
    ),
)


@mcp.tool(
    name="run_mace_single",
    description="Run a single MACE calculation",
)
def run_mace_single(mace_input_schema: mace_input_schema):
    return run_mace_core(mace_input_schema)


@mcp.tool(
    name="run_mace_ensemble",
    description="Run an ensemble of MACE calculations",
)
def run_mace_ensemble(mace_input_schema_ensemble: mace_input_schema_ensemble):
    """
    Run an ensemble of MACE calculations over all structure files in a directory
    using Parsl for parallel execution.

    Parameters
    ----------
    params : mace_input_schema_ensemble
        Input parameters for the ensemble of MACE calculations.

    Returns
    -------
    dict
        Summary of all jobs with minimal per-job results.
    """
    input_dir = Path(mace_input_schema_ensemble.input_structure_directory)

    if not input_dir.is_dir():
        raise ValueError(
            f"The provided structure directory '{input_dir}' does not exist or is not a directory!"
        )

    # Collect all files in the directory
    structure_files = sorted([p for p in input_dir.iterdir() if p.is_file()])
    if not structure_files:
        raise ValueError(f"No structure files found in directory '{input_dir}'")

    # Base output file name used as a pattern for per-structure outputs
    base_output = Path(mace_input_schema_ensemble.output_result_file)
    base_stem = base_output.stem
    base_suffix = base_output.suffix or ".json"

    futures = []
    for struct_path in structure_files:
        # Make per-structure output file unique
        per_struct_output = base_output.with_name(
            f"{struct_path.stem}_{base_stem}{base_suffix}"
        )

        # Build a job dict compatible with run_mace_core
        job = {
            "input_structure_file": str(struct_path),
            "output_result_file": str(per_struct_output),
            "driver": mace_input_schema_ensemble.driver,
            "model": mace_input_schema_ensemble.model,
            "device": mace_input_schema_ensemble.device,
            "temperature": mace_input_schema_ensemble.temperature,
            "pressure": mace_input_schema_ensemble.pressure,
            "fmax": mace_input_schema_ensemble.fmax,
            "steps": mace_input_schema_ensemble.steps,
            "optimizer": mace_input_schema_ensemble.optimizer,
        }

        fut = run_mace_parsl_app(job)
        futures.append((struct_path.name, str(per_struct_output), fut))

    # Gather results (this will block until all Parsl tasks are done)
    results = []
    for struct_name, out_file, fut in futures:
        try:
            res = fut.result()
            status = (
                res.get("status", "unknown") if isinstance(res, dict) else "success"
            )
            energy = res.get("single_point_energy") if isinstance(res, dict) else None
            results.append(
                {
                    "structure": struct_name,
                    "output_result_file": out_file,
                    "status": status,
                    "single_point_energy": energy,
                    "raw_result": res,  # keep full result if needed by the LLM
                }
            )
        except Exception as e:
            results.append(
                {
                    "structure": struct_name,
                    "output_result_file": out_file,
                    "status": "failure",
                    "error_type": type(e).__name__,
                    "message": str(e),
                }
            )

    return {
        "status": "success",
        "n_structures": len(structure_files),
        "results": results,
    }


@mcp.tool(
    name="extract_output_json",
    description="Load output from a JSON file.",
)
def extract_output_json(json_file: str) -> dict:
    """
    Load simulation results from a JSON file produced by run_ase.

    Parameters
    ----------
    json_file : str
        Path to the JSON file containing ASE simulation results.

    Returns
    -------
    Dict[str, Any]
        Parsed results from the JSON file as a Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


# User-specific paths and settings
run_dir = os.getcwd()
worker_init = "module use /soft/modulefiles; module load conda; conda activate /lus/grand/projects/IQC/thang/ChemGraph/env/polaris_env; export TMPDIR=/tmp"

# Load previous checkpoints
checkpoints = get_all_checkpoints(run_dir)

# Get the number of nodes:
node_file = os.getenv("PBS_NODEFILE")
with open(node_file, "r") as f:
    node_list = f.readlines()
    num_nodes = len(node_list)

# Configure Parsl
config = Config(
    executors=[
        HighThroughputExecutor(
            label="htex",
            heartbeat_period=15,
            heartbeat_threshold=120,
            worker_debug=True,
            available_accelerators=4,
            cpu_affinity="list:24-31,56-63:16-23,48-55:8-15,40-47:0-7,32-39",
            prefetch_capacity=0,
            # start_method="spawn",
            provider=LocalProvider(
                launcher=MpiExecLauncher(
                    bind_cmd="--cpu-bind", overrides="--depth=1 --ppn 1"
                ),
                worker_init=worker_init,
                nodes_per_block=num_nodes,
                init_blocks=1,
                min_blocks=0,
                max_blocks=1,
            ),
        ),
    ],
    checkpoint_files=checkpoints,
    run_dir=run_dir,
    checkpoint_mode="task_exit",
    app_cache=True,
)

# Load the Parsl configuration
parsl.load(config)

# Start MCP server
app = mcp.streamable_http_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9001)
