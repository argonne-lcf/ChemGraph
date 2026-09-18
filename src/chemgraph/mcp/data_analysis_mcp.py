import math
import os
import shutil
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Initialize the MCP Server
mcp = FastMCP("ChemGraph Data Analyst")


@mcp.tool(
    name="split_cif_dataset",
    description="""
    Split a folder of CIFs file into batches.
    The batch size/number of batches is based on batch_size or num_workers.
    """,
)
def split_cif_dataset(
    input_dir: str,
    output_root: str,
    num_workers: int = 0,
    batch_size: int = 0,
) -> str:
    """
    Splits a folder of CIF files into batches based on worker count or batch size.

    Args:
        input_dir: Directory containing the source .cif files.
        output_root: Directory where batch subdirectories will be created.
        num_workers: Number of workers to distribute files across (used to calculate batch size).
        batch_size: Explicit number of files per batch.

    Returns:
        A summary string describing the outcome of the split operation.
    """
    if not os.path.exists(input_dir):
        return f"Error: Input directory '{input_dir}' does not exist."

    # Get all .cif files
    cif_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.cif')])
    total_files = len(cif_files)

    if total_files == 0:
        return "Error: No .cif files found in input directory."

    # Determine batch size logic
    if num_workers > 0:
        # Ceiling division to ensure all files are covered roughly evenly
        calculated_batch_size = math.ceil(total_files / num_workers)
    elif batch_size > 0:
        calculated_batch_size = batch_size
    else:
        return "Error: You must specify either 'num_workers' or 'batch_size'."

    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    created_batches = []

    # Process splitting
    for i in range(0, total_files, calculated_batch_size):
        batch_files = cif_files[i : i + calculated_batch_size]
        batch_index = i // calculated_batch_size

        # Create batch directory
        batch_dir_name = f"batch_{batch_index:03d}"
        batch_dir_path = os.path.join(output_root, batch_dir_name)
        os.makedirs(batch_dir_path, exist_ok=True)

        # Move files
        for f in batch_files:
            src = os.path.join(input_dir, f)
            dst = os.path.join(batch_dir_path, f)
            shutil.copy2(src, dst)

        created_batches.append(f"{batch_dir_name} ({len(batch_files)} files)")

    return (
        f"Success: Split {total_files} files into "
        f"{len(created_batches)} batches at '{output_root}'.\n"
        f"Batches created: {', '.join(created_batches)}"
    )


@mcp.tool(
    name="aggregate_simulation_results",
    description="Combine JSONL simulation records, including failures, into a CSV with original source identity.",
)
def aggregate_simulation_results(file_paths: list[str], output_csv_path: str) -> str:
    """Preserve all outcomes and provenance; fail clearly on unreadable inputs."""
    from pathlib import Path, PureWindowsPath
    from chemgraph.tools.ase_core import _resolve_path
    from chemgraph.tools.graspa_analysis import read_records, write_csv, RECORD_COLUMNS

    try:
        rows = []
        for source_file in file_paths:
            for row in read_records(source_file):
                source = row["input_structure_file"]
                path = PureWindowsPath(source) if PureWindowsPath(source).is_absolute() else Path(source)
                rows.append({**row, "cif_base_path": str(path.parent), "cif_filename": path.name,
                             "temperature": row["temperature_in_K"], "pressure": row["pressure_in_Pa"],
                             "source_file": source_file})
        if not rows:
            return "Error: No simulation records found in the provided file list."
        output = Path(_resolve_path(output_csv_path)).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output, rows, RECORD_COLUMNS + ["cif_base_path", "cif_filename", "temperature", "pressure", "source_file"])
    except (OSError, ValueError, TypeError) as exc:
        return f"Error aggregating results: {exc}"
    failures = sum(row["status"] != "success" for row in rows)
    return f"Success: Aggregated {len(rows)} records ({failures} failed) into '{output}'."


@mcp.tool(
    name="rank_mofs_performance",
    description="Rank exact adsorption conditions by uptake or working capacity, retaining full source identity.",
)
def rank_mofs_performance(
    input_csv_path: str,
    ads_pressure: float,
    ads_temp: float,
    des_pressure: float = None,
    des_temp: float = None,
    top_percentile: float = 0.10,
    min_cutoff: Optional[float] = None,
) -> str:
    """Rank complete successful repeats; return a bounded preview and a CSV."""
    from pathlib import Path
    import uuid
    from chemgraph.schemas.graspa_workflow import GraspaAnalysis
    from chemgraph.tools.ase_core import _resolve_path
    from chemgraph.tools.graspa_analysis import read_records, rank_records, write_csv, RANK_COLUMNS

    try:
        if (des_pressure is None) != (des_temp is None):
            raise ValueError("Provide both desorption pressure and temperature")
        analysis = GraspaAnalysis(
            adsorption={"temperature": ads_temp, "pressure": ads_pressure},
            desorption=({"temperature": des_temp, "pressure": des_pressure} if des_temp is not None else None),
            top_fraction=top_percentile,
        )
        ranked, excluded = rank_records(read_records(input_csv_path), analysis)
        metric = "working_capacity" if analysis.desorption else "absolute_uptake"
        if min_cutoff is not None:
            if not math.isfinite(min_cutoff):
                raise ValueError("min_cutoff must be finite")
            selected = [row for row in ranked if row[metric] >= min_cutoff]
            description = f"Values >= {min_cutoff} mol/kg"
        else:
            selected = ranked[:math.ceil(len(ranked) * top_percentile)]
            description = f"Top {top_percentile * 100:g}%"
        output = Path(_resolve_path(f"rankings_{uuid.uuid4().hex}.csv")).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output, selected, RANK_COLUMNS)
    except (OSError, ValueError, TypeError) as exc:
        return f"Error ranking results: {exc}"
    preview = "\n".join(f"{row['input_structure_file']}: {row[metric]}" for row in selected[:5])
    return (
        f"Analysis Complete ({metric}, mol/kg).\nFilter Used: {description}\n"
        f"Found {len(selected)} candidates (out of {len(ranked)} valid MOFs); "
        f"excluded {len(excluded)} incomplete/failed structures.\n"
        f"Full selected ranking: '{output}'. Preview (at most five rows):\n{preview}"
    )


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    run_mcp_server(mcp, default_port=9002)
