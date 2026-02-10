import math
import os
import shutil
import logging
import json
from typing import Optional

import pandas as pd

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
    description="""Reads a list of JSONL simulation files (one JSON object per line) and 
    combines them into a CSV. Extracts nested result data (uptake, T, P) and splits file paths 
    into base directory and filename.
""",
)
def aggregate_simulation_results(
    file_paths: list[str],
    output_csv_path: str,
) -> str:
    """
    Reads a provided list of specific JSONL simulation file paths and combines them into a CSV.
    Splits the absolute 'cif_path' into 'cif_base_path' and 'cif_filename'.
    """
    all_data = []

    for file_path in file_paths:
        if not file_path or not isinstance(file_path, str):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        entry = json.loads(line)

                        if entry.get("status") == "success":
                            # Extract the full path first
                            full_cif_path = entry.get('cif_path', '')

                            flat_entry = {
                                # Split the path into directory and filename
                                'cif_base_path': os.path.dirname(full_cif_path),
                                'cif_filename': os.path.basename(full_cif_path),
                                'uptake_in_mol_kg': entry.get('uptake_in_mol_kg'),
                                # Map 'temperature_in_K' -> 'temperature'
                                'temperature': entry.get('temperature_in_K'),
                                # Map 'pressure_in_Pa' -> 'pressure'
                                'pressure': entry.get('pressure_in_Pa'),
                                'source_file': file_path,
                            }
                            all_data.append(flat_entry)

                    except json.JSONDecodeError:
                        continue

        except (IOError, FileNotFoundError):
            logging.warning("Could not read file %s", file_path)
            continue

    if not all_data:
        return "Error: No valid success data found in the provided file list."

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Ensure numeric columns are actually numeric
    cols = ['uptake_in_mol_kg', 'temperature', 'pressure']
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    try:
        df.to_csv(output_csv_path, index=False)
    except IOError as e:
        return f"Error saving CSV: {str(e)}"

    return f"Success: Aggregated {len(df)} records into '{os.path.abspath(output_csv_path)}'."


@mcp.tool(
    name="rank_mofs_performance",
    description="Ranks MOFs by performance using the aggregated CSV containing split file paths.",
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
    """
    Ranks MOFs from a CSV simulation summary.

    Args:
        input_csv_path: Path to the CSV file.
        ads_pressure: Adsorption pressure (Pa).
        ads_temp: Adsorption temperature (K).
        des_pressure: Optional. Desorption pressure (Pa).
        des_temp: Optional. Desorption temperature (K).
        top_percentile: Fraction to return (e.g. 0.10 for top 10%).
        min_cutoff: Optional. Minimum value (mol/kg) to include.
    """
    if not os.path.exists(input_csv_path):
        return f"Error: CSV file '{input_csv_path}' not found."

    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

    # Check for required column
    if 'cif_filename' not in df.columns:
        return (
            "Error: CSV is missing 'cif_filename' column. "
            "Ensure it was created by the updated aggregator."
        )

    # Ensure numeric types
    for col in ['uptake_in_mol_kg', 'temperature', 'pressure']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Determine Mode: Working Capacity (WC) vs Single Uptake
    is_wc_mode = (des_pressure is not None) and (des_temp is not None)
    metric_name = "working_capacity" if is_wc_mode else "absolute_uptake"

    results = []

    # CHANGED: Group by 'cif_filename' instead of 'cif_path'
    grouped = df.groupby('cif_filename')

    for cif_name, group in grouped:
        # Helper: Robust lookup with tolerances
        def get_uptake(target_p, target_t):
            if target_p is None or target_t is None:
                return None

            # 1. Temp filter (0.2K tolerance)
            t_matches = group[abs(group['temperature'] - target_t) < 0.2]
            if t_matches.empty:
                return None

            # 2. Pressure filter (5% tolerance)
            p_matches = t_matches[
                abs(t_matches['pressure'] - target_p) < (target_p * 0.05)
            ]

            if not p_matches.empty:
                return p_matches['uptake_in_mol_kg'].mean()
            return None

        val_ads = get_uptake(ads_pressure, ads_temp)
        val_des = get_uptake(des_pressure, des_temp) if is_wc_mode else 0.0

        if is_wc_mode:
            # Mode A: Working Capacity
            if val_ads is not None and val_des is not None:
                metric_val = val_ads - val_des
                results.append(
                    {
                        "mof_name": cif_name,
                        metric_name: metric_val,
                        "uptake_ads": val_ads,
                        "uptake_des": val_des,
                        "conditions": (
                            f"Ads({ads_temp}K, {ads_pressure}Pa) -> Des({des_temp}K, {des_pressure}Pa)"
                        ),
                    }
                )
        else:
            # Mode B: Absolute Uptake
            if val_ads is not None:
                results.append(
                    {
                        "mof_name": cif_name,
                        metric_name: val_ads,
                        "conditions": (f"Point({ads_temp}K, {ads_pressure}Pa)"),
                    }
                )

    if not results:
        cond_str = f"Ads({ads_temp}K, {ads_pressure}Pa)"
        if is_wc_mode:
            cond_str += f" -> Des({des_temp}K, {des_pressure}Pa)"
        return f"Error: No valid data found for conditions: {cond_str}"

    # Create DataFrame
    res_df = pd.DataFrame(results)

    # Sort
    res_df = res_df.sort_values(by=metric_name, ascending=False)

    # Filter Strategy
    total_count = len(res_df)

    if min_cutoff is not None:
        res_df = res_df[res_df[metric_name] >= min_cutoff]
        filter_desc = f"Values >= {min_cutoff} mol/kg"
    else:
        count = max(1, int(total_count * top_percentile))
        res_df = res_df.head(count)
        filter_desc = f"Top {int(top_percentile * 100)}%"

    cols_to_show = ['mof_name', metric_name]
    output_str = res_df[cols_to_show].to_string(index=False)

    return (
        f"Analysis Complete ({'Working Capacity' if is_wc_mode else 'Absolute Uptake'}).\n"
        f"Filter Used: {filter_desc}\n"
        f"Found {len(res_df)} candidates (out of {total_count} valid MOFs).\n\n"
        f"{output_str}"
    )


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    run_mcp_server(mcp, default_port=9002)
