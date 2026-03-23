"""Reporting utilities for ChemGraph evaluation benchmarks.

Produces structured JSON summaries and human-readable Markdown tables
from LLM-as-judge benchmark results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def _safe_pct(value: float) -> str:
    """Format a 0-1 fraction as a percentage string."""
    return f"{value * 100:.1f}%"


# ---- JSON report ---------------------------------------------------------


def write_json_report(
    results: Dict[str, Dict[str, dict]],
    metadata: dict,
    output_path: str,
) -> str:
    """Write the full benchmark results to a JSON file.

    Parameters
    ----------
    results : dict
        Nested dict: ``{model_name: {workflow_type: {judge_aggregate + details}}}``.
    metadata : dict
        Run metadata (timestamp, config, etc.).
    output_path : str
        Destination file path.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    report = {
        "metadata": metadata,
        "results": _make_serializable(results),
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"JSON report written to {p}")
    return str(p.resolve())


# ---- Markdown report -----------------------------------------------------


def generate_markdown_report(
    results: Dict[str, Dict[str, dict]],
    metadata: dict,
) -> str:
    """Generate a Markdown comparison report with LLM-judge scores.

    Parameters
    ----------
    results : dict
        ``{model_name: {workflow_type: {"judge_aggregate": {...}, ...}}}``
    metadata : dict
        Run metadata.

    Returns
    -------
    str
        Markdown-formatted report string.
    """
    lines: List[str] = []
    lines.append("# ChemGraph Evaluation Report")
    lines.append("")

    # Metadata
    lines.append("## Run Metadata")
    lines.append("")
    for key, val in metadata.items():
        lines.append(f"- **{key}**: {val}")
    lines.append("")

    # Collect all (model, workflow) combinations
    all_workflows = sorted({wf for model_data in results.values() for wf in model_data})

    for workflow in all_workflows:
        lines.append(f"## Workflow: `{workflow}`")
        lines.append("")

        lines.append("### LLM Judge (Final Answer Accuracy)")
        lines.append("")
        header = "| Model | Queries | Correct | Accuracy | Parse Errors |"
        sep = "|---|---|---|---|---|"
        lines.append(header)
        lines.append(sep)

        for model_name, model_data in results.items():
            if workflow not in model_data:
                continue
            jagg = model_data[workflow].get("judge_aggregate")
            if not jagg:
                continue
            row = (
                f"| {model_name} "
                f"| {jagg.get('n_queries', 0)} "
                f"| {jagg.get('n_correct', 0)} "
                f"| {_safe_pct(jagg.get('accuracy', 0))} "
                f"| {jagg.get('n_parse_errors', 0)} |"
            )
            lines.append(row)

        lines.append("")

    return "\n".join(lines)


def write_markdown_report(
    results: Dict[str, Dict[str, dict]],
    metadata: dict,
    output_path: str,
) -> str:
    """Write the Markdown report to a file.

    Parameters
    ----------
    results : dict
        Benchmark results.
    metadata : dict
        Run metadata.
    output_path : str
        Destination file path.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    md = generate_markdown_report(results, metadata)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"Markdown report written to {p}")
    return str(p.resolve())


# ---- Per-model detail dumps ----------------------------------------------


def write_model_detail(
    model_name: str,
    workflow_type: str,
    raw_tool_calls: list,
    per_query_results: list,
    output_dir: str,
    judge_results: Optional[list] = None,
) -> str:
    """Write per-model raw tool calls and evaluation details.

    Parameters
    ----------
    model_name : str
        Model identifier.
    workflow_type : str
        Workflow type used.
    raw_tool_calls : list
        Raw tool-call dicts extracted from the agent.
    per_query_results : list
        Per-query evaluation result dicts.
    output_dir : str
        Output directory.
    judge_results : list, optional
        Per-query LLM judge result dicts.

    Returns
    -------
    str
        Path to the written detail file.
    """
    detail = {
        "model_name": model_name,
        "workflow_type": workflow_type,
        "raw_tool_calls": raw_tool_calls,
        "per_query_results": _make_serializable(per_query_results),
    }
    if judge_results is not None:
        detail["judge_results"] = _make_serializable(judge_results)

    safe_name = model_name.replace("/", "_").replace(":", "_")
    fname = f"{safe_name}_{workflow_type}_detail.json"
    p = Path(output_dir) / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, default=str)
    logger.info(f"Detail file written to {p}")
    return str(p.resolve())


# ---- Printing to console -------------------------------------------------


def print_summary_table(results: Dict[str, Dict[str, dict]]) -> None:
    """Print a concise LLM-judge comparison table to stdout.

    Parameters
    ----------
    results : dict
        ``{model_name: {workflow_type: {"judge_aggregate": {...}}}}``
    """
    all_workflows = sorted({wf for model_data in results.values() for wf in model_data})

    for workflow in all_workflows:
        print(f"\n{'=' * 60}")
        print(f"  Workflow: {workflow}")
        print(f"{'=' * 60}")

        header = f"  {'Model':<40} {'Judge Acc':>10}"
        print(header)
        print(f"  {'-' * 40} {'-' * 10}")

        for model_name, model_data in results.items():
            if workflow not in model_data:
                continue
            jagg = model_data[workflow].get("judge_aggregate")
            if jagg:
                j = _safe_pct(jagg.get("accuracy", 0))
                print(f"  {model_name:<40} {j:>10}")

    print()


# ---- Helpers --------------------------------------------------------------


def _make_serializable(obj):
    """Recursively convert non-serializable objects to strings."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    elif isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    else:
        return str(obj)
