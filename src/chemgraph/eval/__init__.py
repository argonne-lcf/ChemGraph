"""ChemGraph evaluation and benchmarking module.

Provides a structured framework for evaluating LLM tool-calling
accuracy across multiple models and workflows against ground-truth
datasets using an **LLM-as-judge** strategy: a separate judge LLM
compares the agent's final answer against the ground-truth result
using binary scoring (1 = correct, 0 = wrong).

A default ground-truth dataset (14 queries) is bundled with the
package and used automatically when no explicit dataset is provided.

Quick start::

    import asyncio
    from chemgraph.eval import ModelBenchmarkRunner, BenchmarkConfig

    config = BenchmarkConfig(
        models=["gpt-4o-mini", "gemini-2.5-flash"],
        judge_model="gpt-4o",
    )
    runner = ModelBenchmarkRunner(config)
    results = asyncio.run(runner.run_all())
    runner.report()
"""

from chemgraph.eval.config import BenchmarkConfig
from chemgraph.eval.datasets import GroundTruthItem, default_dataset_path, load_dataset
from chemgraph.eval.llm_judge import (
    JudgeScore,
    aggregate_judge_results,
    judge_single_query,
)
from chemgraph.eval.reporter import (
    generate_markdown_report,
    print_summary_table,
    write_json_report,
    write_markdown_report,
)
from chemgraph.eval.runner import ModelBenchmarkRunner

__all__ = [
    "BenchmarkConfig",
    "GroundTruthItem",
    "JudgeScore",
    "ModelBenchmarkRunner",
    "aggregate_judge_results",
    "default_dataset_path",
    "generate_markdown_report",
    "judge_single_query",
    "load_dataset",
    "print_summary_table",
    "write_json_report",
    "write_markdown_report",
]
