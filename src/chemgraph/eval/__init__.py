"""ChemGraph evaluation and benchmarking module.

Provides a structured framework for evaluating LLM tool-calling
accuracy across multiple models and workflows against ground-truth
datasets.  Two judge strategies are available:

1. **LLM-as-judge** -- a separate judge LLM compares the agent's
   tool-call sequence and final answer against the ground-truth result
   using binary scoring (1 = correct, 0 = wrong).
2. **Structured-output judge** -- a deterministic judge that compares
   the agent's ``ResponseFormatter`` structured output field-by-field
   against a ground-truth ``structured_output`` dict using numeric
   tolerances and string matching (no LLM required).

The ``judge_type`` config option controls which judge(s) run:
``"llm"``, ``"structured"``, or ``"both"``.

A default ground-truth dataset (14 queries) is bundled with the
package and used automatically when no explicit dataset is provided.

Quick start::

    import asyncio
    from chemgraph.eval import ModelBenchmarkRunner, BenchmarkConfig

    config = BenchmarkConfig(
        models=["gpt-4o-mini", "gemini-2.5-flash"],
        judge_model="gpt-4o",
        judge_type="both",  # run both LLM and structured judges
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
from chemgraph.eval.structured_output_judge import (
    StructuredOutputScore,
    aggregate_structured_results,
    judge_structured_output,
)

__all__ = [
    "BenchmarkConfig",
    "GroundTruthItem",
    "JudgeScore",
    "ModelBenchmarkRunner",
    "StructuredOutputScore",
    "aggregate_judge_results",
    "aggregate_structured_results",
    "default_dataset_path",
    "generate_markdown_report",
    "judge_single_query",
    "judge_structured_output",
    "load_dataset",
    "print_summary_table",
    "write_json_report",
    "write_markdown_report",
]
