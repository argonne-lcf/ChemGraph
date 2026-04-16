"""CLI entry point for ChemGraph evaluation benchmarks.

Usage::

    # Quick local evaluation using a profile
    chemgraph eval --profile quick --models gpt-4o-mini --judge-model gpt-4o

    # Standard evaluation with LLM judge
    chemgraph eval --profile standard --models gpt-4o-mini gemini-2.5-flash

    # Minimal invocation (uses bundled default dataset)
    chemgraph-eval --models gpt-4o-mini --judge-model gpt-4o

    # Explicit dataset override
    chemgraph-eval \\
        --models gpt-4o-mini gemini-2.5-flash \\
        --dataset path/to/custom_ground_truth.json \\
        --judge-model gpt-4o \\
        --workflows single_agent \\
        --output-dir eval_results

    # Profile + override
    chemgraph eval --profile quick --models gpt-4o --max-queries 3
"""

import argparse
import asyncio
import sys
from typing import Optional

from chemgraph.eval.config import BenchmarkConfig
from chemgraph.eval.runner import ModelBenchmarkRunner


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add evaluation-specific arguments to an argument parser.

    This function is used by both the standalone ``chemgraph-eval``
    entry point and the ``chemgraph eval`` subcommand so that the
    argument interface is consistent.
    """
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="LLM model names to evaluate.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help=(
            "LLM model name for the judge. Required when "
            "--judge-type is 'llm' or 'both'."
        ),
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help=(
            "Evaluation profile name from config.toml [eval.profiles.*] "
            "(e.g. 'quick', 'standard'). Requires --config. "
            "CLI arguments override profile values."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Path to ground-truth JSON file. "
            "Defaults to the bundled dataset shipped with the package."
        ),
    )
    parser.add_argument(
        "--workflows",
        nargs="+",
        default=None,
        help="Workflow types to test (default: single_agent).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Output directory for results (default: eval_results).",
    )
    parser.add_argument(
        "--report",
        choices=["json", "markdown", "console", "all"],
        default="all",
        help="Report format (default: all).",
    )
    parser.add_argument(
        "--no-structured-output",
        action="store_true",
        help="Disable structured output on the agent.",
    )
    parser.add_argument(
        "--judge-type",
        type=str,
        choices=["llm", "structured", "both"],
        default=None,
        help=(
            "Judge strategy: 'llm' (LLM-as-judge), 'structured' "
            "(deterministic structured-output comparison), or 'both' "
            "(run both judges). Default: llm."
        ),
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=None,
        help="Max LangGraph recursion steps per query (default: 50).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Max number of queries to evaluate (0 = all, default: all).",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Optional tags for the run metadata.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from per-query checkpoint files, skipping "
            "already-completed (model, workflow, query) combinations."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a TOML configuration file (e.g. config.toml). "
            "Provides model base_url, argo_user, and eval profiles."
        ),
    )


def _resolve_profile(args: argparse.Namespace) -> Optional[str]:
    """Resolve the eval profile name from CLI args and config file.

    If ``--profile`` is explicitly set, use it.  Otherwise, if
    ``--config`` is provided and the config file defines
    ``[eval] default_profile``, use that as the profile name.

    Returns ``None`` if no profile should be used.
    """
    if args.profile:
        return args.profile

    if args.config:
        import toml
        from pathlib import Path

        p = Path(args.config)
        if p.exists():
            with open(p) as fh:
                raw = toml.load(fh)
            default = raw.get("eval", {}).get("default_profile")
            if default:
                profiles = raw.get("eval", {}).get("profiles", {})
                if default in profiles:
                    return default

    return None


def build_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Build a ``BenchmarkConfig`` from parsed CLI arguments.

    Handles both profile-based and explicit-argument construction.
    When ``--config`` is provided without ``--profile``, the
    ``[eval] default_profile`` from the config file is used
    automatically if it exists.
    """
    profile = _resolve_profile(args)

    if profile:
        # Profile mode: requires --config
        config_file = args.config
        if not config_file:
            print(
                "Error: --config is required when using --profile.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Collect CLI overrides (None values will be skipped by from_profile)
        overrides = {
            "output_dir": args.output_dir,
            "tags": args.tags or None,
        }
        if args.dataset is not None:
            overrides["dataset"] = args.dataset
        if args.workflows is not None:
            overrides["workflow_types"] = args.workflows
        if args.judge_model is not None:
            overrides["judge_model"] = args.judge_model
        if args.recursion_limit is not None:
            overrides["recursion_limit"] = args.recursion_limit
        if args.max_queries is not None:
            overrides["max_queries"] = args.max_queries
        if args.no_structured_output:
            overrides["structured_output"] = False
        if args.judge_type is not None:
            overrides["judge_type"] = args.judge_type
        if args.resume:
            overrides["resume"] = True

        config = BenchmarkConfig.from_profile(
            profile_name=profile,
            models=args.models,
            config_file=config_file,
            **overrides,
        )
    else:
        # Explicit mode: dataset defaults to the bundled ground truth
        # when --dataset is not provided.
        kwargs: dict = {
            "models": args.models,
            "workflow_types": args.workflows or ["single_agent"],
            "output_dir": args.output_dir,
            "structured_output": not args.no_structured_output,
            "recursion_limit": args.recursion_limit or 50,
            "tags": args.tags or [],
            "max_queries": args.max_queries or 0,
            "config_file": args.config,
            "judge_type": args.judge_type or "llm",
            "resume": args.resume,
        }
        if args.judge_model is not None:
            kwargs["judge_model"] = args.judge_model
        if args.dataset is not None:
            kwargs["dataset"] = args.dataset

        config = BenchmarkConfig(**kwargs)

    return config


def run_eval(args: argparse.Namespace) -> None:
    """Execute an evaluation benchmark from parsed CLI arguments."""
    config = build_config_from_args(args)
    runner = ModelBenchmarkRunner(config)

    print("ChemGraph Evaluation Benchmark")
    if args.profile:
        print(f"  Profile:      {args.profile}")
    print(f"  Models:       {config.models}")
    print(f"  Workflows:    {config.workflow_types}")
    print(f"  Dataset:      {config.dataset}")
    print(f"  Judge Type:   {config.judge_type}")
    if config.judge_model:
        print(f"  Judge Model:  {config.judge_model}")
    if config.max_queries > 0:
        print(f"  Max Queries:  {config.max_queries}")
    if config.resume:
        print(f"  Resume:       enabled")
    if config.config_file:
        print(f"  Config:       {config.config_file}")
    print(f"  Output:       {config.output_dir}")
    print()

    asyncio.run(runner.run_all())
    runner.report(format=args.report)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse arguments for the standalone ``chemgraph-eval`` command."""
    parser = argparse.ArgumentParser(
        prog="chemgraph-eval",
        description="Run ChemGraph multi-model evaluation benchmarks.",
    )
    add_eval_args(parser)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Standalone entry point for ``chemgraph-eval``."""
    args = parse_args(argv)
    run_eval(args)


if __name__ == "__main__":
    main()
