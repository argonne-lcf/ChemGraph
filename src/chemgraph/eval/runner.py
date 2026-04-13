"""Benchmark runner for ChemGraph multi-model evaluation.

Iterates over ``(model, workflow, query)`` combinations, collects
tool-call outputs, and scores them against ground truth using an
LLM-as-judge approach.
"""

import datetime
import inspect
import os
import traceback
from typing import Any, Dict, List

from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.eval.config import BenchmarkConfig
from chemgraph.eval.datasets import GroundTruthItem, load_dataset
from chemgraph.eval.llm_judge import (
    aggregate_judge_results,
    judge_single_query,
    load_judge_model,
)
from chemgraph.eval.reporter import (
    print_summary_table,
    write_json_report,
    write_markdown_report,
    write_model_detail,
)
from chemgraph.eval.structured_output_judge import (
    aggregate_structured_results,
    judge_structured_output,
)
from chemgraph.utils.get_workflow_from_llm import get_workflow_from_state
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class ModelBenchmarkRunner:
    """Run evaluation benchmarks across multiple LLM models and workflows.

    Uses an LLM judge to compare the agent's final answer against the
    ground-truth result (binary: correct/wrong).

    Parameters
    ----------
    config : BenchmarkConfig
        Evaluation configuration specifying models, workflows, dataset,
        and output settings.

    Examples
    --------
    >>> from chemgraph.eval import ModelBenchmarkRunner, BenchmarkConfig
    >>> config = BenchmarkConfig(
    ...     models=["gpt-4o-mini", "gemini-2.5-flash"],
    ...     judge_model="gpt-4o",
    ... )
    >>> runner = ModelBenchmarkRunner(config)
    >>> results = asyncio.run(runner.run_all())
    >>> runner.report()
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        full_dataset: List[GroundTruthItem] = load_dataset(config.dataset)
        # Apply max_queries limit if configured (0 = no limit).
        if config.max_queries > 0:
            self.dataset = full_dataset[: config.max_queries]
            logger.info(
                f"Limiting evaluation to {config.max_queries} of "
                f"{len(full_dataset)} queries"
            )
        else:
            self.dataset = full_dataset
        self.results: Dict[str, Dict[str, dict]] = {}
        self._run_metadata: dict = {}

        # Load judge model only when LLM judge is requested.
        self._judge_llm = None
        if config.judge_type in ("llm", "both"):
            logger.info(f"Loading judge model: {config.judge_model}")
            judge_base_url = config.get_base_url(config.judge_model)
            judge_argo_user = config.get_argo_user()
            self._judge_llm = load_judge_model(
                config.judge_model,
                base_url=judge_base_url,
                argo_user=judge_argo_user,
            )

        if config.judge_type in ("structured", "both"):
            n_with_so = sum(
                1
                for item in self.dataset
                if item.expected_structured_output is not None
            )
            logger.info(
                f"Structured output judge enabled: {n_with_so}/{len(self.dataset)} "
                f"queries have expected structured output"
            )

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    async def _run_single_model_workflow(
        self,
        model_name: str,
        workflow_type: str,
    ) -> dict:
        """Run all queries for one (model, workflow) pair.

        Returns
        -------
        dict
            Contains ``"judge_aggregate"``, ``"judge_details"``, and
            ``"raw_tool_calls"``.
        """
        logger.info(
            f"Starting evaluation: model={model_name}, workflow={workflow_type}"
        )

        # Isolate log directory per model+workflow so parallel runs don't clash.
        run_log_dir = os.path.join(
            self.config.output_dir,
            "logs",
            model_name.replace("/", "_").replace(":", "_"),
            workflow_type,
        )
        os.makedirs(run_log_dir, exist_ok=True)

        try:
            # Resolve per-model base_url and argo_user from config.toml.
            base_url = self.config.get_base_url(model_name)
            argo_user = self.config.get_argo_user()

            # Build desired kwargs and filter to only those accepted by
            # the installed ChemGraph version, so the runner works even
            # against older releases that lack newer parameters.
            desired_kwargs = {
                "model_name": model_name,
                "workflow_type": workflow_type,
                "structured_output": self.config.structured_output,
                "return_option": "state",
                "recursion_limit": self.config.recursion_limit,
                "enable_memory": False,
                "base_url": base_url,
                "argo_user": argo_user,
                "log_dir": run_log_dir,
            }
            sig = inspect.signature(ChemGraph.__init__)
            valid_params = set(sig.parameters.keys()) - {"self"}
            filtered_kwargs = {
                k: v for k, v in desired_kwargs.items() if k in valid_params
            }

            cg = ChemGraph(**filtered_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialise ChemGraph for {model_name}: {e}")
            return self._make_error_result(
                f"Initialisation failed: {e}",
                len(self.dataset),
            )

        raw_tool_calls: List[dict] = []
        per_query_judge_results: List[dict] = []
        per_query_structured_results: List[dict] = []

        for idx, item in enumerate(self.dataset):
            query_result = await self._run_single_query(
                cg, item, idx, model_name, workflow_type
            )
            raw_tool_calls.append(query_result["raw"])
            if query_result.get("judge") is not None:
                per_query_judge_results.append(query_result["judge"])
            if query_result.get("structured_judge") is not None:
                per_query_structured_results.append(query_result["structured_judge"])

        result: Dict[str, Any] = {
            "raw_tool_calls": raw_tool_calls,
        }

        # LLM judge results.
        if self.config.judge_type in ("llm", "both"):
            judge_agg = aggregate_judge_results(per_query_judge_results)
            result["judge_aggregate"] = judge_agg
            result["judge_details"] = per_query_judge_results

        # Structured output judge results.
        if self.config.judge_type in ("structured", "both"):
            struct_agg = aggregate_structured_results(per_query_structured_results)
            result["structured_judge_aggregate"] = struct_agg
            result["structured_judge_details"] = per_query_structured_results

        # Log summary.
        parts = [f"Completed eval {model_name}/{workflow_type}:"]
        if "judge_aggregate" in result:
            jagg = result["judge_aggregate"]
            parts.append(
                f"llm_judge={jagg['accuracy']:.1%} "
                f"({jagg['n_correct']}/{jagg['n_queries']})"
            )
        if "structured_judge_aggregate" in result:
            sagg = result["structured_judge_aggregate"]
            parts.append(
                f"struct_judge={sagg['accuracy']:.1%} "
                f"({sagg['n_correct']}/{sagg['n_queries']})"
            )
        logger.info(" ".join(parts))

        return result

    async def _run_single_query(
        self,
        cg: ChemGraph,
        item: GroundTruthItem,
        idx: int,
        model_name: str,
        workflow_type: str,
    ) -> dict:
        """Execute and evaluate a single query.

        Returns ``{"raw": ..., "judge": ..., "structured_judge": ...}``.
        """
        try:
            config = {"configurable": {"thread_id": str(idx)}}
            state = await cg.run(item.query, config)
            llm_workflow = get_workflow_from_state(state)
            model_tool_calls = llm_workflow.get("tool_calls", [])
            model_result = llm_workflow.get("result", "")
        except Exception as e:
            logger.warning(f"Query {idx} failed for {model_name}/{workflow_type}: {e}")
            logger.debug(traceback.format_exc())
            model_tool_calls = []
            model_result = f"ERROR: {e}"
            llm_workflow = {"tool_calls": [], "result": model_result}

        result: Dict[str, Any] = {"raw": llm_workflow}

        # --- LLM judge ---
        if self.config.judge_type in ("llm", "both") and self._judge_llm is not None:
            judge_result = await judge_single_query(
                judge_llm=self._judge_llm,
                query=item.query,
                expected_result=item.expected_result,
                model_result=model_result,
                expected_tool_calls=item.expected_tool_calls,
                model_tool_calls=model_tool_calls,
            )
            judge_result["query_id"] = item.id
            judge_result["query"] = item.query
            judge_result["category"] = item.category
            result["judge"] = judge_result

        # --- Structured output judge ---
        if self.config.judge_type in ("structured", "both"):
            if item.expected_structured_output is not None:
                struct_result = judge_structured_output(
                    expected=item.expected_structured_output,
                    actual=model_result,
                )
                struct_result["query_id"] = item.id
                struct_result["query"] = item.query
                struct_result["category"] = item.category
                result["structured_judge"] = struct_result
            else:
                logger.debug(
                    f"Query {idx}: no expected_structured_output, "
                    f"skipping structured judge"
                )

        return result

    async def run_all(self) -> Dict[str, Dict[str, dict]]:
        """Execute the full benchmark: all models x all workflows.

        Models are run **sequentially** to avoid API rate-limit issues
        and to keep log directories clean.  Within a model, queries run
        sequentially as well (the ``ChemGraph.run`` method already uses
        async streaming internally).

        Returns
        -------
        dict
            ``{model_name: {workflow_type: {"judge_aggregate": ..., ...}}}``
        """
        timestamp = datetime.datetime.now().isoformat()
        self._run_metadata = {
            "timestamp": timestamp,
            "dataset": self.config.dataset,
            "n_queries": len(self.dataset),
            "models": self.config.models,
            "workflow_types": self.config.workflow_types,
            "judge_model": self.config.judge_model,
            "judge_type": self.config.judge_type,
            "structured_output": self.config.structured_output,
            "tags": self.config.tags,
        }

        self.results = {}

        for model_name in self.config.models:
            self.results[model_name] = {}
            for workflow_type in self.config.workflow_types:
                result = await self._run_single_model_workflow(
                    model_name, workflow_type
                )
                self.results[model_name][workflow_type] = result

                # Write per-model detail file immediately so partial
                # results survive if a later model fails.
                write_model_detail(
                    model_name=model_name,
                    workflow_type=workflow_type,
                    raw_tool_calls=result["raw_tool_calls"],
                    per_query_results=[],
                    output_dir=self.config.output_dir,
                    judge_results=result.get("judge_details"),
                    structured_judge_results=result.get("structured_judge_details"),
                )

        return self.results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self, format: str = "all") -> None:
        """Generate and write evaluation reports.

        Parameters
        ----------
        format : str
            ``"json"``, ``"markdown"``, ``"console"``, or ``"all"``
            (default).
        """
        if not self.results:
            logger.warning("No results to report. Run run_all() first.")
            return

        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if format in ("json", "all"):
            write_json_report(
                results=self.results,
                metadata=self._run_metadata,
                output_path=os.path.join(
                    self.config.output_dir, f"benchmark_{ts}.json"
                ),
            )

        if format in ("markdown", "all"):
            write_markdown_report(
                results=self.results,
                metadata=self._run_metadata,
                output_path=os.path.join(self.config.output_dir, f"benchmark_{ts}.md"),
            )

        if format in ("console", "all"):
            print_summary_table(self.results)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_error_result(error_msg: str, n_queries: int) -> dict:
        """Build an error placeholder result for a failed model init."""
        return {
            "judge_aggregate": {
                "n_queries": n_queries,
                "n_correct": 0,
                "accuracy": 0.0,
                "n_parse_errors": 0,
                "error": error_msg,
            },
            "judge_details": [],
            "raw_tool_calls": [],
        }
