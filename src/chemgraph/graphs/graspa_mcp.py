"""Planner/executor/analyst gRASPA workflow with deterministic batch collection."""

import json
from pathlib import Path
import uuid

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from chemgraph.execution.graspa_workflow import (
    GraspaCollector, call_tool, gather_limited, load_journal, run_lock, save_journal, validate_records,
)
from chemgraph.prompt.graspa_prompt import planner_prompt, executor_prompt, analyst_prompt
from chemgraph.schemas.graspa_schema import graspa_input_schema_ensemble
from chemgraph.schemas.graspa_workflow import GraspaPlan, GraspaWorkflowOptions
from chemgraph.state.graspa_state import PlannerState
from chemgraph.tools.graspa_analysis import analyze_records, read_records, write_json

_DEFAULT_CHECKPOINTER = object()


def _validation_feedback(exc):
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(map(str, item['loc'])) or 'response'}: {item['msg']}"
            for item in exc.errors(include_input=False, include_context=False, include_url=False)[:5]
        )[:1000]
    if isinstance(exc, (OutputParserException, json.JSONDecodeError)):
        # Parser exceptions can embed the entire raw response.
        return "Response could not be parsed; return only the requested structured output"
    return str(exc)[:1000]


async def _validated_response(llm, schema, messages, validate, root, stage, config):
    """Correct structured output at most twice, retaining the original context."""
    model = llm.with_structured_output(schema)
    diagnostics = root / f"validation-{stage}-{uuid.uuid4().hex}.json"
    failures = []
    feedback = []
    for attempt in range(1, 4):
        try:
            value = await model.ainvoke(messages + feedback, config=config)
        except (ValidationError, OutputParserException, json.JSONDecodeError) as exc:
            error = _validation_feedback(exc)
        else:
            try:
                return validate(schema.model_validate(value))
            except (ValueError, FileNotFoundError) as exc:
                error = _validation_feedback(exc)
        failures.append({"attempt": attempt, "error": error})
        write_json(diagnostics, {"stage": stage, "failures": failures})
        feedback = [HumanMessage(content=f"Correct your previous response: {error}. Preserve the original request and selected task.")]
    raise ValueError(f"{stage} validation failed after 3 attempts: {error}. See {diagnostics}") from None


def construct_graspa_mcp_graph(
    llm,
    planner_prompt: str = planner_prompt,
    executor_prompt: str = executor_prompt,
    analyst_prompt: str = analyst_prompt,
    executor_tools: list = None,
    analysis_tools: list = None,
    checkpointer=_DEFAULT_CHECKPOINTER,
    options: GraspaWorkflowOptions | dict | None = None,
    log_dir: str | None = None,
):
    """Build a graph whose LLMs plan and explain, while Python owns job completion.

    The optional analysis tools supplement canonical local analysis. The supplied
    executor tools may come from the maintained HPC or deprecated Parsl server.
    """
    options = GraspaWorkflowOptions.model_validate({} if options is None else options)
    contract = options.request_contract
    tools = executor_tools or []
    if checkpointer is _DEFAULT_CHECKPOINTER:
        checkpointer = MemorySaver()

    async def plan(state, config: RunnableConfig):
        from chemgraph.tools.ase_core import _resolve_path

        names = [tool.name for tool in tools]
        if "run_graspa_ensemble" not in names or len(names) != len(set(names)):
            raise ValueError("Supply uniquely named MCP tools including run_graspa_ensemble")
        query = state["messages"][-1].content
        if not isinstance(query, str):
            raise ValueError("The gRASPA workflow expects a text request")
        if options.run_directory:
            root = Path(_resolve_path(str(Path(options.run_directory).expanduser()))).resolve()
        else:
            base = Path(log_dir or _resolve_path(".")).expanduser().resolve()
            root = base / "graspa_workflows" / uuid.uuid4().hex
        with run_lock(root):
            journal_path = root / "workflow.json"
            if journal_path.exists():
                if not options.resume:
                    raise ValueError("Run directory already contains a workflow; use resume=True or a fresh directory")
                journal = load_journal(root)
                if journal["query"] != query:
                    raise ValueError("Resume requires the original query and frozen plan")
                if journal.get("request_contract") != (contract.model_dump(mode="json") if contract else None):
                    raise ValueError("Resume requires the original request contract")
            else:
                if options.resume:
                    raise ValueError("No saved workflow exists in run_directory")
                journal = {"version": 1, "query": query, "plan": None, "requests": {}, "batches": {},
                           "request_contract": contract.model_dump(mode="json") if contract else None}
                save_journal(root, journal)
            if journal["plan"] is None:
                def validate_plan(result):
                    if contract:
                        contract.check_plan(result)
                    return result.model_dump(mode="json")

                journal["plan"] = await _validated_response(
                    llm, GraspaPlan, [SystemMessage(content=planner_prompt), HumanMessage(content=query)],
                    validate_plan, root, "planning", config,
                )
                save_journal(root, journal)
            if contract:
                contract.check_plan(GraspaPlan.model_validate(journal["plan"]))
            write_json(root / "plan.json", journal["plan"])
        return {"run_directory": str(root), "plan_path": str(root / "plan.json"),
                "executor_results": {}, "analysis": {}, "workflow_status": "planned"}

    async def prepare(state, config: RunnableConfig):
        from chemgraph.mcp.graspa_mcp_hpc import _local_structure_files

        root = Path(state["run_directory"])
        with run_lock(root):
            journal = load_journal(root)
            if journal["requests"]:
                if contract:
                    if len(journal["requests"]) != 1:
                        raise ValueError("Request contract requires exactly one saved ensemble")
                    for params in journal["requests"].values():
                        contract.check_request(params, frozen=True)
                return {"workflow_status": "prepared"}
            planned = GraspaPlan.model_validate(journal["plan"])

            async def request(task):
                context = json.dumps({"original_request": journal["query"], "selected_task_index": task.task_index,
                                      "analysis": planned.analysis.model_dump(mode="json") if planned.analysis else None})

                def validate_request(value):
                    params = value.model_dump(mode="json")
                    if contract:
                        contract.check_request(params)
                    if not params["remote_structure_directory"]:
                        # Freeze discovery before submitting any task; preserve duplicates.
                        params["input_structures"] = _local_structure_files(params["input_structures"])
                        if params["output_directory"] is None and Path(params["output_result_file"]).parent == Path("."):
                            params["output_directory"] = str(root / "simulations")
                    if contract:
                        contract.check_request(params, frozen=True)
                    return params

                params = await _validated_response(
                    llm, graspa_input_schema_ensemble,
                    [SystemMessage(content=executor_prompt), HumanMessage(content=context), HumanMessage(content=task.prompt)],
                    validate_request, root, f"preparation-task_{task.task_index}", config,
                )
                return f"task_{task.task_index}", params

            # No submissions occur until every prepared request validates.
            requests = dict(await gather_limited(request, planned.tasks, config))
            if planned.analysis:
                conditions = {(c["temperature"], c["pressure"]) for p in requests.values() for c in p["conditions"]}
                for condition in (planned.analysis.adsorption, planned.analysis.desorption):
                    if condition and (condition.temperature, condition.pressure) not in conditions:
                        raise ValueError("Analysis condition was not included in the prepared simulations")
            journal["requests"] = requests
            save_journal(root, journal)
        return {"workflow_status": "prepared"}

    async def execute(state, config: RunnableConfig):
        collector = GraspaCollector(state["run_directory"], tools, options, config)
        outcomes = await collector.collect_all()
        return {"executor_results": outcomes, "workflow_status": "collected"}

    def analyze(state):
        root = Path(state["run_directory"])
        with run_lock(root):
            journal = load_journal(root)
            records = []
            errors = {key: outcome for key, outcome in state["executor_results"].items()
                      if outcome["status"] == "collection_error"}
            for task_id, entry in journal["batches"].items():
                if not entry.get("records_path"):
                    continue
                try:
                    rows = read_records(entry["records_path"])
                    records.extend(validate_records(
                        rows, journal["requests"][task_id], entry.get("n_tasks"),
                        terminal=entry["phase"] == "collected", legacy=entry.get("legacy", False),
                    ))
                except (OSError, ValueError, TypeError) as exc:
                    errors[task_id] = {"task_id": task_id, "status": "collection_error",
                                       "message": str(exc)[:1000]}
            analysis = GraspaPlan.model_validate(journal["plan"]).analysis
            summary = analyze_records(records, root, None if errors else analysis)
            if errors:
                summary.update(status="incomplete", collection_errors=len(errors),
                               error_preview=list(errors.values())[:5],
                               conditions=analysis.model_dump(mode="json") if analysis else None)
                # A failed resume must not leave an earlier ranking looking current.
                for filename in ("rankings.csv", "top_candidates.csv", "excluded.json"):
                    (root / filename).unlink(missing_ok=True)
                write_json(root / "analysis.json", summary)
            journal["status"] = summary["status"]
            save_journal(root, journal)
        return {"analysis": summary, "workflow_status": summary["status"]}

    async def explain(state, config: RunnableConfig):
        summary = state["analysis"]
        messages = [SystemMessage(content=analyst_prompt),
                    HumanMessage(content=json.dumps(summary, allow_nan=False))]
        additional = {tool.name: tool for tool in analysis_tools or []}
        root = Path(state["run_directory"])
        try:
            model = llm.bind_tools(list(additional.values())) if additional else llm
            # Optional follow-up tools cannot replace or change canonical analysis.
            for turn in range(3):
                response = await model.ainvoke(messages, config=config)
                messages.append(response)
                calls = getattr(response, "tool_calls", [])
                if not calls:
                    break
                for call in calls:
                    try:
                        output = await call_tool(additional, call["name"], call["args"], config)
                        artifact = root / f"analyst-tool-{uuid.uuid4().hex}.json"
                        write_json(artifact, output)
                        content = json.dumps({"artifact_path": str(artifact), "preview": str(output)[:1500]})
                    except Exception as exc:
                        content = f"Follow-up tool failed: {str(exc)[:1500]}"
                    messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
            else:
                response = await llm.ainvoke(messages + [HumanMessage(content="Report the saved analysis now; do not call tools.")], config=config)
            if getattr(response, "tool_calls", None):
                response = AIMessage(content="Canonical analysis: " + json.dumps(summary))
        except Exception as exc:
            write_json(root / "report_error.json", {"message": str(exc)})
            response = AIMessage(content="Analyst explanation unavailable. Canonical analysis: " + json.dumps(summary))
        (root / "response.txt").write_text(str(response.content) + "\n")
        return {"messages": [response]}

    graph = StateGraph(PlannerState)
    for name, node in (("Planner", plan), ("Prepare", prepare), ("Executors", execute),
                       ("Analyze", analyze), ("Analyst", explain)):
        graph.add_node(name, node)
    graph.add_edge(START, "Planner")
    graph.add_edge("Planner", "Prepare")
    graph.add_edge("Prepare", "Executors")
    graph.add_edge("Executors", "Analyze")
    graph.add_edge("Analyze", "Analyst")
    graph.add_edge("Analyst", END)
    return graph.compile(checkpointer=checkpointer)
