"""Evaluation-only execution and answer formatting for the unchanged Deep Agent."""

import json
import os
from pathlib import Path
import time
from urllib.parse import quote

from chemgraph.agent.deepagent_backend import host_shell_environment
from chemgraph.agent.llm_agent import ChemGraph, PromptConfig
from chemgraph.agent.turn import serialize_state
from chemgraph.agent.usage import UsageCollector, combine_usage
from chemgraph.graphs.deep_agent import DEFAULT_DEEPAGENT_PROMPT
from chemgraph.models.loader import load_chat_model
from chemgraph.prompt.single_agent_prompt import formatter_prompt
from chemgraph.utils.get_workflow_from_llm import get_workflow_from_state
from chemgraph.utils.parsing import parse_response_formatter


async def format_result(model, state, usage):
    """Extract ResponseFormatter JSON using the same prompt and retry policy as single_agent."""
    messages = [
        {"role": "system", "content": formatter_prompt},
        {"role": "user", "content": json.dumps(state["messages"], default=str)},
    ]
    attempts = []
    for _ in range(2):
        response = await model.ainvoke(messages, config={"callbacks": [usage]})
        text = response.text
        attempts.append(text)
        formatted, error = parse_response_formatter(text)
        if error is None:
            return formatted.model_dump(mode="json"), attempts
        messages.extend([
            {"role": "assistant", "content": text},
            {"role": "user", "content": f"Error: {error}\nReturn only valid ResponseFormatter JSON."},
        ])
    return {**formatted.model_dump(mode="json"), "_parse_error": error}, attempts


async def run_query(config, model_name, query, query_id):
    """Run in one directory per model/thread without passing ground truth to the model."""
    from deepagents.backends import LocalShellBackend

    started = time.monotonic()
    previous_log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")
    safe_model = model_name.replace("/", "_").replace(":", "_")
    workspace_path = (Path(config.deepagent_workspace) / safe_model / "deep_agent"
                      / f"thread_{quote(str(query_id), safe='')}")
    workspace_path.mkdir(parents=True, exist_ok=True)
    workspace = str(workspace_path)
    run_config = {"configurable": {"thread_id": query_id}}
    formatting_usage = UsageCollector("eval", query_id, model=model_name)
    raw = {"query_id": query_id, "query": query, "workspace": workspace,
           "tool_calls": [], "result": "", "status": "initialization_error"}
    agent = None
    try:
        # This process-wide override requires sequential evaluation queries.
        os.environ["CHEMGRAPH_LOG_DIR"] = workspace
        model_options = {"model_name": model_name, "base_url": config.get_base_url(model_name),
                         "argo_user": config.get_argo_user()}
        agent = ChemGraph(
            **model_options, workflow_type="deep_agent", return_option="state",
            structured_output=False, enable_memory=False, log_dir=workspace,
            recursion_limit=config.recursion_limit, deepagent_discover_skills=False,
            # Native chemistry tools use host paths, so file tools must use them too.
            deepagent_backend=LocalShellBackend(
                root_dir=workspace, virtual_mode=False,
                env=host_shell_environment(), inherit_env=False,
            ),
            deepagent_auto_approve=config.deepagent_auto_approve,
            prompts=PromptConfig(deepagent=DEFAULT_DEEPAGENT_PROMPT + (
                f"\nThe working and output directory for this evaluation thread is {workspace}.\n"
                "File tools and chemistry tools use real host paths. Shell commands start "
                "in this directory and CHEMGRAPH_LOG_DIR points here. Keep generated inputs, "
                "scripts, and results here; use absolute paths under this directory in tool "
                "arguments. There is no /workspace mount for this run. Model runtime temporary "
                "directories are not output locations.\n"
            )),
        )
        raw["status"] = "execution_error"
        state = await agent.run(query, run_config)
        raw.update(get_workflow_from_state(state), state=state)
        if config.structured_output:
            raw["status"] = "formatting_error"
            model = load_chat_model(**model_options, temperature=0.0)
            raw["structured_output"], raw["formatter_responses"] = await format_result(
                model, state, formatting_usage,
            )
            if raw["structured_output"].get("_parse_error"):
                raw["error"] = raw["structured_output"]["_parse_error"]
                return raw
        raw["status"] = "completed"
    except Exception as exc:
        raw["error"] = f"{type(exc).__name__}: {exc}"
        raw["structured_output"] = {"_parse_error": raw["error"]}
        if agent is not None and "state" not in raw:
            try:
                state = serialize_state(await agent.aget_state(run_config))
                raw.update(get_workflow_from_state(state), state=state)
            except Exception:
                pass  # Initialization may have failed before any checkpoint exists.
    finally:
        formatting_usage.finish(raw["status"])
        execution_usage = agent.last_usage if agent is not None else None
        raw["usage"] = {"execution": execution_usage, "formatting": formatting_usage.summary}
        raw["usage"]["total"] = combine_usage([
            *([execution_usage] if execution_usage is not None else []), formatting_usage.summary,
        ])
        raw["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if previous_log_dir is None:
            os.environ.pop("CHEMGRAPH_LOG_DIR", None)
        else:
            os.environ["CHEMGRAPH_LOG_DIR"] = previous_log_dir
    return raw
