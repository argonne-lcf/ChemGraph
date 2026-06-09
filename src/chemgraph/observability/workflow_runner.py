"""Observed execution helpers for traditional ChemGraph workflows."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.agent.llm_agent import serialize_state
from chemgraph.observability.events import WorkflowEventContext
from chemgraph.observability.events import current_workflow_event_context
from chemgraph.observability.events import emit_workflow_event
from chemgraph.observability.events import new_span_id
from chemgraph.observability.events import workflow_event_context


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def normalize_model_name(model_name: str, base_url: str | None) -> str:
    value = model_name.strip()
    if base_url and "argoapi" in base_url and value.startswith("GPT-"):
        return "argo:" + value.lower()
    return value


def compact_value(value: Any, *, max_chars: int = 8000) -> Any:
    try:
        text = json.dumps(value, default=str, sort_keys=True)
    except TypeError:
        text = str(value)
    if len(text) <= max_chars:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return {
        "truncated": True,
        "preview": text[:max_chars],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _write_status(
    *,
    run_dir: Path,
    run_id: str,
    workflow_span_id: str,
    query: str,
    workflow_type: str,
    model_name: str,
    base_url: str | None,
    status: str,
    started_at: float,
    error: str | None = None,
) -> None:
    now = time.time()
    payload = {
        "mode": "chemgraph_workflow",
        "run_id": run_id,
        "workflow_span_id": workflow_span_id,
        "query": query,
        "workflow_type": workflow_type,
        "model_name": model_name,
        "base_url": base_url,
        "status": status,
        "started": started_at,
        "updated": now,
        "finished": now if status in {"completed", "failed"} else None,
        "events_path": str(run_dir / "events.jsonl"),
    }
    if error:
        payload["error"] = error
    _write_json(run_dir / "status.json", payload)


def _write_manifest(
    *,
    run_dir: Path,
    run_id: str,
    workflow_span_id: str,
    query: str,
    workflow_type: str,
    model_name: str,
    base_url: str | None,
) -> None:
    _write_json(
        run_dir / "manifest.json",
        {
            "mode": "chemgraph_workflow",
            "run_id": run_id,
            "workflow_span_id": workflow_span_id,
            "query": query,
            "workflow_type": workflow_type,
            "model_name": model_name,
            "base_url": base_url,
            "events_path": str(run_dir / "events.jsonl"),
        },
    )


def _workflow_log_dir(run_dir: Path, workflow_span_id: str) -> str:
    path = run_dir / "chemgraph_workflows" / workflow_span_id
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


async def run_observed_chemgraph_workflow(
    *,
    query: str,
    run_dir: str | Path | None = None,
    run_id: str | None = None,
    workflow_type: str = "single_agent",
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    argo_user: str | None = None,
    return_option: Literal["last_message", "state"] = "state",
    recursion_limit: int = 50,
    parent_span_id: str | None = None,
    agent_id: str = "chemgraph-workflow",
    role: str = "TraditionalChemGraphWorkflow",
    write_run_files: bool = True,
) -> dict[str, Any]:
    """Run a traditional ChemGraph workflow while emitting dashboard events."""
    current_context = current_workflow_event_context()
    run_dir_value = run_dir
    if run_dir_value is None and current_context and current_context.run_dir:
        run_dir_value = current_context.run_dir
    if run_dir_value is None:
        run_dir_value = "runs/local-chemgraph-workflow"
    effective_run_dir = Path(run_dir_value).resolve()
    effective_run_dir.mkdir(parents=True, exist_ok=True)

    workflow_span_id = new_span_id("chemgraph-workflow")
    effective_run_id = run_id or effective_run_dir.name
    base_url = base_url or _env_first(
        "CHEMGRAPH_WORKFLOW_BASE_URL",
        "ACADEMY_LM_BASE_URL",
    )
    model_name = normalize_model_name(
        model_name
        or _env_first("CHEMGRAPH_WORKFLOW_MODEL", "ACADEMY_LM_MODEL")
        or "argo:gpt-5.4",
        base_url,
    )
    api_key = api_key or _env_first(
        "CHEMGRAPH_WORKFLOW_API_KEY",
        "ACADEMY_LM_API_KEY",
        "OPENAI_API_KEY",
    )
    argo_user = argo_user or _env_first(
        "CHEMGRAPH_WORKFLOW_ARGO_USER",
        "ACADEMY_LM_USER",
        "ARGO_USER",
    )

    started_at = time.time()
    if write_run_files:
        _write_manifest(
            run_dir=effective_run_dir,
            run_id=effective_run_id,
            workflow_span_id=workflow_span_id,
            query=query,
            workflow_type=workflow_type,
            model_name=model_name,
            base_url=base_url,
        )
        _write_status(
            run_dir=effective_run_dir,
            run_id=effective_run_id,
            workflow_span_id=workflow_span_id,
            query=query,
            workflow_type=workflow_type,
            model_name=model_name,
            base_url=base_url,
            status="running",
            started_at=started_at,
        )

    context_manager = (
        workflow_event_context(
            jsonl_path=effective_run_dir / "events.jsonl",
            context=WorkflowEventContext(
                run_id=effective_run_id,
                run_dir=str(effective_run_dir),
                agent_id=agent_id,
                role=role,
                parent_span_id=parent_span_id,
                tool_name=None,
            ),
        )
        if current_context is None
        else None
    )

    if context_manager is None:
        return await _run_observed_chemgraph_workflow_inner(
            query=query,
            run_dir=effective_run_dir,
            run_id=effective_run_id,
            workflow_span_id=workflow_span_id,
            workflow_type=workflow_type,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            argo_user=argo_user,
            return_option=return_option,
            recursion_limit=recursion_limit,
            write_run_files=write_run_files,
            started_at=started_at,
        )
    with context_manager:
        return await _run_observed_chemgraph_workflow_inner(
            query=query,
            run_dir=effective_run_dir,
            run_id=effective_run_id,
            workflow_span_id=workflow_span_id,
            workflow_type=workflow_type,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            argo_user=argo_user,
            return_option=return_option,
            recursion_limit=recursion_limit,
            write_run_files=write_run_files,
            started_at=started_at,
        )


async def _run_observed_chemgraph_workflow_inner(
    *,
    query: str,
    run_dir: Path,
    run_id: str,
    workflow_span_id: str,
    workflow_type: str,
    model_name: str,
    base_url: str | None,
    api_key: str | None,
    argo_user: str | None,
    return_option: Literal["last_message", "state"],
    recursion_limit: int,
    write_run_files: bool,
    started_at: float,
) -> dict[str, Any]:
    log_dir = _workflow_log_dir(run_dir, workflow_span_id)
    config = {"configurable": {"thread_id": workflow_span_id}}

    emit_workflow_event(
        "run_started",
        {
            "workflow_type": workflow_type,
            "model_name": model_name,
            "query": query,
        },
        span_id=workflow_span_id,
    )
    emit_workflow_event(
        "workflow_started",
        {
            "workflow_type": workflow_type,
            "model_name": model_name,
            "query": query,
            "log_dir": log_dir,
        },
        span_id=workflow_span_id,
    )
    try:
        emit_workflow_event(
            "workflow_node_started",
            {"workflow_node": "ChemGraph", "phase": "construct"},
            span_id=new_span_id("chemgraph-node"),
            parent_span_id=workflow_span_id,
        )
        agent = ChemGraph(
            model_name=model_name,
            workflow_type=workflow_type,
            base_url=base_url,
            api_key=api_key,
            argo_user=argo_user,
            return_option=return_option,
            recursion_limit=recursion_limit,
            log_dir=log_dir,
        )
        emit_workflow_event(
            "workflow_node_finished",
            {"workflow_node": "ChemGraph", "phase": "construct"},
            span_id=new_span_id("chemgraph-node"),
            parent_span_id=workflow_span_id,
        )
        emit_workflow_event(
            "workflow_node_started",
            {"workflow_node": "LangGraph", "phase": "run"},
            span_id=new_span_id("chemgraph-node"),
            parent_span_id=workflow_span_id,
        )
        result = await agent.run(
            query,
            config=config,
            workflow_span_id=workflow_span_id,
        )
        state_payload = serialize_state(agent.get_state(config=config))
        emit_workflow_event(
            "workflow_node_finished",
            {"workflow_node": "LangGraph", "phase": "run"},
            span_id=new_span_id("chemgraph-node"),
            parent_span_id=workflow_span_id,
        )
        emit_workflow_event(
            "workflow_finished",
            {
                "workflow_type": workflow_type,
                "status": "completed",
                "log_dir": log_dir,
            },
            span_id=workflow_span_id,
        )
        emit_workflow_event(
            "run_finished",
            {
                "workflow_type": workflow_type,
                "status": "completed",
            },
            span_id=workflow_span_id,
        )
        payload = {
            "status": "completed",
            "workflow_type": workflow_type,
            "model_name": model_name,
            "span_id": workflow_span_id,
            "log_dir": log_dir,
            "return_option": return_option,
            "result": compact_value(serialize_state(result)),
            "state": compact_value(state_payload),
        }
        if write_run_files:
            _write_status(
                run_dir=run_dir,
                run_id=run_id,
                workflow_span_id=workflow_span_id,
                query=query,
                workflow_type=workflow_type,
                model_name=model_name,
                base_url=base_url,
                status="completed",
                started_at=started_at,
            )
            _write_json(run_dir / "result.json", payload)
        return payload
    except Exception as exc:
        error = repr(exc)
        emit_workflow_event(
            "workflow_finished",
            {
                "workflow_type": workflow_type,
                "status": "failed",
                "error": error,
                "log_dir": log_dir,
            },
            span_id=workflow_span_id,
        )
        emit_workflow_event(
            "run_finished",
            {
                "workflow_type": workflow_type,
                "status": "failed",
                "error": error,
            },
            span_id=workflow_span_id,
        )
        if write_run_files:
            _write_status(
                run_dir=run_dir,
                run_id=run_id,
                workflow_span_id=workflow_span_id,
                query=query,
                workflow_type=workflow_type,
                model_name=model_name,
                base_url=base_url,
                status="failed",
                started_at=started_at,
                error=error,
            )
        raise
