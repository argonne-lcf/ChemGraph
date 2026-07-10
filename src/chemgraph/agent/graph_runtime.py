"""Public ChemGraph runtime builder for configured graph demos.

This module owns MCP/science tool binding for configured ChemGraph graphs. Thin
transport layers such as ``academy_sim`` pass peer tools as extra tools but do
not load or reason about MCP servers themselves.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.tools import BaseTool

from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.mcp.client import load_mcp_tools
from chemgraph.models.settings import LLMSettings
from chemgraph.prompt.multi_agent_prompt import executor_prompt as default_executor_prompt
from chemgraph.prompt.multi_agent_prompt import planner_prompt as default_planner_prompt
from chemgraph.prompt.single_agent_prompt import single_agent_prompt

TraceFn = Callable[[str, dict[str, Any]], None]
GraphRunner = Callable[[str], Awaitable['GraphRunResult']]


@dataclass(frozen=True)
class GraphRunResult:
    """Result metadata for one configured ChemGraph graph turn."""

    output: str
    executed_tool_names: tuple[str, ...] = ()
    terminal_tool: str | None = None


class ConfiguredGraphSpec(Protocol):
    """Fields required to build a configured ChemGraph graph."""

    name: str
    workflow_type: str
    system_prompt: str | None
    planner_prompt: str | None
    executor_prompt: str | None
    workflow_kwargs: dict[str, Any]
    science_tools: tuple[Any, ...]


async def make_configured_graph_runner(
    *,
    spec: ConfiguredGraphSpec,
    llm_settings: LLMSettings,
    extra_tools: Sequence[BaseTool] = (),
    prompt_suffix: str = '',
    trace: TraceFn | None = None,
    log_dir: str | None = None,
    terminal_tool_names: Sequence[str] = (),
) -> GraphRunner:
    """Build a ChemGraph graph from config and return ``input -> output``.

    MCP tools are loaded here, in ChemGraph-owned code. Callers may pass
    transport-specific peer tools through ``extra_tools``.
    """

    trace = trace or (lambda _event, _payload: None)
    science_tools = await load_mcp_tools(spec.science_tools)
    tools = [*science_tools, *extra_tools]

    last_finished_payload: dict[str, Any] = {}

    def run_trace(event: str, payload: dict[str, Any]) -> None:
        nonlocal last_finished_payload
        if event == 'workflow_finished':
            last_finished_payload = dict(payload)
        trace(event, payload)

    kwargs: dict[str, Any] = {
        'model_name': llm_settings.model,
        'workflow_type': spec.workflow_type,
        'base_url': llm_settings.base_url,
        'api_key': llm_settings.api_key,
        'argo_user': llm_settings.user,
        'tools': tools,
        'on_event': run_trace,
        'log_dir': log_dir,
        'terminal_tool_names': tuple(terminal_tool_names),
    }
    if prompt_suffix or spec.system_prompt is not None:
        kwargs['system_prompt'] = (spec.system_prompt or single_agent_prompt) + prompt_suffix
    if prompt_suffix or spec.planner_prompt is not None:
        kwargs['planner_prompt'] = (
            spec.planner_prompt or default_planner_prompt
        ) + prompt_suffix
    if prompt_suffix or spec.executor_prompt is not None:
        kwargs['executor_prompt'] = (
            spec.executor_prompt or default_executor_prompt
        ) + prompt_suffix
    kwargs.update(spec.workflow_kwargs)

    cg = ChemGraph(**kwargs)
    thread_id = spec.name

    async def run_graph(input_text: str) -> GraphRunResult:
        nonlocal last_finished_payload
        last_finished_payload = {}
        message = await cg.run(input_text, config={'thread_id': thread_id})
        return GraphRunResult(
            output=getattr(message, 'content', None) or str(message),
            executed_tool_names=tuple(
                str(name)
                for name in last_finished_payload.get('executed_tool_names', ())
            ),
            terminal_tool=last_finished_payload.get('terminal_tool'),
        )

    return run_graph
