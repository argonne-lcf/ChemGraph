"""Long-lived ChemGraph supervisor built from LangChain agent middleware."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any

from deepagents.backends import StateBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.middleware import FilesystemMiddleware, SubAgentMiddleware
from deepagents.middleware.subagents import CompiledSubAgent
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, TodoListMiddleware
from deepagents.middleware._state import private_state_field_names
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.summarization import create_summarization_middleware
from langchain_core.messages import AIMessage, ToolMessage, convert_to_messages
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt

from chemgraph.agent.events import SUBAGENT_METADATA_KEY
from chemgraph.graphs.deep_agent import (
    DEFAULT_DEEPAGENT_PROMPT,
    construct_deep_agent_graph,
)
from chemgraph.graphs.single_agent import construct_single_agent_graph
from chemgraph.memory.subagent_recorder import SubagentRunRecorder
from chemgraph.registry.tools import ToolRegistry
from chemgraph.graphs.workspace import (
    _DEFAULT_INTERRUPT_POLICY, default_tool_registry, prepare_workspace_runtime,
)


DEFAULT_MAIN_AGENT_PROMPT = """\
You are the long-lived ChemGraph main agent. Complete requests using skills,
workspace tools, local chemistry tools, and configured specialists. Read the
relevant available skill before a specialized workflow. Use direct tools for
focused operations and task for substantial specialist work. Delegation is
optional; give workers self-contained inputs and constraints, use their exact
registered names, and review their results. Do not run workspace mutations in
parallel. Ask for missing inputs and never invent scientific results.

Inspect tool schemas and preserve the user's calculator and execution method.
Follow existing action approvals. Treat /workspace as the project root when
mounted; follow the shell/virtual path mappings. Packaged skills under
/chemgraph-skills/ are readable resources, not shell paths. Local registry tools
execute on this host independently of the workspace backend; use absolute host
paths for their artifacts. Establish remote input visibility before submission.
Return a self-contained account of results, artifact paths, and unresolved work.
"""


def latest_assistant_text(messages: list[Any]) -> str:
    """Return the latest assistant message text from a message history."""
    for message in reversed(convert_to_messages(messages)):
        if isinstance(message, AIMessage):
            return str(message.text)
    return ""


def _preserve_terminal_tool_output(result: Any) -> Any:
    """Expose trailing worker tool output to Deep Agents as a final AI message."""
    if not isinstance(result, dict) or result.get("structured_response") is not None:
        return result

    messages = convert_to_messages(result.get("messages", []))
    if not messages or not isinstance(messages[-1], ToolMessage):
        return result

    trailing_text: list[str] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        if text := str(message.text).strip():
            trailing_text.append(text)
    if not trailing_text:
        return result

    return {
        **result,
        "messages": [
            *messages,
            AIMessage(content="\n".join(reversed(trailing_text))),
        ],
    }


def _adapt_subagent(
    spec: CompiledSubAgent,
    recorder: SubagentRunRecorder | None = None,
) -> CompiledSubAgent:
    runnable = spec["runnable"]

    def child_config(config: RunnableConfig) -> RunnableConfig:
        adapted_config = dict(config)
        metadata = dict(adapted_config.get("metadata") or {})
        metadata[SUBAGENT_METADATA_KEY] = spec["name"]
        adapted_config["metadata"] = metadata
        return adapted_config

    def invoke(state: Any, config: RunnableConfig) -> Any:
        run_id = recorder.start(spec["name"], state, config) if recorder else None
        try:
            result = runnable.invoke(state, config=child_config(config))
        except GraphInterrupt:
            if recorder and run_id:
                recorder.interrupted(run_id)
            raise
        except Exception as exc:
            if recorder and run_id:
                recorder.failed(run_id, exc)
            raise
        result = _preserve_terminal_tool_output(result)
        if recorder and run_id:
            recorder.completed(run_id, result)
        return result

    async def ainvoke(state: Any, config: RunnableConfig) -> Any:
        run_id = recorder.start(spec["name"], state, config) if recorder else None
        try:
            result = await runnable.ainvoke(state, config=child_config(config))
        except GraphInterrupt:
            if recorder and run_id:
                recorder.interrupted(run_id)
            raise
        except Exception as exc:
            if recorder and run_id:
                recorder.failed(run_id, exc)
            raise
        result = _preserve_terminal_tool_output(result)
        if recorder and run_id:
            recorder.completed(run_id, result)
        return result

    return {
        "name": spec["name"],
        "description": spec["description"],
        "runnable": RunnableLambda(invoke, afunc=ainvoke),
    }


def _validate_subagents(
    subagents: Sequence[CompiledSubAgent],
    recorder: SubagentRunRecorder | None = None,
) -> list[CompiledSubAgent]:
    if not subagents:
        raise ValueError("At least one subagent must be registered.")

    validated: list[CompiledSubAgent] = []
    names: set[str] = set()
    for spec in subagents:
        name = spec.get("name")
        description = spec.get("description")
        runnable = spec.get("runnable")
        if not isinstance(name, str):
            raise TypeError("Subagent names must be strings.")
        if not name:
            raise ValueError("Subagent names must not be empty.")
        if name != name.strip():
            raise ValueError("Subagent names must not have surrounding whitespace.")
        if name in names:
            raise ValueError(f"Duplicate subagent name: {name!r}.")
        if not isinstance(description, str):
            raise TypeError(f"Subagent {name!r} must have a string description.")
        if not description.strip():
            raise ValueError(f"Subagent {name!r} must have a description.")
        if not callable(getattr(runnable, "invoke", None)) or not callable(
            getattr(runnable, "ainvoke", None)
        ):
            raise TypeError(
                f"Subagent {name!r} must provide invoke and ainvoke methods."
            )
        names.add(name)
        validated.append(_adapt_subagent(spec, recorder))
    return validated


def _validate_main_tools(main_tools: Sequence[BaseTool]) -> None:
    names = [getattr(item, "name", "") for item in main_tools]
    if any(not name for name in names):
        raise ValueError("Every supervisor tool must have a non-empty name.")
    reserved = {
        "ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep",
        "execute", "task", "write_todos", "search_tools", "load_tools",
    }
    if conflicts := sorted(reserved.intersection(names)):
        formatted = ", ".join(repr(name) for name in conflicts)
        raise ValueError(
            f"Supervisor tool name(s) {formatted} are reserved for middleware."
        )
    if len(names) != len(set(names)):
        raise ValueError("Supervisor tool names must be unique.")


def construct_main_agent_graph(
    llm: Any,
    *,
    subagents: Sequence[CompiledSubAgent] | None = None,
    main_tools: list[BaseTool] | None = None,
    backend: BackendProtocol | None = None,
    tool_registry: ToolRegistry | None = None,
    skills: Sequence[str] | None = None,
    skill_dirs: Sequence[str] | None = None,
    discover_skills: bool = True,
    user_skills_dir: str | None = None,
    interrupt_on: dict[str, Any] | None | object = _DEFAULT_INTERRUPT_POLICY,
    recursion_limit: int = 200,
    human_supervised: bool = False,
    subagent_tools: list[BaseTool] | None = None,
    subagent_system_prompt: str | None = None,
    subagent_formatter_prompt: str | None = None,
    subagent_report_prompt: str | None = None,
    subagent_structured_output: bool = False,
    subagent_generate_report: bool = False,
    subagent_max_retries: int = 1,
    subagent_human_supervised: bool = False,
    subagent_terminal_tool_names: Collection[str] = (),
    enable_deepagent: bool = False,
    deepagent_backend: BackendProtocol | None = None,
    deepagent_skills: Sequence[str] | None = None,
    deepagent_discover_skills: bool = True,
    deepagent_user_skills_dir: str | None = None,
    deepagent_skill_dirs: Sequence[str] | None = None,
    deepagent_recursion_limit: int = 200,
    deepagent_system_prompt: str = DEFAULT_DEEPAGENT_PROMPT,
    system_prompt: str = DEFAULT_MAIN_AGENT_PROMPT,
    checkpointer: Any | None = None,
    subagent_recorder: SubagentRunRecorder | None = None,
):
    """Construct an independent main agent with direct tools and optional delegation.

    No backend means checkpoint-backed files and no shell. Workspace and skill
    capabilities share DeepAgent's setup, while the graph is built here with
    ``create_agent``. ``tool_registry=None`` selects non-interactive built-ins;
    an empty registry disables discovery. ``interrupt_on=None`` disables direct
    action review for callers explicitly managing their own approval boundary.
    Compiled workers retain their own tools and review policies. Use
    ``MainAgentSession`` to drive and restore durable threads.
    """
    if recursion_limit <= 0:
        raise ValueError("recursion_limit must be positive.")
    if deepagent_recursion_limit <= 0:
        raise ValueError("deepagent_recursion_limit must be positive.")
    if deepagent_backend is not None and not enable_deepagent:
        raise ValueError("deepagent_backend requires enable_deepagent=True.")
    if deepagent_skills and not enable_deepagent:
        raise ValueError("deepagent_skills requires enable_deepagent=True.")

    if deepagent_skill_dirs and not enable_deepagent:
        raise ValueError("deepagent_skill_dirs requires enable_deepagent=True.")

    if subagents is None:
        worker_kwargs: dict[str, Any] = {
            "tools": subagent_tools,
            "structured_output": subagent_structured_output,
            "generate_report": subagent_generate_report,
            "max_retries": subagent_max_retries,
            "human_supervised": subagent_human_supervised,
            "terminal_tool_names": subagent_terminal_tool_names,
            "checkpointer": None,
        }
        if subagent_system_prompt is not None:
            worker_kwargs["system_prompt"] = subagent_system_prompt
        if subagent_formatter_prompt is not None:
            worker_kwargs["formatter_prompt"] = subagent_formatter_prompt
        if subagent_report_prompt is not None:
            worker_kwargs["report_prompt"] = subagent_report_prompt
        worker = construct_single_agent_graph(llm, **worker_kwargs)
        registered_subagents: list[CompiledSubAgent] = [
            {
                "name": "chemgraph",
                "description": (
                    "Executes computational chemistry and molecular simulation "
                    "tasks with the existing ChemGraph single-agent workflow."
                ),
                "runnable": worker,
            }
        ]
    else:
        registered_subagents = list(subagents)

    if enable_deepagent:
        workspace_agent = construct_deep_agent_graph(
            llm,
            tools=[],
            skills=deepagent_skills,
            skill_dirs=deepagent_skill_dirs,
            discover_skills=deepagent_discover_skills,
            user_skills_dir=deepagent_user_skills_dir,
            system_prompt=deepagent_system_prompt,
            backend=(
                deepagent_backend
                if deepagent_backend is not None
                else StateBackend()
            ),
            checkpointer=None,
            recursion_limit=deepagent_recursion_limit,
            name="deepagent",
        )
        registered_subagents.append(
            {
                "name": "deepagent",
                "description": (
                    "Explores repositories and workspaces, edits files, runs tests, "
                    "and completes long multi-step coding or data-analysis tasks. "
                    "Use chemgraph instead for molecular simulations."
                ),
                "runnable": workspace_agent,
            }
        )

    validated_subagents = _validate_subagents(registered_subagents, subagent_recorder)
    supervisor_tools = list(main_tools or [])
    _validate_main_tools(supervisor_tools)
    if tool_registry is None:
        tool_registry = default_tool_registry(
            supervisor_tools, human_supervised=human_supervised,
        )
    effective_backend, _, workspace_middleware, policy = prepare_workspace_runtime(
        backend=backend, tools=supervisor_tools, tool_registry=tool_registry,
        skills=skills, skill_dirs=skill_dirs, discover_skills=discover_skills,
        user_skills_dir=user_skills_dir, interrupt_on=interrupt_on,
    )
    middleware = [
        TodoListMiddleware(),
        *workspace_middleware,
        FilesystemMiddleware(backend=effective_backend),
        create_summarization_middleware(llm, effective_backend),
        PatchToolCallsMiddleware(),
    ]
    middleware.append(SubAgentMiddleware(
        backend=effective_backend,
        subagents=validated_subagents,
        private_state_keys=private_state_field_names(
            *(item.state_schema for item in middleware if item.state_schema)
        ),
        task_description=(
            "Delegate a self-contained task to a configured specialist. "
            "Use direct tools for focused work. Do not run workspace mutations "
            "in parallel. Available agents:\n{available_agents}"
        ),
    ))
    if policy:
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=policy))
    graph = create_agent(
        model=llm,
        tools=supervisor_tools,
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
        name="main_agent",
    )
    return graph.with_config({"recursion_limit": recursion_limit})


__all__ = [
    "DEFAULT_DEEPAGENT_PROMPT",
    "DEFAULT_MAIN_AGENT_PROMPT",
    "construct_main_agent_graph",
    "latest_assistant_text",
]
