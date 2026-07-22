"""Reasoning-engine registry.

Every ``llm_decide`` node picks an engine to actually run the turn. From
the outside they all look the same: message + tools + state in, tool
calls emitted, control returns. Inside, one might be a single ChemGraph
ReAct loop, another might be a multi-agent LangGraph with planner and
executor, another might be raw DSPy.

## The plug point

Add ``params.engine: "<name>"`` on an ``llm_decide`` node. When the
runtime hits that node, it looks the name up in the registry and calls
the resolved runner. Default when unset: ``chemgraph.single_agent``,
which preserves today's behavior bit-for-bit.

## Contract

An engine is any callable that matches ``chemgraph.agent.turn.run_turn``'s
kwargs shape (query, tools, model_name, base_url, api_key, argo_user,
system_prompt, recursion_limit, thread_id, terminal_tool_names,
on_event) → returns something with attributes
``final_text``, ``executed_tool_names``, ``terminal_tool``, ``thread_id``.

## Registering a new engine

    @register_engine("myteam.custom")
    async def _my_engine(**kwargs):
        # your reasoning loop
        return MyResult(...)

Users then set ``params.engine = "myteam.custom"`` on any llm_decide node.
No changes to agent.py, no changes to behavior_graph.py, no canvas
changes required (though the canvas dropdown reads the registry so new
engines appear automatically).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

DEFAULT_ENGINE_NAME = "chemgraph.single_agent"

Engine = Callable[..., Awaitable[Any]]
_REGISTRY: dict[str, Engine] = {}
# Lazy engine factories -- registered with a zero-arg loader that returns
# the real callable on first use. Lets us keep heavy imports (ChemGraph,
# torch, etc.) out of module-load time.
_LAZY_REGISTRY: dict[str, Callable[[], Engine]] = {}


def register_engine(name: str) -> Callable[[Engine], Engine]:
    """Decorator: register a reasoning engine under ``name``.

        @register_engine("myteam.custom")
        async def _engine(**kwargs): ...
    """
    def _decorate(fn: Engine) -> Engine:
        _REGISTRY[name] = fn
        return fn
    return _decorate


def register_lazy_engine(name: str, loader: Callable[[], Engine]) -> None:
    """Register a factory that returns the real engine on first use.

    Use when the engine's implementation lives in a heavy dependency
    (e.g. ChemGraph, LangGraph) that we don't want to import until the
    operator actually asks for it. See _register_builtin_engines() below.
    """
    _LAZY_REGISTRY[name] = loader


def resolve_engine(name: str | None) -> Engine:
    """Return the callable registered under ``name``.

    None → default engine. Unknown name → RuntimeError with the list of
    known engines so operators see immediately what typo they made.
    """
    key = name or DEFAULT_ENGINE_NAME
    if key in _REGISTRY:
        return _REGISTRY[key]
    if key in _LAZY_REGISTRY:
        engine = _LAZY_REGISTRY[key]()
        _REGISTRY[key] = engine
        return engine
    raise RuntimeError(
        f"unknown reasoning engine {key!r}. "
        f"Registered: {sorted(list_engines())}. "
        f"Register a new engine with @register_engine in a plugin module."
    )


def list_engines() -> list[str]:
    """All engine names, both eager and lazy."""
    return sorted(set(_REGISTRY) | set(_LAZY_REGISTRY))


# --- Built-in engines --------------------------------------------------

def _register_builtin_engines() -> None:
    """Register the engines that ship with swarm.

    Kept as a function (not module-scope calls) so tests can reset the
    registry cleanly and so lazy loaders don't fire at import time.
    """
    register_lazy_engine("chemgraph.single_agent", _load_single_agent)
    register_lazy_engine("chemgraph.multi_agent", _load_multi_agent)


def _load_single_agent() -> Engine:
    """Today's default: chemgraph.agent.turn.run_turn (single ReAct loop)."""
    try:
        from chemgraph.agent.turn import run_turn
    except ImportError as exc:
        raise RuntimeError(
            "engine 'chemgraph.single_agent' requires the chemgraphagent "
            "package. Install swarm with the chemgraph extra "
            "(pip install 'swarm[chemgraph]') or register a different "
            "engine as the default."
        ) from exc
    return run_turn


def _load_multi_agent() -> Engine:
    """PI's academy_sim shape: planner+executor as a single ChemGraph
    multi-agent workflow, with cross-wake conversation memory.

    We construct ``chemgraph.agent.llm_agent.ChemGraph`` directly (rather
    than using ``make_configured_graph_runner``) because we need a
    reference to the underlying workflow to inspect its LangGraph
    checkpoint between academy wakes. Purpose: multi_agent's MemorySaver
    persists planner/executor state per thread_id, so reusing the same
    ChemGraph + thread_id across wakes lets the planner remember which
    pipeline step it just finished instead of restarting at step 1.

    When the graph reaches END (planner said "done"), we rotate the
    thread_id so the next inject starts a fresh conversation. That
    avoids the "cached-at-END = subsequent invokes are no-ops" trap
    from earlier iterations.
    """
    try:
        from chemgraph.agent.llm_agent import ChemGraph  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "engine 'chemgraph.multi_agent' requires "
            "chemgraph.agent.llm_agent.ChemGraph (from the "
            "dev-academy-sim branch). Check out that branch on the "
            "compute nodes' venv and this engine becomes usable with "
            "no swarm-side changes."
        ) from exc
    return _make_multi_agent_engine_v2()


_MULTI_AGENT_PLANNER_PROMPT = """\
You are the Planner of a workflow that must complete an operator-defined mission end-to-end. Your job is to dispatch tool calls to the Executor and FINISH only after EVERY step in the mission is done -- including peer messaging and file transfers, not just local science tools.

MISSION (verbatim, this is the source of truth for what "done" means):
{mission}

STATE TRANSITIONS:

PHASE 1 (first invocation): parse the mission's numbered steps. Set `next_step` to "executor_subgraph". Generate ONE task per mission step, in order. Each task's prompt should tell the executor exactly which tool to call and with what args, quoting concrete values from the operator's input message.

PHASE 2 (subsequent invocations): read the conversation history for executor results.
- If the LAST completed mission step is not the final one, dispatch the NEXT step as a new task.
- Do NOT re-dispatch steps that already completed successfully in the history.
- Do NOT FINISH just because local science tools succeeded -- the mission includes peer coordination (send_message) and file transfers (transfer_file) that MUST run before finishing.
- FINISH only when the mission's final step (typically send_message reply or finish_turn) has completed.

PHASE 2a (failed tasks): if a task failed, retry with the SAME task_index and a corrected prompt (fix bad args, change tool). Do not skip a step; if truly unrecoverable, note in thought_process and continue.

OUTPUT FORMAT: return ONLY valid JSON, no markdown, no prose.

Dispatch:
{{
  "thought_process": "brief reasoning; name which mission step this task advances",
  "next_step": "executor_subgraph",
  "tasks": [
    {{"task_index": <N>, "prompt": "<explicit tool call the executor should make>"}}
  ]
}}

Finish (only when EVERY mission step has completed):
{{
  "thought_process": "mission complete: <one-line summary of what was accomplished, including any peer replies>",
  "next_step": "FINISH"
}}
"""


_CG_CACHE: dict[str, Any] = {}
_MULTI_AGENT_ITERATIONS_PATCHED = False


def _patch_multi_agent_iterations(default: int) -> None:
    """Rewrite ``unified_planner_router.__defaults__`` so the router uses
    a higher ``max_planner_iterations`` than the hardcoded 3. Idempotent
    per process. Signature is positional-with-defaults today:
    ``(state, structured_output=False, max_planner_iterations=3,
    max_task_retries=2, human_supervised=False)`` -- so max_planner_iterations
    is at index 1 in __defaults__.
    """
    global _MULTI_AGENT_ITERATIONS_PATCHED
    if _MULTI_AGENT_ITERATIONS_PATCHED:
        return
    from chemgraph.graphs import multi_agent as _ma  # type: ignore
    fn = _ma.unified_planner_router
    defaults = list(fn.__defaults__ or ())
    # Guardrail: if the signature ever moves the arg, skip silently
    # rather than corrupt other defaults.
    if len(defaults) >= 2 and defaults[1] == 3:
        defaults[1] = default
        fn.__defaults__ = tuple(defaults)
    _MULTI_AGENT_ITERATIONS_PATCHED = True
# Track the active LangGraph thread_id per swarm thread_id. When the
# graph reaches END we rotate this (thread_id-1 -> thread_id-2) so the
# next operator inject starts a fresh conversation while previous
# conversations remain checkpointed for inspection.
_ACTIVE_LG_THREAD: dict[str, str] = {}
_LG_ROTATION: dict[str, int] = {}


def _make_multi_agent_engine(make_runner) -> Engine:
    """Legacy adapter: wraps a make_runner callable (the older
    make_configured_graph_runner shape). Retained for tests that mock
    make_runner. Production uses _make_multi_agent_engine_v2 which
    talks to ChemGraph directly for cross-wake memory.
    """
    async def _engine(*, query, tools, model_name, base_url, api_key, argo_user,
                      system_prompt, recursion_limit, thread_id,
                      terminal_tool_names, on_event):
        from chemgraph.models.settings import LLMSettings  # type: ignore
        effective_model = model_name
        if base_url and "argoapi" in base_url:
            m = model_name[len("argo:"):] if model_name.startswith("argo:") else model_name
            effective_model = f"argo:{m.lower()}"
        on_event("multi_agent_engine_invoked", {
            "requested_model": model_name, "effective_model": effective_model,
            "base_url": base_url, "argo_user_set": bool(argo_user),
        })
        lm = LLMSettings(model=effective_model, base_url=base_url,
                         api_key=api_key, user=argo_user)
        spec = _MultiAgentSpec(system_prompt=system_prompt)
        runner = await make_runner(
            spec=spec, llm_settings=lm, extra_tools=tuple(tools),
            trace=on_event, terminal_tool_names=tuple(terminal_tool_names),
        )
        result = await runner(query)
        executed = tuple(result.executed_tool_names or ())
        terminal = result.terminal_tool
        if not executed:
            executed = ("finish_turn",)
            terminal = terminal or "finish_turn"
            on_event("multi_agent_synthesized_finish_turn", {
                "reason": "multi_agent returned text without action tools",
                "output_preview": (result.output or "")[:200],
            })
        class _A:
            pass
        _A.final_text = result.output
        _A.executed_tool_names = executed
        _A.terminal_tool = terminal
        _A.thread_id = thread_id
        return _A
    return _engine


def _make_multi_agent_engine_v2() -> Engine:
    """Build an adapter that satisfies our engine contract.

    Our contract (matches chemgraph.agent.turn.run_turn):
      async def engine(*, query, tools, model_name, base_url, api_key,
                       argo_user, system_prompt, recursion_limit,
                       thread_id, terminal_tool_names, on_event)
          -> object with .final_text, .executed_tool_names,
                        .terminal_tool, .thread_id

    Cross-wake memory: caches the ChemGraph object per swarm thread_id
    so multi_agent's MemorySaver persists planner+executor state across
    academy wakes. Between invokes we inspect ``workflow.get_state()``:
    if the current LangGraph thread is at END (empty ``next``), rotate
    to a fresh LangGraph thread_id so the next inject starts a new
    conversation instead of hitting the "already-finished" no-op trap.
    """
    async def _engine(
        *,
        query: str,
        tools,
        model_name: str,
        base_url: str,
        api_key: str,
        argo_user: str,
        system_prompt: str,
        recursion_limit: int,
        thread_id: str,
        terminal_tool_names,
        on_event,
    ):
        from chemgraph.models.settings import LLMSettings  # type: ignore
        from chemgraph.agent.llm_agent import ChemGraph  # type: ignore
        # dev-academy-sim's load_openai_model gates on the model name:
        # the name must be in supported_argo_models (which uses the
        # exact form `argo:gpt-4.1-mini`, lowercase, hyphenated,
        # argo:-prefixed) or the user parameter never lands on the
        # Argo request and the endpoint 500s.
        effective_model = model_name
        if base_url and "argoapi" in base_url:
            m = model_name[len("argo:"):] if model_name.startswith("argo:") else model_name
            m = m.lower()
            effective_model = f"argo:{m}"
        # Diagnostic: emit what the adapter actually sends so we can
        # verify from events.jsonl whether normalization landed.
        on_event("multi_agent_engine_invoked", {
            "requested_model": model_name,
            "effective_model": effective_model,
            "base_url": base_url,
            "argo_user_set": bool(argo_user),
        })
        lm = LLMSettings(
            model=effective_model, base_url=base_url,
            api_key=api_key, user=argo_user,
        )
        # Build ChemGraph once per swarm thread_id, cache it. MemorySaver
        # lives inside cg.workflow so cache reuse = memory persistence.
        cg = _CG_CACHE.get(thread_id)
        if cg is None:
            # ponytail: monkeypatch chemgraph.multi_agent's hardcoded
            # max_planner_iterations=3. Multi_agent forces FINISH after
            # 3 planner dispatches, which caps end-to-end pipelines at
            # 3 tool calls. Our mission needs 6 (build/opt/charges/
            # transfer/message/wait). No constructor kwarg exposes it,
            # so we bump the function default. Upgrade path: ask
            # dev-graspa to expose it, drop the patch.
            _patch_multi_agent_iterations(default=20)
            # Override chemgraph's default planner_prompt: it's hardcoded
            # for single-molecule chemistry ("Each subtask must correspond
            # to calculating a property of a single molecule only") and
            # treats any local science tool completion as FINISH -- which
            # means it never dispatches transfer_file or send_message
            # steps. Our system_prompt (== the agent mission) has the
            # actual step list; use it as the planner brief too.
            planner_override = _MULTI_AGENT_PLANNER_PROMPT.format(
                mission=system_prompt,
            )
            cg = ChemGraph(
                model_name=effective_model,
                workflow_type="multi_agent",
                base_url=lm.base_url,
                api_key=lm.api_key,
                argo_user=lm.user,
                tools=list(tools),
                on_event=on_event,
                terminal_tool_names=tuple(terminal_tool_names),
                system_prompt=system_prompt,
                planner_prompt=planner_override,
                max_retries=3,
            )
            _CG_CACHE[thread_id] = cg
            _LG_ROTATION[thread_id] = 0
            _ACTIVE_LG_THREAD[thread_id] = f"{thread_id}-0"
        # Rotation: if the current LangGraph thread finished (next=()),
        # move to a fresh thread_id so this invoke starts a new plan
        # instead of returning immediately from an END checkpoint.
        lg_thread = _ACTIVE_LG_THREAD[thread_id]
        try:
            state = cg.workflow.get_state({"configurable": {"thread_id": lg_thread}})
            finished = not state.next
        except Exception:
            finished = False  # No checkpoint yet -> first run, don't rotate
        if finished:
            _LG_ROTATION[thread_id] += 1
            lg_thread = f"{thread_id}-{_LG_ROTATION[thread_id]}"
            _ACTIVE_LG_THREAD[thread_id] = lg_thread
            on_event("multi_agent_thread_rotated", {
                "swarm_thread_id": thread_id,
                "new_lg_thread": lg_thread,
                "reason": "prior LangGraph thread reached END",
            })
        # Run. cg.run keys the checkpointer by config.thread_id.
        message = await cg.run(query, config={"thread_id": lg_thread})
        # Adapt to our contract. cg.run returns the last message; we
        # need executed_tool_names + terminal_tool. cg exposes them via
        # its on_event stream (workflow_finished payload), captured by
        # a small trace wrapper in the graph_runtime adapter. Here we
        # don't have that plumbing, so pull from the final state.
        final_state = cg.workflow.get_state({"configurable": {"thread_id": lg_thread}}).values
        executed = tuple(
            str(m.name) for m in final_state.get("messages", [])
            if getattr(m, "type", None) == "tool" and getattr(m, "name", None)
        )
        terminal = None
        for tn in reversed(executed):
            if tn in tuple(terminal_tool_names):
                terminal = tn
                break

        # Fake a result object matching our earlier return shape.
        class _R:
            output = getattr(message, "content", None) or str(message)
        _R.executed_tool_names = executed
        _R.terminal_tool = terminal
        result = _R
        # Adapt his GraphRunResult (output, executed_tool_names,
        # terminal_tool) to our contract's return shape. Callers read
        # .final_text / .executed_tool_names / .terminal_tool / .thread_id.
        #
        # Multi_agent's internal LangGraph is designed to be self-contained:
        # planner + executor may produce a final text answer without calling
        # any Academy action tools (send_message / finish_turn). Swarm's
        # run_academy_turn guard requires at least one action-tool call per
        # turn (otherwise it thinks the LLM silently did nothing). Bridge
        # the mismatch by synthesizing a `finish_turn` entry when the inner
        # graph produced text but no explicit terminal tool. This matches
        # the swarm's world model without dropping any real tool calls the
        # multi_agent workflow made.
        executed = tuple(result.executed_tool_names or ())
        terminal = result.terminal_tool
        if not executed:
            executed = ("finish_turn",)
            terminal = terminal or "finish_turn"
            on_event("multi_agent_synthesized_finish_turn", {
                "reason": "multi_agent returned text without action tools",
                "output_preview": (result.output or "")[:200],
            })
        class _Adapted:
            pass
        _Adapted.final_text = result.output
        _Adapted.executed_tool_names = executed
        _Adapted.terminal_tool = terminal
        _Adapted.thread_id = thread_id
        return _Adapted
    return _engine


class _MultiAgentSpec:
    """Minimal ConfiguredGraphSpec for the multi_agent workflow.

    Fields chosen to satisfy the Protocol without requiring the caller
    to construct a full spec object. Planner and executor prompts are
    left as None so ChemGraph's multi_agent defaults apply; callers
    who want custom prompts should pass system_prompt_override on the
    llm_decide node (which lands here as `system_prompt`).

    ``workflow_kwargs`` includes ``max_retries=3`` because the default
    of 1 isn't enough for the planner's JSON parser to recover when the
    LLM rambles before emitting the required schema. Three gives the
    LLM a chance to self-correct without turning the demo into a
    heisenbug when Argo returns a slightly chatty response.
    """
    workflow_type = "multi_agent"
    workflow_kwargs: dict = {"max_retries": 3}
    science_tools: tuple = ()
    planner_prompt: str | None = None
    executor_prompt: str | None = None
    name = "chemgraph.academy.multi_agent"

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt


_register_builtin_engines()
