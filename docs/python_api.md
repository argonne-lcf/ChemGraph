# Python API

Use the Python API to embed ChemGraph in notebooks, services, or larger
workflows. The standard agent is asynchronous.

## Run a single-agent query

```python
import asyncio

from chemgraph.agent.llm_agent import ChemGraph


async def main():
    agent = ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        return_option="last_message",
    )
    result = await agent.run(
        "Build water from SMILES O, optimize it with EMT, and report the energy."
    )
    print(result.content)


asyncio.run(main())
```

In an async notebook or application, call `await agent.run(...)` directly
instead of starting a second event loop.

## Token usage and per-call limits

`ChemGraph.run` honors an explicit positive integer `config["recursion_limit"]`
before the agent's configured limit and does not mutate your configuration.
The default remains 200 graph steps; this is not a model-call or token budget.

```python
result = await agent.run(query, config={"recursion_limit": 350})
print(agent.last_usage)     # current user turn, including approvals and retries
print(agent.session_usage)  # all recorded turns in this session

if agent.session_store is not None:
    session_counts = agent.session_store.get_usage(agent.session_id)
    turn_counts = agent.session_store.get_usage(
        agent.session_id, turn_id=agent.last_usage["turn_id"]
    )
```

Usage mappings include `input_tokens`, `output_tokens`, `total_tokens`,
`cached_input_tokens`, `reasoning_output_tokens`, `call_count`,
`incomplete_calls`, `partial`, and per-field `unreported_counts`. Counts are
known subtotals; unknown fields are `None`. Cache/reasoning details are subsets
of input/output. A store query with no recorded usage has `recorded=False`;
old conversation history is not retroactively counted.

Session summaries also expose `history_unaccounted`. When true, `partial` stays
true regardless of complete counts for new turns. No calls or tokens are
estimated for the missing history. On database upgrade, existing transcripts
without the coverage marker are conservatively flagged, even if some usage rows
exist; newly created sessions start with complete coverage. Individual new-turn
summaries are independent of that historical gap.

`MainAgentSession` exposes the same `last_usage` and `session_usage` properties;
its turn result also includes `usage`. `resume()` and `retry()` retain the
original usage turn. `run_turn` returns `usage` and accepts an optional
`session_store`; without it, it does not write a session database. Setting
`enable_memory=False` on `ChemGraph` retains in-memory counters only.

Accounting runs independently of `on_event`. Model-finished events retain their
existing payloads and include a `call_id` and provider `token_counts` when
available. Available usage remains readable after a workflow fails. Python
callers decide how to display it; the CLI prints numeric counters locally,
without making another model call.

## Return values

Use `return_option="last_message"` for the final message object or
`return_option="state"` for the full graph state. Full state is useful when an
application must inspect tool calls, messages, or structured output.

```python
agent = ChemGraph(return_option="state")
state = await agent.run("What is the SMILES string for aspirin?")
```

## Threads and checkpoints

Pass graph configuration when a workflow needs a stable thread identity:

```python
config = {"configurable": {"thread_id": "my-run-001"}}
result = await agent.run("Continue the analysis.", config=config)
```

Choose thread IDs that are unique in your application and do not contain
credentials or sensitive user data.

## Main-agent sessions

The checkpointed `main_agent` is not run through `ChemGraph.run()`. Import and
construct `MainAgentSession` from `chemgraph.agent.main_session`, then use its
session-oriented async methods. This API is
intended for durable, interactive supervisor workflows; consult the class
docstrings in the installed version for constructor and persistence options.

`MainAgentSession` accepts an optional `on_event` callback with the signature
`(event_name, payload)`. Tagged `tool_call_started` payloads include
`subagent_name`, allowing callers to distinguish delegated tool activity from
supervisor tools. The supervisor can use `read_file` for checkpoint-backed
files returned by subagents, but this does not expose host files or session
artifacts.

For CLI use, the equivalent is:

```bash
chemgraph run --interactive --workflow main_agent
```

## Workspace Deep Agent

The workspace workflow is separately callable through `ChemGraph.run()`:

```python
import os

from deepagents.backends import LocalShellBackend

from chemgraph.agent.llm_agent import ChemGraph

agent = ChemGraph(
    model_name="claude-sonnet-4-20250514",
    workflow_type="deep_agent",
    deepagent_backend=LocalShellBackend(
        root_dir="/path/to/checkout",
        virtual_mode=True,
        env={
            name: os.environ[name]
            for name in ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "TMPDIR")
            if name in os.environ
        },
        inherit_env=False,
    ),
    deepagent_skills=[
        "/workspace/shared-skills/",
        "/workspace/.agents/skills/",
    ],
)
result = await agent.run(
    "Review the test failures.",
    config={"configurable": {"thread_id": "workspace-review"}},
)
```

A virtual `LocalShellBackend` is exposed to the agent at `/workspace`, so file
tool path `/workspace/src/example.py` maps directly to
`/path/to/checkout/src/example.py`. The generated Deep Agents system context
also supplies that host path for shell commands. Custom backends, existing
composite routes, and non-virtual local execution are preserved when adding
the bundled skill route.

Bundled skills load without configuration. For supported local workspaces,
personal `~/.chemgraph/skills/` and project `.agents/skills/` directories are
also discovered. `deepagent_skill_dirs=["../shared-skills/"]` mounts host collections independently
of the workspace. `deepagent_skills` adds ordered POSIX backend-relative sources;
later sources override matching names. Set `deepagent_discover_skills=False`
to use only bundled and explicit sources. On `construct_deep_agent_graph`, the
same options are `skills=` and `discover_skills=`. See [skills](skills.md) for
custom backends, source precedence, and session behavior.

With `StateBackend`, only explicitly supplied state-backed skills need files
seeded in graph state. Bundled resources work without seeding. Standalone and
supervisor-hosted Deep Agents share `DEFAULT_DEEPAGENT_PROMPT`, which permits
using attached chemistry tools. Tool availability is configured separately;
the built-in supervisor workspace worker has no chemistry tools attached.
`PromptConfig.deepagent=None` selects the shared default; a supplied string,
including an empty string, is preserved.

The default approval policy interrupts before shell commands and file
mutations. Without a `human_input_handler`, `run()` raises
`HumanInputRequired`; its `payload` retains the structured Deep Agents action
requests and must be resumed with matching structured decisions. Its
`interrupts` tuple retains every pending request and its LangGraph interrupt
ID; callers must resume concurrent requests with an exact mapping from those
IDs to responses. `question` and `payload` continue to describe the first
request for compatibility. A configured handler may use the legacy
`handler(question)` signature or `handler(question, payload)` when it needs the
raw structured request; both synchronous and asynchronous handlers are
supported. Setting
`deepagent_auto_approve=True` removes this boundary and should be limited to an
externally isolated, explicitly trusted workspace.

Handlers can reject an action with feedback using the existing decision format:

```python
response = {"decisions": [{"type": "reject", "message": "Use EMT instead of MACE."}]}
```

The rejected tool is not executed; its feedback is returned to the model, and
revised tool calls follow the normal approval policy. Supply one decision per
action in request order. The CLI builds this response from typed guidance;
Enter at a CLI review approves only that action, not future actions.

The same constructor can be composed as a worker:

```python
from chemgraph.graphs.deep_agent import construct_deep_agent_graph
from chemgraph.registry import AgentRegistry

standalone_graph = construct_deep_agent_graph(model, backend=backend)
worker = AgentRegistry().as_subagent(
    "deepagent",
    llm=model,
    backend=backend,
    skills=["/workspace/.agents/skills/"],
)
```

`as_subagent()` compiles the graph with `checkpointer=None` so it inherits its
parent checkpoint. The registry returns the canonical worker name `deep_agent`,
even when requested through the `deepagent` alias; use `worker["name"]` when
composing task calls. `construct_main_agent_graph(enable_deepagent=True, ...)`
uses this same workflow under the stable subagent name `deepagent`.

Caller-owned asynchronous checkpointers, including `AsyncSqliteSaver`, are
supported by `await agent.run(...)` and `await agent.apersist_run_state(config)`.
Keep the saver open on the same event loop for the run and any resumes. The
synchronous state methods remain available for synchronous checkpointers.

## Custom tools

`ChemGraph` can be extended with compatible LangChain tools. Keep tools narrow,
validate their inputs, and avoid exposing destructive filesystem or shell
operations to untrusted prompts. Optional dependencies in application code
should be imported lazily so a core installation can still load.

## Human supervision

Supported workflows can pause for human input when supervision is enabled.
Design non-interactive applications so they do not unexpectedly wait forever,
and treat an approval boundary as part of the application's security model.

## Artifacts

Set `CHEMGRAPH_LOG_DIR` before constructing the agent to choose the parent
directory for session artifacts:

```python
import os

os.environ["CHEMGRAPH_LOG_DIR"] = "/absolute/path/to/runs"
```

## API stability

ChemGraph is evolving and does not currently re-export `ChemGraph` from the
package root. Prefer the documented module import, pin a version for deployed
applications, and check release notes before upgrading.
