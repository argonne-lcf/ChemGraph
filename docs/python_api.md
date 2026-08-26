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
