# Tool and agent registries

ChemGraph provides explicit registries for discovering in-process tools and
worker graphs without eagerly importing every implementation. These registries
are intended to support orchestration and, later, controlled benchmarking.

The first registry release deliberately excludes MCP tools, skills, selector
middleware, and benchmark code. The `main_agent` graph is also excluded from
both registries: it remains the orchestration graph that consumes registered
workers.

## Tool registry

`ToolRegistry` contains the LangChain `BaseTool` objects implemented in
`chemgraph.tools`. The manifest stores import paths and metadata, so creating a
registry does not import optional tool modules.

```python
from chemgraph.registry import ToolRegistry

registry = ToolRegistry()

# Inspect metadata without loading tool implementations.
print(registry.names())
ase_specs = registry.specs(tags={"ase"})

# Resolve only the tools needed by a graph or model.
tools = registry.resolve(["molecule_name_to_smiles", "run_ase"])
```

Tools can be selected by one or more tags. A tool must contain every requested
tag to match:

```python
analysis_tools = registry.resolve(tags={"ase", "analysis"})
```

Some tools need optional Python packages, environment variables, or external
executables. Use `availability()` to inspect those requirements, or set
`require_available=True` to fail before loading the tool:

```python
status = registry.availability("run_docking")
if status.available:
    docking = registry.get("run_docking", require_available=True)
else:
    print(status.issues)
```

Passing `require_available=False` does not make an unavailable dependency work;
it only defers validation to the tool implementation.

## Agent registry

`AgentRegistry` contains ChemGraph worker graph constructors. It provides the
same canonical workflow names used by the command-line interface, except for
`main_agent`:

```python
from chemgraph.registry import AgentRegistry

registry = AgentRegistry()
print(registry.names())

worker = registry.build("single_agent", llm=model)
```

The registered workers are `single_agent`, `multi_agent`, `python_relp`,
`graspa`, `mock_agent`, `graspa_mcp`, `rag_agent`, `single_agent_xanes`, and
`molecular_docking`. Existing `python_repl` and `graspa_agent` spellings are
supported as aliases.

Standalone workers keep their existing default in-memory checkpointer. When a
worker is handed to an orchestration graph, use `as_subagent()` or
`as_subagents()`. These adapters compile the worker with `checkpointer=None` so
it inherits checkpointing from the parent graph:

```python
workers = registry.as_subagents(
    ["single_agent", "python_relp"],
    llm=model,
)

main_graph = construct_main_agent_graph(model, subagents=workers)
```

`as_subagents()` validates the whole requested set before constructing any
worker. Per-worker constructor arguments are supplied through `options`:

```python
workers = registry.as_subagents(
    ["single_agent", "graspa_mcp"],
    llm=model,
    options={
        "graspa_mcp": {
            "executor_tools": executor_tools,
            "analysis_tools": analysis_tools,
        }
    },
)
```

Both registries support explicit custom registration with `ToolSpec` or
`AgentSpec`. Duplicate names and aliases are rejected unless a caller
intentionally replaces a tool specification.
