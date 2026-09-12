# Python and CLI workflows

Use the installed CLI's `chemgraph run --help` for available options. Workflow
`deep_agent` is standalone; `main_agent` is a supervisor with a chemistry worker
and an optional workspace worker enabled by `--deepagent`.

```sh
chemgraph run --interactive --workflow deep_agent --deepagent-workspace .
chemgraph run --interactive --workflow main_agent --deepagent --deepagent-workspace .
```

Attach an externally managed MCP server with `--mcp-url URL`. Starting a server
and attaching to one are separate actions. Do not assume that an HTTP MCP server
shares the CLI machine's filesystem.

In Python, use `ChemGraph` from `chemgraph.agent.llm_agent`, supply the intended
`workflow_type`, and call `await agent.run(query)`. For a local Deep Agent
workspace, supply a `deepagents.backends.LocalShellBackend` as
`deepagent_backend`. The default state backend has no shell. Shell access follows
the existing approval policy; preserve structured interrupts when resuming.

Bundled skills load automatically. Personal `~/.chemgraph/skills/` and workspace
`.agents/skills/` directories are discovered for supported local backends.
`deepagent_skills` / repeated `--deepagent-skill` values are additional,
backend-relative sources, with later sources overriding earlier ones.

For chemistry tool artifacts, relative writes use `CHEMGRAPH_LOG_DIR` through
`chemgraph.tools.ase_core._resolve_path`; readers use `_resolve_existing_path`.
Prefer absolute paths when crossing process boundaries. A local virtual
`/workspace/file.xyz` maps to the configured host workspace for shell commands,
but that mapping does not establish visibility on a remote MCP server.

Select HPC execution using the existing `[execution]` configuration and installed
extras. Do not treat a ChemGraph Parsl/Globus execution backend as a Deep Agents
filesystem backend. They implement different interfaces.
