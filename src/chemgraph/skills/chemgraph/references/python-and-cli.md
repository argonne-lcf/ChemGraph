# Python and CLI workflows

Use the installed CLI's `chemgraph run --help` for available options. Workflow
`deep_agent` is standalone; `main_agent` directly uses skills, workspace files,
shell commands, and local tools, with optional delegation. It keeps a chemistry
worker by default. Repeat `--subagent NAME` to replace that worker explicitly.

```sh
chemgraph run --interactive --workflow deep_agent --deepagent-workspace .
chemgraph run --interactive --workflow main_agent --workspace . --subagent single_agent
chemgraph run --interactive --workflow main_agent --workspace . --subagent deep_agent
```

For direct PBS calculations, write a workspace script following
[the ASE batch example](ase-batch.md) and use the `pbs-hpc` skill to submit it.
Run the Deep Agent on the submission host with a shared workspace; no `--mcp-url`
or ChemGraph execution-backend configuration is needed for this route.

Attach an externally managed MCP server with `--mcp-url URL`. Starting a server
and attaching to one are separate actions. Do not assume that an HTTP MCP server
shares the CLI machine's filesystem.

In Python, use `ChemGraph` from `chemgraph.agent.llm_agent`, supply the intended
`workflow_type`, and call `await agent.run(query)` for ordinary workflows. Drive
`main_agent` using `MainAgentSession(agent.workflow, session_metadata=agent.main_agent_metadata)`
and its `run`, `restore`, and `resume` methods. Supply `backend`, `skill_dirs`,
`skills`, `tool_registry`, `subagent_names`, and `subagent_options` for the main
agent. Caller-owned configurations need a matching non-secret `configuration_id`
for durable reconstruction; change it when their behavior changes. For a local Deep Agent
workspace, supply a `deepagents.backends.LocalShellBackend` as
`deepagent_backend`. The default state backend has no shell. Shell access follows
the existing approval policy; preserve structured interrupts when resuming.

Bundled skills load automatically. Personal `~/.chemgraph/skills/` and workspace
`.agents/skills/` directories are discovered for supported local backends.
Main-agent CLI `--skill`/TOML `skills` and DeepAgent
`--deepagent-skill`/TOML `deepagent_skills` values are host directories
relative to the invocation directory. Python `deepagent_skills` remains
backend-relative; `deepagent_skill_dirs` mounts host directories. Later sources
override earlier ones.

For chemistry tool artifacts, relative writes use `CHEMGRAPH_LOG_DIR` through
`chemgraph.tools.ase_core._resolve_path`; readers use `_resolve_existing_path`.
Prefer absolute paths when crossing process boundaries. A local virtual
`/workspace/file.xyz` maps to the configured host workspace for shell commands,
but that mapping does not establish visibility on a remote MCP server.

Select HPC execution using the existing `[execution]` configuration and installed
extras. Do not treat a ChemGraph Parsl/Globus execution backend as a Deep Agents
filesystem backend. They implement different interfaces.

Without an explicit main-agent workspace, file tools use checkpoints and no
shell is available. Registry tools still run on the host under their review
policy. `--tool NAME` restricts the main-agent or standalone DeepAgent catalog;
TOML `tools = []` disables discovery. Old graph checkpoints cannot resume after
the capabilities upgrade, but old transcripts remain readable.
