# Workflows

Select an agent architecture with `--workflow` on the CLI or `workflow_type` in
Python. `single_agent` is the recommended first choice.

| Workflow | Purpose | Requirements and constraints |
| --- | --- | --- |
| `single_agent` | One general chemistry agent with local tools | Default; CLI and Python |
| `main_agent` | Durable supervisor with checkpointed subagents | Interactive CLI or `MainAgentSession` only |
| `deep_agent` | Repository exploration, coding, and workspace tasks | CLI and Python; alias `deepagent`; broad local shell access |
| `multi_agent` | Routes tasks among specialized agents | More model calls and orchestration overhead |
| `python_relp` | Chemistry agent with Python REPL capability | Executes Python in the current process; alias `python_repl` |
| `graspa` | gRASPA-oriented agent | Site-specific executable/configuration; alias `graspa_agent` |
| `mock_agent` | Deterministic development/testing route | Not intended for scientific work |
| `graspa_mcp` | gRASPA through MCP | Site and MCP setup required |
| `rag_agent` | Retrieval-augmented questions over documents | Install `chemgraph[rag]` |
| `single_agent_xanes` | XANES-focused single agent | Install `chemgraph[xanes]`; FDMNES and/or Materials Project access |
| `molecular_docking` | Protein-ligand docking workflow | Install `chemgraph[docking]`; Vina setup |

## Single agent

Start here for structure building, property lookup, ASE calculations, analysis,
and reports:

```bash
chemgraph run --workflow single_agent \
  -q "Optimize water with EMT and report its final energy."
```

It minimizes orchestration complexity while exposing the normal tool set.

## Main agent

`main_agent` manages longer-lived tasks through specialized subagents and
durable checkpoints. It is intentionally session-oriented:

```bash
chemgraph run --interactive --workflow main_agent
```

In Python, construct `MainAgentSession`; `ChemGraph.run()` rejects this workflow.
See [Python API](python_api.md).

## Deep Agent

`deep_agent` is one reusable workspace workflow with two entry points. It can
run directly through `ChemGraph(workflow_type="deep_agent")`, or it can be
registered under `main_agent` as the `deepagent` subagent. Both paths use
`construct_deep_agent_graph`, so the prompt, backend, tools, recursion limit,
and approval policy have one implementation.

```bash
# Direct, process-local interactive thread with action reviews.
chemgraph run --interactive --workflow deep_agent --deepagent-workspace .

# The same worker delegated by the durable supervisor.
chemgraph run --interactive --workflow main_agent --deepagent \
  --deepagent-workspace .
```

The standalone interactive workflow keeps one thread while that CLI process is
open; it does not provide cross-process restoration in this first version.
File mutations and shell commands require structured approve/reject decisions.
Headless execution is rejected unless both an explicit workspace and
`--deepagent-dangerously-skip-approvals` are supplied.

With the CLI's virtual local backend, `/workspace` is the project root exposed
to the Deep Agent. For example, `--deepagent-workspace test/` maps
`/workspace/script.py` to `test/script.py` on the host. Deep Agents also tells
the model the corresponding absolute host path to use with shell commands.
Older files created under `test/workspace/` are not moved automatically.

Run state and session records remain standard ChemGraph artifacts. By default,
state snapshots are written under the process's `cg_logs/session_*` directory
and session messages go to the configured ChemGraph session store; neither is
placed inside the selected Deep Agent workspace. Approval-interrupted direct
runs save both the pending checkpoint and the completed state after resumption.

!!! danger "Host shell access"
    The local backend can modify files under its workspace and its shell is not
    confined to that directory. Use only trusted prompts and disposable,
    isolated workspaces. The skip-approvals flag removes the action-review
    boundary for that run.

For comparisons with Claude Code or native Codex, keep the task set, starting
checkout, time budget, and scoring fixed, and record which runtime performed
each run. Selecting a `codex:` model here evaluates that model through
ChemGraph's Deep Agent harness; it does not reproduce the native Codex product
runtime. `deep_agent` is intentionally not included in the chemistry-focused
`chemgraph eval` workflow matrix.

## Multi-agent

`multi_agent` delegates among specialized graphs. It is useful when a request
crosses distinct chemistry capabilities, but consumes more model tokens and may
take more graph steps than `single_agent`.

## Python REPL

`python_relp` (spelling retained for compatibility) allows generated Python to
run inside the ChemGraph process.

!!! danger "Arbitrary code execution"
    Use this workflow only with trusted prompts and data in an isolated
    environment. Generated code can read, modify, or delete files accessible to
    the process and may invoke installed programs.

## Specialized workflows

RAG accepts supported text/PDF sources and may use provider embeddings or a
local fallback. XANES, docking, gRASPA, and MCP workflows need the corresponding
scientific engine, credentials, site configuration, or server. Consult
[Installation](installation.md), [Calculators](calculators.md), and
[MCP servers](mcp_servers.md) before selecting them.

## Interface compatibility

The Streamlit interface exposes a subset of workflows. The CLI is the broadest
discovery surface, but `main_agent` is interactive-only, `deep_agent` has an
explicit local-access safety boundary, and some specialized workflows depend
on local resources. Run `chemgraph --help` and
`chemgraph models` against the installed release instead of assuming every
workflow is available in every environment.
