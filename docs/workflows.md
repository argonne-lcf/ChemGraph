# Workflows

Select an agent architecture with `--workflow` on the CLI or `workflow_type` in
Python. `single_agent` is the recommended first choice.

| Workflow | Purpose | Requirements and constraints |
| --- | --- | --- |
| `single_agent` | One general chemistry agent with local tools | Default; CLI and Python |
| `main_agent` | Durable supervisor with checkpointed subagents | Interactive CLI or `MainAgentSession` only |
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
discovery surface, but `main_agent` is interactive-only and some specialized
workflows depend on local resources. Run `chemgraph --help` and
`chemgraph models` against the installed release instead of assuming every
workflow is available in every environment.
