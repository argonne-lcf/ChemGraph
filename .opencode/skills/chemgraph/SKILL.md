---
name: chemgraph
description: Develop, test, and extend ChemGraph -- an agentic framework for automated molecular simulations using LLMs, LangGraph, ASE, and MCP servers
license: Apache-2.0
compatibility: opencode
metadata:
  audience: developers
  workflow: development
---

## What is ChemGraph

ChemGraph is a Python framework (package name `chemgraphagent`) built at Argonne National Laboratory that automates computational chemistry workflows using LLMs. It connects natural language queries to molecular simulations via an agent architecture built on LangGraph/LangChain, ASE (Atomic Simulation Environment), RDKit, and MCP (Model Context Protocol) servers.

Key capabilities: molecule lookup (PubChem), 3D structure generation (RDKit), geometry optimization, vibrational analysis, thermochemistry, IR spectra, and HPC-scale ensemble simulations via Parsl.

## Project layout

```
ChemGraph/
  src/
    chemgraph/              # Core package
      agent/                # Main ChemGraph agent class (llm_agent.py)
      graphs/               # LangGraph workflow definitions (9 workflows)
      tools/                # LangChain tool implementations
      mcp/                  # FastMCP server implementations
      models/               # LLM provider integrations (OpenAI, Anthropic, Gemini, Groq, Ollama, ALCF, Argo)
      prompt/               # System prompt templates per model/workflow
      schemas/              # Pydantic data models (AtomsData, ASEInput/Output, calculators)
      memory/               # Session memory (SQLite-backed persistence, schemas)
      state/                # LangGraph state definitions
      hpc_configs/          # Parsl configs for ALCF Polaris/Aurora
      utils/                # Config, logging, evaluation utilities
    ui/                     # Streamlit web app (app.py) and Rich CLI (cli.py)
  tests/                    # pytest test suite (20+ files)
  scripts/                  # MCP examples, Parsl examples
  notebooks/                # Jupyter demo notebooks
  docs/                     # MkDocs documentation source
  config.toml               # Default runtime configuration
  pyproject.toml            # Package metadata and dependencies
  docker-compose.yml        # Multi-profile Docker (jupyter, streamlit, mcp, cli)
```

## Architecture overview

### Agent entry point

`src/chemgraph/agent/llm_agent.py` contains the `ChemGraph` class. This is the central orchestrator:
- Selects and loads LLM models from any supported provider
- Dispatches to the correct workflow graph
- Runs async execution via LangGraph's `astream`
- Handles state serialization and logging

### Workflows (graphs/)

Each file defines a LangGraph `StateGraph`. The 9 workflows are:

| Workflow | File | Purpose |
|---|---|---|
| `single_agent` | `single_agent.py` | Default. One LLM with chemistry tools |
| `multi_agent` | `multi_agent.py` | Planner/Executor/Aggregator pipeline |
| `python_relp` | `python_relp_agent.py` | Interactive Python REPL |
| `graspa` | `graspa_agent.py` | Gas adsorption in MOFs |
| `mock_agent` | `mock_agent.py` | Testing workflow |
| `single_agent_mcp` | `single_agent_mcp.py` | Single agent via MCP tools |
| `graspa_mcp` | `graspa_mcp.py` | gRASPA via MCP + Parsl |
| `mof_builder_mcp` | `mof_builder_mcp.py` | MOF construction via MCP |

### Tools (tools/)

LangChain `@tool`-decorated functions. Key files:

- `ase_tools.py` -- `run_ase` (energy/opt/vib/thermo), `save_atomsdata_to_file`, `file_to_atomsdata`
- `cheminformatics_tools.py` -- `molecule_name_to_smiles`, `smiles_to_coordinate_file`, `smiles_to_atomsdata`
- `generic_tools.py` -- `calculator` (safe math eval), Python REPL
- `report_tools.py` -- `generate_html` (interactive HTML reports with NGL 3D viewer)
- `graspa_tools.py` -- gRASPA simulation tools
- `architector_tools.py` -- Metal complex tools
- `pormake_tools.py` -- MOF topology/structure tools
- `parsl_tools.py` -- MACE with Parsl for HPC parallel execution

### MCP servers (mcp/)

FastMCP-based servers. Each exposes chemistry tools over stdio or HTTP:

- `mcp_tools.py` -- General chemistry MCP server (name-to-SMILES, structure gen, ASE simulations, file I/O). Port 9003.
- `mace_mcp_parsl.py` -- MACE ML potential with Parsl HPC. Port 9004.
- `graspa_mcp_parsl.py` -- gRASPA simulation with Parsl HPC. Port 9005.
- `data_analysis_mcp.py` -- Data analysis (CIF splitting, JSONL aggregation, isotherm plotting). Port 9006.
- `server_utils.py` -- Shared startup utility (`run_mcp_server`), handles stdio vs streamable_http transport, logging to stderr.

### Schemas (schemas/)

Pydantic models for data validation:

- `atomsdata.py` -- `AtomsData` (numbers, positions, cell, pbc)
- `ase_input.py` -- `ASEInputSchema` / `ASEOutputSchema`
- `agent_response.py` -- `ResponseFormatter`, `VibrationalFrequency`, `IRSpectrum`, etc.
- `calculators/` -- One schema per calculator: `mace_calc.py`, `emt_calc.py`, `tblite_calc.py`, `nwchem_calc.py`, `orca_calc.py`, `psi4_calc.py`, `fairchem_calc.py`, `mopac_calc.py`, `aimnet2_calc.py`

### Memory (memory/)

SQLite-backed session persistence:
- `store.py` -- `SessionStore` class: CRUD for sessions, context building for resume, prefix-based session ID lookup. Database at `~/.chemgraph/sessions.db`.
- `schemas.py` -- `SessionMessage` (role, content, tool_name, timestamp), `Session` (full record with messages), `SessionSummary` (lightweight listing model)

### State (state/)

LangGraph state definitions:
- `state.py` -- `State` (messages + remaining_steps), `MultiAgentState`
- `multi_agent_state.py` -- `ManagerWorkerState` for Planner/Executor/Aggregator
- `graspa_state.py`, `mof_state.py` -- Domain-specific states

## How to add a new LangChain tool

1. Create or edit a file in `src/chemgraph/tools/`
2. Define the function with the `@tool` decorator from `langchain_core.tools`
3. Use Pydantic schemas for structured input (see `schemas/ase_input.py` for the pattern)
4. Import the tool in the relevant graph file (`src/chemgraph/graphs/`) and add it to the tools list
5. Add tests in `tests/`

Example pattern from `cheminformatics_tools.py`:

```python
from langchain_core.tools import tool

@tool
def molecule_name_to_smiles(name: str) -> str:
    """Convert a molecule name to SMILES using PubChem."""
    import pubchempy as pcp
    comps = pcp.get_compounds(name.strip(), "name")
    if not comps:
        raise ValueError(f"No PubChem compound found for: {name}")
    return comps[0].canonical_smiles
```

## How to add a new MCP server tool

MCP tools are defined in `src/chemgraph/mcp/` using FastMCP:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="My Server", instructions="...")

@mcp.tool(name="my_tool", description="What it does")
async def my_tool(param: str) -> dict:
    # implementation
    return {"status": "success", "result": ...}

if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server
    run_mcp_server(mcp, default_port=9007)
```

The `run_mcp_server` utility handles:
- `--transport stdio` (default) for LangGraph/OpenCode MCP clients
- `--transport streamable_http` with `--port` and `--host` for HTTP access
- Logging to stderr (critical for stdio mode) and optional file logging via `CHEMGRAPH_LOG_DIR`

## How to add a new calculator

1. Create a Pydantic schema in `src/chemgraph/schemas/calculators/` (follow `mace_calc.py` pattern)
2. The schema must define `calculator_type`, implement `get_calculator()` returning an ASE calculator, and optionally `get_atoms_properties()`
3. Register it in `src/chemgraph/tools/mcp_helper.py` `load_calculator()` with a new `elif` branch
4. Add tests

## How to add a new workflow

1. Create a new graph file in `src/chemgraph/graphs/`
2. Define a `StateGraph` with nodes, edges, and conditional routing
3. Register it in `src/chemgraph/agent/llm_agent.py` in the workflow dispatch logic
4. Add a prompt template in `src/chemgraph/prompt/` if needed
5. Add a state class in `src/chemgraph/state/` if the workflow needs custom state

## Running tests

```bash
# Run all tests (excluding LLM-dependent tests)
pytest tests/

# Run with LLM tests
pytest tests/ --run-llm

# Run specific test file
pytest tests/test_mcp.py

# Run async tests
pytest tests/test_mcp.py -v
```

Test markers:
- `@pytest.mark.llm` -- requires LLM API access (skipped by default)
- `@pytest.mark.asyncio` -- async tests

## Running MCP servers

```bash
# stdio mode (for LangGraph / OpenCode / Claude Desktop)
python -m chemgraph.mcp.mcp_tools

# HTTP mode
python -m chemgraph.mcp.mcp_tools --transport streamable_http --port 9003

# With log directory
CHEMGRAPH_LOG_DIR=/tmp/chemgraph_logs python -m chemgraph.mcp.mcp_tools
```

## Running the CLI

```bash
# Single query
chemgraph --query "Calculate the energy of water using MACE"

# Interactive mode
chemgraph --interactive

# List supported models
chemgraph --list-models

# Session management
chemgraph --list-sessions
chemgraph --show-session a3b2
chemgraph --delete-session a3b2c1d4
chemgraph -q "Follow-up query" --resume a3b2
```

## Running the Streamlit UI

```bash
streamlit run src/ui/app.py
```

## Configuration

`config.toml` at the project root controls runtime settings:
- `[general]` -- model, workflow, recursion_limit, verbosity
- `[chemistry.calculators]` -- default calculator (mace_mp), fallback (emt)
- `[chemistry.optimization]` -- optimizer method, fmax, steps
- `[api.*]` -- LLM provider base URLs and timeouts

## Coding conventions

- Python >= 3.10, formatted with Ruff (line-length 88)
- Pydantic for all data models and tool input schemas
- Async-first for MCP tool implementations
- All MCP server logging must go to stderr (stdout is reserved for stdio transport)
- Energies in eV, frequencies in cm^-1, distances in Angstroms
- Pre-commit hooks configured via `.pre-commit-config.yaml` (Ruff linter + formatter)

## Docker

```bash
# Streamlit UI
docker compose --profile streamlit up

# MCP server
docker compose --profile mcp up

# Jupyter notebooks
docker compose --profile jupyter up
```

## Key dependencies

- `langgraph` + `langchain` -- agent orchestration
- `ase` -- atomic simulation environment
- `rdkit` -- cheminformatics, 3D structure generation
- `pubchempy` -- PubChem molecule lookup
- `mcp` + `fastmcp` -- Model Context Protocol servers
- `mace-torch` -- MACE ML potentials
- `pydantic` -- data validation
- `parsl` -- HPC parallel execution (optional)
- `streamlit` + `stmol` -- web UI (optional)

## When to use this skill

Use this skill when:
- Developing new features, tools, or workflows for ChemGraph
- Adding or modifying MCP servers
- Adding new calculator integrations
- Writing or debugging tests
- Understanding the project architecture
- Refactoring or extending existing code
