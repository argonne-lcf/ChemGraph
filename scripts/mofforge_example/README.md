# mofforge + ChemGraph (MOF tools)

[mofforge](https://github.com/tdpham2/mofforge) is a MOF build / modify / screen
toolkit built on pymatgen. It ships a ChemGraph integration out of the box: it
exposes its MOF capabilities as **MCP tools** via two entry points, so a
ChemGraph agent can build or screen MOFs and then hand the resulting CIFs to
ChemGraph's own ASE / gRASPA simulation tools.

| Entry point | Server | Execution |
|---|---|---|
| `mofforge-mcp` | stock FastMCP (no ChemGraph dependency) | all tools **inline** |
| `mofforge-mcp-chemgraph` | `CGFastMCP` (ChemGraph HPC) | heavy tools (`build`, `render`, `screen_and_place` fan-out) submitted to an execution backend with job tracking |

This directory covers the **local (edge-side)** integration using the stock
`mofforge-mcp` server. The HPC path (`mofforge-mcp-chemgraph` + Parsl on Aurora)
is a later milestone.

## Install

```bash
pip install -e .                                          # ChemGraph (repo root)
pip install -e "/path/to/mofforge[mcp,chem,build]"        # mofforge + pormake
```

`build` pulls the pormake backend, which is enough for the examples here.
TOBACCO is an optional external clone configured via `MOFFORGE_TOBACCO_PATH`.

## Data paths

mofforge's database / structure tools read their data from environment
variables. The four-server launcher in this directory forwards `MOFFORGE_*` and
other runtime variables to the server processes. Set them in the terminal that
runs `start_mcp_servers.py`, not only in the agent terminal:

| Variable | Purpose |
|---|---|
| `MOFFORGE_LOG_DIR` | base dir for relative output paths (CIFs, PNGs) |
| `MOFFORGE_COREMOF_DATA_PATH` | CoRE MOF metadata CSV |
| `MOFFORGE_COREMOF_STRUCTURES_PATH` | dir of CoRE MOF CIF files |
| `MOFFORGE_CSD_DATA_PATH` | CSD MOF subset export (TSV) |
| `MOFFORGE_TOBACCO_PATH` | TOBACCO 3.0 clone (optional build backend) |

The `build`-only examples here need just `MOFFORGE_LOG_DIR`. Database
search/screen tools need the CoRE / CSD paths (data is a separate download; see
mofforge `docs/chemgraph.md`).

## 1. Deterministic verification (no LLM)

Proves the full edge chain: load mofforge MCP tools → build a MOF → UMA energy
through the FairChem `run_fairchem_single` MCP tool → validate.

```bash
export MOFFORGE_LOG_DIR=/tmp/mofforge_out
export CHEMGRAPH_LOG_DIR=/tmp/mofforge_out
export FAIRCHEM_PYTHON=/path/to/fairchem-env/bin/python
export HF_TOKEN=...
python scripts/mofforge_example/verify_local_integration.py --device cpu
```

Expected tail:

```
[1] Loaded 23 mofforge MCP tools: [...]
[2] Built 200-atom MOF (dia/pormake) -> .../dia_N109_E41.cif
[3] UMA (uma-s-1p1/odac) single-point energy: ... eV
[4] Validation ran (is_valid=False)
OK: mofforge -> ChemGraph local integration verified.
```

The verifier uses the UMA `odac` task because its input is a periodic MOF.
`is_valid=False` is expected for a raw, unrelaxed pormake placement. The
actual energy varies with the installed FairChem/UMA release.

## 2. Agent-driven run (LLM)

Runs the same tools through a ChemGraph agent over the stdio MCP transport. The
agent decides which mofforge tools to call.

```bash
export MOFFORGE_LOG_DIR=/tmp/mofforge_out
export CHEMGRAPH_LOG_DIR=/tmp/mofforge_out
BBS=$(python -c "import pormake,os;print(os.path.join(os.path.dirname(pormake.__file__),'database','bbs'))")

chemgraph run \
  -m "argo:gpt-4o" \
  -w single_agent_mcp \
  --mcp-command "mofforge-mcp --transport stdio" \
  -o last_message \
  -q "Build a diamond (dia) topology MOF with the pormake backend using node file $BBS/N109.xyz and edge file $BBS/E41.xyz, then validate the resulting structure. Report the output CIF path and whether it is valid."
```

The agent calls `mofforge_build` then `mofforge_validate` and summarizes the
result. Swap `-m` for any model your environment has credentials for.

## 3. Async single agent with standalone HTTP MCP servers

This demo deliberately separates MCP server lifecycle from the ChemGraph
agent:

1. `start_mcp_servers.py` starts and monitors four streamable-HTTP servers.
2. `demo_single_agent_all_mcp.py` connects to those endpoints and exposes their
   tools to one ChemGraph `single_agent` workflow.

The four servers provide:

1. mofforge for MOF search, construction, modification, and validation;
2. FairChem/UMA for relaxation and energy calculations;
3. PACMOF2 for partial-charge assignment; and
4. gRASPA for adsorption simulations.

The script does not define another agent, graph, or system prompt. After
loading the MCP tools it delegates directly to ChemGraph:

```python
agent = ChemGraph(
    model_name=model,
    workflow_type="single_agent",
    tools=tools,
)
result = await agent.run(query)
```

### Terminal 1: start the MCP servers

Set server-side data paths, execution configuration, and any runtime
credentials before launching:

```bash
export MOFFORGE_LOG_DIR=/tmp/mofforge_out
export CHEMGRAPH_LOG_DIR=/tmp/mofforge_out

python scripts/mofforge_example/start_mcp_servers.py \
  --backend local
```

The launcher waits for all four servers, prints their URLs, and remains in the
foreground so their job trackers stay alive. Press Ctrl-C to stop every server
together. If one server fails, the launcher stops the others and exits with an
error.

### Terminal 2: connect ChemGraph

First inspect the complete tool inventory without using an LLM:

```bash
python scripts/mofforge_example/demo_single_agent_all_mcp.py \
  --list-tools-only
```

Then run the default lightweight query, which calls mofforge discovery tools
and explains the downstream workflow without launching a simulation:

```bash
python scripts/mofforge_example/demo_single_agent_all_mcp.py \
  --model argo:gpt-4o
```

Pass `--query` to run a custom workflow. The independently running HTTP
servers keep submitted batch IDs available across client sessions:

```bash
python scripts/mofforge_example/demo_single_agent_all_mcp.py \
  --model argo:gpt-4o \
  --query "Validate /data/mofs/example.cif, relax it with FairChem, assign PACMOF2 charges, then run a 298 K water adsorption simulation with gRASPA."
```

### Isolated worker environments

Tool discovery only requires each interpreter to import its MCP server.
Actually invoking a tool requires the corresponding engine and worker
dependencies. Configure separate interpreters in the launcher terminal when
the packages conflict:

```bash
export MOFFORGE_PYTHON=/path/to/mofforge-env/bin/python
export FAIRCHEM_PYTHON=/path/to/fairchem-env/bin/python
export PACMOF2_PYTHON=/path/to/pacmof2-env/bin/python
export GRASPA_PYTHON=/path/to/graspa-env/bin/python

python scripts/mofforge_example/start_mcp_servers.py \
  --backend local
```

The same values can be supplied with `--mofforge-python`,
`--fairchem-python`, `--pacmof2-python`, and `--graspa-python`. Select an
execution layer on the launcher with `--backend`; the default is `local`, while
`parsl`, `ensemble_launcher`, and `globus_compute` use the existing ChemGraph
backend configuration.

### Custom or remote endpoints

The agent defaults to the four loopback URLs printed by the launcher. Override
individual endpoints for alternate ports, remote hosts, or SSH tunnels:

```bash
python scripts/mofforge_example/demo_single_agent_all_mcp.py \
  --mofforge-url http://127.0.0.1:19010/mcp/ \
  --fairchem-url http://127.0.0.1:19008/mcp/ \
  --pacmof2-url http://127.0.0.1:19009/mcp/ \
  --graspa-url http://127.0.0.1:19001/mcp/ \
  --list-tools-only
```

Use the matching `--mofforge-port`, `--fairchem-port`, `--pacmof2-port`, and
`--graspa-port` options when changing ports on the launcher.

The tools exchange structure and result paths, not file contents. All four
servers must therefore see those paths through the same local or shared
filesystem. If loopback requests are intercepted by a configured HTTP proxy,
add `127.0.0.1,localhost` to both `NO_PROXY` and `no_proxy`.

All prompting, reasoning, LangGraph routing, tool invocation, and final
response generation come from ChemGraph's standard `single_agent` workflow.
LLM credentials belong in the agent terminal; the launcher intentionally does
not forward them to MCP server processes.

The demo leaves the already-prefixed `mofforge_*` names unchanged. FairChem,
PACMOF2, and gRASPA tools receive server prefixes so their common
`check_job_status`, `get_job_results`, and other job-management tools cannot be
confused.

> [!NOTE]
> Successful tool discovery does not prove that all simulation runtimes are
> installed. FairChem requires the `uma` environment, PACMOF2 is installed
> separately from source, and the current gRASPA core uses an ALCF
> site-specific executable path.

The standalone mofforge MCP server is used here because it exposes the complete
MOF tool surface without depending on ChemGraph. The optional
`mofforge-mcp-chemgraph` entry point remains appropriate when mofforge build,
render, or screening fan-out should run through `CGFastMCP`.

## Building-block notes

pormake ships its building blocks as plain `N*.xyz` / `E*.xyz` files under
`pormake/database/bbs/`. Pass **file paths** (as above) rather than bare DB
names — mofforge currently appends a hash suffix to bare names
(`N109` → `N109_<hash>.xyz`) that does not resolve against the shipped files.

Topology connectivity must match the node: `dia` is 4-connected (matches
`N109`); `pcu` is 6-connected and needs a 6-connection node.
