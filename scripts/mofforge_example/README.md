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
variables. The four-server demo in this directory forwards `MOFFORGE_*` and
other runtime variables explicitly to its stdio subprocesses. ChemGraph's
single-server CLI loader has a narrower environment whitelist, so do not assume
these variables are forwarded by every MCP client:

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

Proves the full edge chain: load mofforge MCP tools → build a MOF → MACE energy
via ChemGraph's `run_ase` → validate.

```bash
export MOFFORGE_LOG_DIR=/tmp/mofforge_out
export CHEMGRAPH_LOG_DIR=/tmp/mofforge_out
python scripts/mofforge_example/verify_local_integration.py
```

Expected tail:

```
[1] Loaded 23 mofforge MCP tools: [...]
[2] Built 200-atom MOF (dia/pormake) -> .../dia_N109_E41.cif
[3] MACE single-point energy: -1359.8566 eV
[4] Validation ran (is_valid=False)
OK: mofforge -> ChemGraph local integration verified.
```

(`is_valid=False` is expected for a raw, unrelaxed pormake placement.)

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

## 3. Async single agent with all MOF workflow tools

`demo_single_agent_all_mcp.py` exposes four persistent MCP servers to one
ChemGraph `single_agent` workflow:

1. mofforge for MOF search, construction, modification, and validation;
2. FairChem/UMA for relaxation and energy calculations;
3. PACMOF2 for partial-charge assignment; and
4. gRASPA for adsorption simulations.

First verify server startup and inspect the complete tool inventory without
using an LLM:

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

Pass `--query` to run a custom workflow. The MCP sessions remain open for the
whole agent turn so submitted batch IDs can be polled from the same server
process:

```bash
python scripts/mofforge_example/demo_single_agent_all_mcp.py \
  --model argo:gpt-4o \
  --query "Validate /data/mofs/example.cif, relax it with FairChem, assign PACMOF2 charges, then run a 298 K water adsorption simulation with gRASPA."
```

### Isolated worker environments

Tool discovery only requires each interpreter to import its MCP server.
Actually invoking a tool requires the corresponding engine and worker
dependencies. Configure separate interpreters when the packages conflict:

```bash
export MOFFORGE_PYTHON=/path/to/mofforge-env/bin/python
export FAIRCHEM_PYTHON=/path/to/fairchem-env/bin/python
export PACMOF2_PYTHON=/path/to/pacmof2-env/bin/python
export GRASPA_PYTHON=/path/to/graspa-env/bin/python

python scripts/mofforge_example/demo_single_agent_all_mcp.py \
  --list-tools-only
```

The same values can be supplied with `--mofforge-python`,
`--fairchem-python`, `--pacmof2-python`, and `--graspa-python`. Select an
execution layer with `--backend`; the default is `local`, while `parsl`,
`ensemble_launcher`, and `globus_compute` use the existing ChemGraph backend
configuration.

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
