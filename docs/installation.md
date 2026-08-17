# Installation

ChemGraph requires Python 3.11 or newer. Use a virtual environment because its
scientific dependencies may conflict with packages in other projects.

## Install from PyPI

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install chemgraph
```

Confirm that the command-line entry point and model registry load:

```bash
chemgraph --help
chemgraph models
```

The core installation includes the agent framework, ASE, RDKit, MACE, EMT, the
Streamlit dependencies, and the general MCP server. MACE downloads model weights
when first used; start with EMT if you need an offline calculator smoke test.

## Install from source

Use a source checkout for the Streamlit interface, examples, documentation, or
development:

```bash
git clone https://github.com/argonne-lcf/ChemGraph.git
cd ChemGraph
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

With [uv](https://docs.astral.sh/uv/), the equivalent development setup is:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Optional extras

Install only the integrations needed by your workflow:

| Extra | Install command | Adds |
| --- | --- | --- |
| `calculators` | `pip install "chemgraph[calculators]"` | TBLite |
| `uma` | `pip install "chemgraph[uma]"` | UMA through fairchem-core |
| `ui` | `pip install "chemgraph[ui]"` | Additional UI dependencies |
| `rag` | `pip install "chemgraph[rag]"` | Document ingestion and vector stores |
| `xanes` | `pip install "chemgraph[xanes]"` | XANES workflow dependencies |
| `docking` | `pip install "chemgraph[docking]"` | Meeko docking preparation |
| `parsl` | `pip install "chemgraph[parsl]"` | Parsl execution |
| `ensemble_launcher` | `pip install "chemgraph[ensemble_launcher]"` | ALCF ensemble launcher |
| `globus_compute` | `pip install "chemgraph[globus_compute]"` | Globus Compute execution |
| `academy` | `pip install "chemgraph[academy]"` | Academy multi-agent runtime |
| `codex` | `pip install "chemgraph[codex]"` | Experimental Codex subscription route |

Extras can be combined:

```bash
python -m pip install "chemgraph[academy,parsl,globus_compute]"
```

!!! note "UMA environment"
    UMA's `e3nn` requirements can conflict with the MACE stack in the core
    environment. A separate virtual environment is the safest setup for UMA.

## External programs

Some integrations are Python adapters, not bundled simulation programs:

- ORCA and NWChem must be installed and configured for ASE separately.
- FDMNES is required for local XANES execution; set `FDMNES_EXE`.
- AutoDock Vina is normally installed from conda-forge for docking.
- Site-specific gRASPA and HPC modules require executables, schedulers, and
  filesystem paths available at the target facility.

## Conda environments

If an integration needs compiled or external dependencies, create a clean
environment first and install ChemGraph with pip inside it:

```bash
conda create -n chemgraph python=3.11
conda activate chemgraph
python -m pip install chemgraph
```

## Docker

Docker avoids a local Python installation and provides CLI, Streamlit, MCP, and
Jupyter-oriented images. See [Docker](docker_support.md) for commands, ports,
credential forwarding, and artifact volumes.

## Next step

Continue with the [quickstart](quickstart.md), then configure a provider in
[Models and authentication](models.md).
