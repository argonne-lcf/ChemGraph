<p align="left">
  <img src="logo/chemgraph-color-dark__rgb-hires.jpg" alt="ChemGraph logo" width="240">
</p>

[![Tests](https://github.com/argonne-lcf/ChemGraph/actions/workflows/tests.yml/badge.svg)](https://github.com/argonne-lcf/ChemGraph/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/chemgraph.svg)](https://pypi.org/project/chemgraph/)
[![Python](https://img.shields.io/pypi/pyversions/chemgraph.svg)](https://pypi.org/project/chemgraph/)
[![Documentation](https://img.shields.io/badge/docs-MkDocs-4051b5)](https://argonne-lcf.github.io/ChemGraph/)
[![Docker](https://img.shields.io/badge/Docker-GHCR-2496ED?logo=docker&logoColor=white)](https://github.com/argonne-lcf/ChemGraph/pkgs/container/chemgraph)
[![License](https://img.shields.io/github/license/argonne-lcf/ChemGraph)](LICENSE)

# ChemGraph

ChemGraph is an agent framework for computational chemistry and materials
science. It connects natural-language requests to molecular construction,
simulation, analysis, and reporting tools built with LangGraph, ASE, RDKit,
and the Model Context Protocol (MCP).

Use ChemGraph from the command line, Python, a Streamlit web interface, or as
an MCP server. Local workflows can use ASE calculators such as EMT and MACE;
optional integrations add TBLite, UMA, docking, XANES, retrieval-augmented
generation, and distributed execution on systems such as ALCF Polaris and
Aurora.

> ChemGraph can launch calculations and write files. Review generated inputs,
> calculator settings, convergence, units, and scientific conclusions before
> relying on a result.

## Quickstart

ChemGraph requires Python 3.11 or newer. A virtual environment keeps its
scientific dependencies separate from other projects.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install chemgraph
```

Prefer [uv](https://docs.astral.sh/uv/)? Use it to create the same environment
and install ChemGraph:

```bash
uv venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
uv pip install chemgraph
```

Choose one model provider and set only the credential it needs. The default
model is `gpt-4o-mini`.

| Provider | Setup | Example model |
| --- | --- | --- |
| OpenAI | `export OPENAI_API_KEY="..."` | `gpt-4o-mini` |
| Anthropic | `export ANTHROPIC_API_KEY="..."` | `claude-3-5-haiku-20241022` |
| Google | `export GEMINI_API_KEY="..."` | `gemini-2.5-flash` |
| Groq | `export GROQ_API_KEY="..."` | `groq:<model-id>` |
| Argo (Argonne) | `export ARGO_USER="<anl-username>"` | `argo:gpt-4o` |
| ALCF inference endpoints | `export ALCF_ACCESS_TOKEN="..."` | Use `chemgraph models` |
| Ollama | Start Ollama locally; no API key | `llama3.2` |

See [Models and authentication](https://argonne-lcf.github.io/ChemGraph/models/)
for endpoint setup, ALCF token instructions, supported model identifiers, and
the experimental Codex subscription route.

Check the installation and run a small tool-using query:

```bash
pip install chemgraph[calculators]
```

> Note: On some platforms/Python combinations (especially where no prebuilt `tblite`
> wheel is available), installing the `calculators` extra may require a local
> Fortran toolchain.

**Install from Source (Alternative Methods)**

If you need to install from source for the latest version:

**Using pip from source**

1. Clone the repository:
   ```bash
   git clone https://github.com/argonne-lcf/ChemGraph
   cd ChemGraph
    ```
2. Create and activate a virtual environment:
   ```bash
   # Using venv (built into Python)
   python -m venv chemgraph-env
   source chemgraph-env/bin/activate  # On Unix/macOS
   # OR
   .\chemgraph-env\Scripts\activate  # On Windows
   ```

3. Install ChemGraph:
   ```bash
   pip install -e .
   ```

**Using Conda from source**

> ⚠️ **Note on Compatibility**  
> ChemGraph supports both MACE and UMA (Meta's machine learning potential). However, due to the current dependency conflicts, particularly with `e3nn`—**you cannot install both in the same environment**.  
> To use both libraries, create **separate Conda environments**, one for each.

1. Clone the repository:
   ```bash
   git clone --depth 1 https://github.com/argonne-lcf/ChemGraph
   cd ChemGraph
   ```

2. Create and activate the conda environment from the provided environment.yml:
   ```bash
   conda env create -f environment.yml
   conda activate chemgraph
   ```

   The `environment.yml` file automatically installs all required dependencies including:
   - Python 3.11
   - Core packages (numpy, pandas, pytest, rich, toml)
   - Computational chemistry packages (nwchem, tblite)
   - All ChemGraph dependencies via pip
   

**Using uv from source**

1. Clone the repository:
   ```bash
   git clone https://github.com/argonne-lcf/ChemGraph
   cd ChemGraph
   ```

2. Create and activate a virtual environment using uv:
    ```bash
    uv venv --python 3.11 chemgraph-env

    source chemgraph-env/bin/activate # Unix/macos
    # OR
    .\chemgraph-env\Scripts\activate  # On Windows
   ```

3. Install ChemGraph using uv:
    ```bash
    uv pip install -e .
    ```

**Optional: Install with UMA support**

> ⚠️ **Note on e3nn Conflict for UMA Installation:** The `uma` extras (requiring `e3nn>=0.5`) conflict with the base `mace-torch` dependency (which pins `e3nn==0.4.4`). 
> 
> **For PyPI installations**, you can try:
> ```bash
> pip install chemgraph[uma]
> ```
> However, this may fail due to the e3nn version conflict. If it does, you'll need to install from source using the workaround below.
>
> **For source installations**, if you need to install UMA support in an environment where `mace-torch` might cause this conflict, you can try the following workaround:
> 1. **Temporarily modify `pyproject.toml`**: Open the `pyproject.toml` file in the root of the ChemGraph project.
> 2. Find the line containing `"mace-torch",` in the `dependencies` list.
> 3. Comment out this line by adding a `#` at the beginning (e.g., `#    "mace-torch",`).
> 4. **Install UMA extras**: Run `pip install -e ".[uma]"`.
> 5. **(Optional) Restore `pyproject.toml`**: After installation, you can uncomment the `mace-torch` line if you still need it for other purposes in the same environment. Be aware that `mace-torch` might not function correctly due to the `e3nn` version mismatch (`e3nn>=0.5` will be present for UMA).
>
> **The most robust solution for using both MACE and UMA with their correct dependencies is to create separate Conda environments, as highlighted in the "Note on Compatibility" above.**

> **Important for UMA Model Access:** The `facebook/UMA` model is a gated model on Hugging Face. To use it, you must:
> 1. Visit the [facebook/UMA model page](https://huggingface.co/facebook/UMA) on Hugging Face.
> 2. Log in with your Hugging Face account.
> 3. Accept the model's terms and conditions if prompted.
> Your environment (local or CI) must also be authenticated with Hugging Face, typically by logging in via `huggingface-cli login` or ensuring `HF_TOKEN` is set and recognized.
</details>

<details open>
  <summary><strong>Example Usage</strong></summary>

1. Before exploring example usage in the `notebooks/` directory, ensure you have specified the necessary API tokens in your environment. For example, you can set the OpenAI API token and Anthropic API token using the following commands:

   ```bash
   # Set OpenAI API token
   export OPENAI_API_KEY="your_openai_api_key_here"

   # Set Anthropic API token
   export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
   
   # Set Google API token
   export GEMINI_API_KEY="your_google_api_key_here"
   ```

2. **Explore Example Notebooks**: Navigate to the `notebooks/` directory to explore various example notebooks demonstrating different capabilities of ChemGraph.

   - **[Single-Agent System with MACE](notebooks/1_Demo_single_agent.ipynb)**: This notebook demonstrates how a single agent can utilize multiple tools with MACE/xTB support.

   - **[Single-Agent System with UMA](notebooks/Demo_single_agent_UMA.ipynb)**: This notebook demonstrates how a single agent can utilize multiple tools with UMA support.

   - **[Multi-Agent System](notebooks/2_Demo-multi_agent.ipynb)**: This notebook demonstrates a multi-agent setup where planner and executor agents decompose and run computational chemistry tasks.

   - **[Model Context Protocol (MCP) Server](notebooks/3_Demo_using_MCP.ipynb)**: This notebook demonstrates how to run an MCP server and connect to ChemGraph.

   - **[Infrared absorption spectrum prediction](notebooks/Demo_infrared_spectrum.ipynb)**: This notebook demonstrates how to calculate an infrared absorption spectrum.


</details>

<details>
  <summary><strong>Streamlit Web Interface</strong></summary>

ChemGraph includes a **Streamlit web interface** for chat-driven computational chemistry workflows. The UI auto-initializes the selected agent, streams tool-call progress while a query runs, shows generated structures and reports, and stores conversations in the same local session database used by the CLI.

### Features

- **🧪 Interactive Chat Interface**: Natural language queries for computational chemistry tasks
- **🧬 3D Molecular Visualization**: Interactive molecular structure display using `stmol` and `py3Dmol`
- **📊 Report Integration**: Embedded and downloadable HTML reports from computational calculations
- **💾 Data Export**: Download molecular structures as XYZ or JSON files
- **🧮 Math Rendering**: Display LaTeX-style equations and reaction arrows in assistant responses
- **🔧 Multiple Workflows**: Support for single-agent, multi-agent, Python REPL, and gRASPA workflows
- **💬 Session Memory**: Browse, load, and delete saved conversations from `~/.chemgraph/sessions.db`
- **👤 Human Supervision**: Optional follow-up prompts when the agent needs confirmation or missing inputs

### Installation Requirements

The Streamlit UI dependencies are included by default when you install ChemGraph:

```bash
# Install ChemGraph (includes UI dependencies)
pip install -e .
```

**Alternative Installation Options:**
```bash
# Install only UI dependencies separately (if needed)
pip install -e ".[ui]"

# Install with UMA support (separate environment recommended)
pip install -e ".[uma]"
```

### Running the Streamlit Interface

1. **Set up your API keys** (same as for notebooks):
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
   ```

2. **Launch the Streamlit app**:
   ```bash
   streamlit run src/ui/app.py
   ```

3. **Access the interface**: Open your browser to `http://localhost:8501`

### Using the Interface

#### Configuration
- Use the **Configuration** page to edit `config.toml`, provider base URLs, API timeouts, workflow, recursion limit, report generation, and human supervision.
- API keys entered in the UI are applied only to the current Streamlit process and are not written to `config.toml`.
- The main sidebar shows calculators detected during ChemGraph initialization and marks the default calculator used when a query does not specify one.
- To change model, workflow, thread, or report settings, edit them on the **Configuration** page, save, then use **Reload Config** or **Refresh Agents**.


#### Interaction
1. **Open the main page**: The agent initializes automatically from the active configuration.
2. **Ask Questions**: Use the chat input to enter computational chemistry queries.
3. **Monitor Tools**: Tool calls and completions stream in the assistant response while the workflow runs.
4. **Respond to Prompts**: If human supervision is enabled and the agent pauses, answer in the same chat input.
5. **View and Export Results**: Structures, IR artifacts, HTML reports, and download controls appear with the response when available.

#### Example Queries
- "What is the SMILES string for caffeine?"
- "Optimize the geometry of water molecule using DFT"
- "Calculate the single point energy of methane and show the structure"
- "Generate the structure of aspirin and calculate its vibrational frequencies"

#### Molecular Visualization
The interface automatically detects molecular structure data in agent responses and provides:
- **Interactive 3D Models**: Multiple visualization styles (ball & stick, sphere, stick, wireframe)
- **Structure Information**: Chemical formula, composition, mass, center of mass
- **Export Options**: Download as XYZ files or JSON data
- **Fallback Display**: Table view when 3D visualization is unavailable

#### Conversation Management
- **History Display**: All queries and responses are preserved in conversation bubbles
- **Saved Sessions**: Recent sessions can be loaded or deleted from the sidebar
- **Structure Detection**: Molecular structures are automatically extracted and visualized
- **Report Integration**: HTML reports and run artifacts are embedded directly in the interface
- **Debug Information**: Expandable sections show detailed message processing information

### Troubleshooting

**3D Visualization Issues:**
- Ensure `stmol` is installed: `pip install stmol`
- If 3D display fails, the interface falls back to table/text display
- Check browser compatibility for WebGL support

**Agent Initialization:**
- Verify API keys are set correctly
- Verify provider base URLs and local model endpoints on the Configuration page
- Check that ChemGraph package is installed: `pip install -e .`
- Ensure all dependencies are available in your environment

**Performance:**
- For large molecular systems, visualization may take longer to load
- Start a new chat or load a smaller saved session if rendering many prior structures becomes slow
- Use **Refresh Agents** after changing credentials or external model services

</details>

<details>
  <summary><strong>Configuration with TOML</strong></summary>

ChemGraph supports comprehensive configuration through TOML files, allowing you to customize model settings, API configurations, chemistry parameters, and more.

### Configuration File Structure

Create a `config.toml` file in your project directory to configure ChemGraph behavior:

```toml
# ChemGraph Configuration File
# This file contains all configuration settings for ChemGraph CLI and agents

[general]
# Default model to use for queries
model = "gpt-4o-mini"
# Workflow type: single_agent, multi_agent, python_relp, graspa, molecular_docking, mock_agent
# Alias accepted by CLI/UI: python_repl -> python_relp
workflow = "single_agent"
# Output format: state, last_message
output = "state"
# Enable structured output
structured = false
# Generate detailed reports
report = true
# Default LangGraph thread ID
thread = 1

# Recursion limit for agent workflows
recursion_limit = 20
# Allow the agent to pause and ask for human input
human_supervised = false
# Enable verbose output
verbose = false

[llm]
# Temperature for LLM responses (0.0 to 1.0)
temperature = 0.1
# Maximum tokens for responses
max_tokens = 4000
# Top-p sampling parameter
top_p = 0.95
# Frequency penalty (-2.0 to 2.0)
frequency_penalty = 0.0
# Presence penalty (-2.0 to 2.0)
presence_penalty = 0.0

[api]
# Custom base URLs for different providers
[api.openai]
base_url = "https://api.openai.com/v1"
timeout = 30
argo_user = ""

[api.anthropic]
base_url = "https://api.anthropic.com"
timeout = 30

[api.google]
base_url = "https://generativelanguage.googleapis.com/v1beta"
timeout = 30

[api.alcf]
base_url = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
timeout = 30

[api.local]
# For local models like Ollama
base_url = "http://localhost:11434"
timeout = 60

[chemistry]
# Default calculation settings
[chemistry.optimization]
# Optimization method: BFGS, L-BFGS-B, CG, etc.
method = "BFGS"
# Force tolerance for convergence
fmax = 0.05
# Maximum optimization steps
steps = 200

[chemistry.frequencies]
# Displacement for finite difference
displacement = 0.01
# Number of processes for parallel calculation
nprocs = 1

[chemistry.calculators]
# Default calculator for different tasks
default = "mace_mp"
# Available calculators: mace_mp, emt, nwchem, orca, psi4, tblite
fallback = "emt"

[output]
# Output file settings
[output.files]
# Default output directory
directory = "./chemgraph_output"
# File naming pattern
pattern = "{timestamp}_{query_hash}"
# Supported formats: xyz, json, html, png
formats = ["xyz", "json", "html"]

[output.visualization]
# 3D visualization settings
enable_3d = true
# Molecular viewer: py3dmol, ase_gui
viewer = "py3dmol"
# Image resolution for saved figures
dpi = 300

[logging]
# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
level = "INFO"
# Log file location
file = "./chemgraph.log"
# Enable console logging
console = true

[features]
# Enable experimental features
enable_experimental = false
# Enable caching of results
enable_cache = true
# Cache directory
cache_dir = "./cache"
# Cache expiration time in hours
cache_expiry = 24

[security]
# Enable API key validation
validate_keys = true
# Enable request rate limiting
rate_limit = true
# Max requests per minute
max_requests_per_minute = 60
```

The core CLI and UI currently consume `[general]`, `[api]`, `[chemistry]`, and
`[output]` directly. The agent uses deterministic LLM defaults internally
(`temperature=0.0`, fixed token limits); `[llm]` entries are kept for
documentation/forward compatibility rather than active runtime tuning.

### Using Configuration Files

#### With the Command Line Interface

```bash
# Use configuration file
chemgraph --config config.toml -q "What is the SMILES string for water?"

# Override specific settings
chemgraph --config config.toml -q "Optimize methane" -m gpt-4o --verbose
```

#### Using Argo (Argonne Internal)

ChemGraph supports Argo through its OpenAI-compatible endpoint.

1. Set your Argo/OpenAI base URL in `config.toml`:

```toml
[api.openai]
base_url = "https://apps-dev.inside.anl.gov/argoapi/v1"
timeout = 30
argo_user = "<your_anl_domain_username>"
```

2. Set environment variables:

```bash
# Required by OpenAI-compatible clients in ChemGraph; for Argo use your ANL username
export OPENAI_API_KEY="<your_anl_domain_username>"

# Optional fallback only: used when api.openai.argo_user is not set in config.toml
export ARGO_USER="<your_anl_domain_username>"
```

3. Use an Argo model ID with the `argo:` prefix (from `supported_argo_models` in `src/chemgraph/models/supported_models.py`), for example:

```text
argo:gpt-4o, argo:gpt-4o-latest, argo:gpt-5, argo:gpt-5-mini,
argo:gemini-2.5-flash, argo:claude-sonnet-4.5
```

4. Run with config:

```bash
chemgraph --config config.toml -m argo:gpt-4o-latest -q "calculate the energy for water molecule using mace_mp"
```

Notes:
- Argo endpoints are available on Argonne internal network (or VPN on an Argonne-managed machine).
- For current Argo endpoint guidance and policy updates, refer to your internal Argo documentation.

#### Using ALCF Inference Endpoints

ChemGraph supports [ALCF Inference Endpoints](https://docs.alcf.anl.gov/services/inference-endpoints/), which provide API access to open-source models running on dedicated ALCF hardware (Sophia and Minerva clusters).

1. Configure the endpoint in `config.toml` (already set by default):

```toml
[api.alcf]
base_url = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
timeout = 30
```

`base_url` is the Sophia endpoint. Models hosted on Minerva
(`nemotron-3-ultra`, `inkling-bf16`) resolve to their own endpoint
automatically and ignore this setting; pass `--base-url` to override either.

2. Authenticate via Globus OAuth:

```bash
pip install globus_sdk
wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py
python inference_auth_token.py authenticate
```

3. Set the access token (valid for ~48 hours):

```bash
export ALCF_ACCESS_TOKEN=$(python inference_auth_token.py get_access_token)
```

4. Run with an ALCF model (use the model name directly, no prefix needed):

```bash
chemgraph --config config.toml -m meta-llama/Meta-Llama-3.1-70B-Instruct \
  -q "Calculate the energy of water using MACE"
```

The aspirin example uses an LLM and PubChem, so it requires network access.
For a first calculation, explicitly choose the lightweight EMT calculator:

```bash
chemgraph run \
  -q "Build water from SMILES O, optimize it with EMT, and report the final energy." \
  --output last_message
```

ChemGraph creates a session directory under `cg_logs/` by default. Tool output
such as XYZ structures, JSON results, trajectories, spectra, and HTML reports
is written there. Set `CHEMGRAPH_LOG_DIR` before starting ChemGraph to choose a
different artifact directory.

## Start here

- [Install ChemGraph](https://argonne-lcf.github.io/ChemGraph/installation/)
- [Follow the quickstart](https://argonne-lcf.github.io/ChemGraph/quickstart/)
- [Choose a model and authenticate](https://argonne-lcf.github.io/ChemGraph/models/)
- [Browse workflows](https://argonne-lcf.github.io/ChemGraph/workflows/)
- [Open the full documentation](https://argonne-lcf.github.io/ChemGraph/)

## Common ways to use ChemGraph

### Run one query

`single_agent` is the default workflow and the best starting point.

```bash
chemgraph run \
  --model gpt-4o-mini \
  --workflow single_agent \
  --query "Calculate the vibrational frequencies of water with EMT."
```

Useful run options include:

| Option | Purpose |
| --- | --- |
| `-m`, `--model` | Select the LLM provider/model identifier |
| `-w`, `--workflow` | Select an agent workflow |
| `-o`, `--output` | Return full `state` or only `last_message` |
| `-s`, `--structured` | Request structured final output |
| `-r`, `--report` | Allow generation of an HTML report |
| `--human-supervised` | Allow supported workflows to pause for input |
| `--output-file` | Save the CLI response to a file |
| `-v` / `-vv` | Enable INFO / DEBUG diagnostics |

The older form `chemgraph -q "..."` remains supported, but documentation uses
the explicit `chemgraph run` subcommand.

### Work interactively

```bash
chemgraph run --interactive
```

Inside the interactive shell, use `/help` to list commands. Sessions are saved
to `~/.chemgraph/sessions.db` and can also be inspected from the CLI:

```bash
chemgraph session list
chemgraph session show <session-id>
chemgraph run --resume <session-id> -q "Continue with a frequency calculation."
```

The `main_agent` workflow is a long-lived supervisor with durable checkpoints
and must be used interactively:

```bash
chemgraph run --interactive --workflow main_agent
chemgraph run --interactive --workflow main_agent --resume <session-id>
```

See the [CLI guide](https://argonne-lcf.github.io/ChemGraph/cli/) for session
semantics, interactive commands, MCP connections, tracing, and the
development-only workspace Deep Agent.

### Use the Python API

`ChemGraph.run()` is asynchronous. Import the class from its current public
module path:

```python
import asyncio

from chemgraph.agent.llm_agent import ChemGraph


async def main():
    agent = ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        return_option="last_message",
    )
    result = await agent.run("What is the SMILES string for aspirin?")
    print(result.content)


asyncio.run(main())
```

The checkpointed `main_agent` uses `MainAgentSession` rather than
`ChemGraph.run()`. See the [Python API guide](https://argonne-lcf.github.io/ChemGraph/python_api/)
for state returns, thread IDs, custom tools, and durable sessions.

### Use the Streamlit interface

The Streamlit entry point currently lives in the source tree. Run it from a
repository checkout:

```bash
git clone https://github.com/argonne-lcf/ChemGraph.git
cd ChemGraph
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
streamlit run src/ui/app.py
```

Open `http://localhost:8501`. For an image-based setup, use the Docker command
below. The [Streamlit guide](https://argonne-lcf.github.io/ChemGraph/streamlit_web_interface/)
describes configuration, supported workflows, sessions, and artifacts.

### Expose chemistry tools through MCP

Start the general tool server over stdio:

```bash
python -m chemgraph.mcp.mcp_tools
```

Or start streamable HTTP:

```bash
python -m chemgraph.mcp.mcp_tools \
  --transport streamable_http \
  --host 127.0.0.1 \
  --port 9003
```

MCP clients connect to `http://localhost:9003/mcp/`. ChemGraph can also load
MCP tools into an agent:

```bash
chemgraph run \
  --mcp-url http://localhost:9003/mcp/ \
  -q "Build a 3D structure for methane."
```

See [MCP servers](https://argonne-lcf.github.io/ChemGraph/mcp_servers/) for
stdio client configuration and the experimental HPC servers.

## Choose a workflow

| Workflow | Use it for | Important requirements |
| --- | --- | --- |
| `single_agent` | General molecule lookup, ASE calculations, and reports | Default and recommended first workflow |
| `main_agent` | Long-lived supervisor with delegated chemistry work | Interactive mode; use `MainAgentSession` in Python |
| `multi_agent` | Planner/executor decomposition and parallel subtasks | More model calls and orchestration overhead |
| `python_relp` | LLM-directed Python and arithmetic (`python_repl` is an alias) | Executes Python in the ChemGraph process; use only with trusted prompts |
| `molecular_docking` | Ligand/receptor docking with AutoDock Vina | `docking` extra plus Vina from conda-forge |
| `rag_agent` | Query PDF/text documents alongside chemistry tools | `rag` extra; embedding model or OpenAI embeddings |
| `single_agent_xanes` | XANES data retrieval, simulation, and plotting | `xanes` extra, `MP_API_KEY`, and/or `FDMNES_EXE` |
| `graspa` | gRASPA adsorption workflows | Site-specific gRASPA executable/runtime |
| `graspa_mcp` | Planner/executor workflow using supplied MCP tools | Advanced integration; MCP tools must be provided |
| `mock_agent` | One-pass tool-call experiments | Primarily useful for development and evaluation |

The [workflow guide](https://argonne-lcf.github.io/ChemGraph/workflows/) covers
capabilities, limitations, and interface support in more detail.

## Calculators and optional dependencies

The core installation includes ASE, EMT, and MACE. ChemGraph detects calculator
engines and external executables at startup, then exposes only the calculators
available in that environment.

| Capability | Installation | Notes |
| --- | --- | --- |
| EMT | Core install | Lightweight; useful for setup checks, not general high-accuracy chemistry |
| MACE | Core install | First use downloads model weights and can be slow |
| TBLite / xTB | `pip install "chemgraph[calculators]"` | May require a Fortran toolchain when no wheel is available |
| UMA / FAIRChem | `pip install "chemgraph[uma]"` | Use a separate environment from MACE if `e3nn` resolution conflicts |
| NWChem | Install the `nwchem` executable separately | Must be on `PATH` or configured through ASE |
| ORCA | Install ORCA separately | Must be on `PATH` or configured through ASE |
| AIMNet2 | Install `aimnet2calc` separately | Detected lazily when installed |

Other extras are available for `rag`, `docking`, `xanes`, `parsl`,
`ensemble_launcher`, `globus_compute`, `academy`, and experimental `codex`
support. See [Installation](https://argonne-lcf.github.io/ChemGraph/installation/)
and [Calculators](https://argonne-lcf.github.io/ChemGraph/calculators/).

## Install from source

Use a source checkout for development, notebooks, the Streamlit UI, and the
latest unreleased changes:

```bash
git clone https://github.com/argonne-lcf/ChemGraph.git
cd ChemGraph
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install only the extras required by the workflow you plan to run:

```bash
python -m pip install -e ".[rag]"
python -m pip install -e ".[academy,parsl,globus_compute]"
```

Conda and uv instructions are available in the
[installation guide](https://argonne-lcf.github.io/ChemGraph/installation/).

## Docker

Prefer containers? The
[published ChemGraph image](https://github.com/argonne-lcf/ChemGraph/pkgs/container/chemgraph)
supports the CLI, Streamlit, MCP, and JupyterLab. See
[Docker support](https://argonne-lcf.github.io/ChemGraph/docker_support/) for
commands, Compose profiles, credential forwarding, ports, and artifact mounts.

## Kubernetes

The repository includes deployment templates for the Streamlit UI and general
MCP server. They require cluster-specific review: the manifests default to the
`dev` image tag, include ALCF proxy settings, expose LoadBalancer services, and
do not configure persistent storage or application authentication. See the
[Kubernetes guide](https://argonne-lcf.github.io/ChemGraph/kubernetes/) before
applying files under [`k8s/`](k8s/README.md).

## Distributed and HPC execution

ChemGraph includes pluggable execution backends for local processes, Parsl,
Ensemble Launcher, and Globus Compute, plus an Academy-based persistent
multi-agent campaign runtime. These paths require additional dependencies,
site configuration, allocations, endpoints, or credentials and are not part
of the first-run workflow.

Start with:

- [HPC and Academy](https://argonne-lcf.github.io/ChemGraph/hpc_and_academy/)
- [`scripts/demo/`](scripts/demo/README.md) for execution-backend demos
- [Academy MACE screening example](examples/academy/example-002-mace-ensemble-screening/README.md)
- [Connecting to Argo from an ALCF compute node](examples/connecting_to_argo/README.md)

## Configuration

Most first runs need only command-line flags and environment variables. A TOML
file is useful for provider endpoints, MCP connections, logging, UI settings,
evaluation profiles, and execution backends:

```bash
chemgraph run --config config.toml -q "What is the SMILES string for water?"
```

The CLI and Streamlit UI do not consume every historical key in the repository
example identically. Use the [configuration reference](https://argonne-lcf.github.io/ChemGraph/configuration_with_toml/)
to see which settings are active in each interface. Never store API keys or
access tokens in a committed TOML file.

## Troubleshooting

```bash
chemgraph --help
chemgraph models
chemgraph run --check-keys
chemgraph run -vv -q "What is the SMILES string for water?"
```

- Calculator warnings at startup mean an optional engine was not detected;
  install it only if the requested workflow needs it.
- A first MACE or local-embedding run may pause while model weights download.
- `main_agent` requires `--interactive`.
- The Streamlit source command must be run from a repository checkout.
- A stale installed package can shadow a checkout; activate the intended
  environment and use an editable install.

See the [troubleshooting guide](https://argonne-lcf.github.io/ChemGraph/troubleshooting/)
for provider, calculator, path, UI, MCP, and session diagnostics.

## Documentation and examples

- [Full documentation](https://argonne-lcf.github.io/ChemGraph/)
- [Example notebooks and runnable guides](https://argonne-lcf.github.io/ChemGraph/example_usage/)
- [Evaluation and benchmarking](https://argonne-lcf.github.io/ChemGraph/evaluation/)
- [ChemGraph Leaderboard](https://huggingface.co/spaces/Autonomous-Scientific-Agents/chemgraph-leaderboard)
- [Project structure](https://argonne-lcf.github.io/ChemGraph/project_structure/)
- [Contributing guide](CONTRIBUTING.md)

## Citation

If ChemGraph supports your research, cite:

> Thang D. Pham, Aditya Tanikanti, and Murat Keçeli. “ChemGraph as an
> agentic framework for computational chemistry workflows.”
> *Communications Chemistry* 9, 33 (2026).
> <https://doi.org/10.1038/s42004-025-01776-9>

Users of the HPC orchestration features should also cite the
[multi-agent orchestration preprint](https://arxiv.org/abs/2604.07681).
BibTeX is available on the [citation page](https://argonne-lcf.github.io/ChemGraph/citation/).

## Contributing and license

Contributions are welcome. Branch from the latest `main`, keep each change
focused, and run `ruff check .` plus `pytest tests/ -k "not tblite"` before
opening a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete
workflow.

ChemGraph is distributed under the [Apache License 2.0](LICENSE).
