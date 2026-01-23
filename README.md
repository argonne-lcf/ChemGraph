# ChemGraph

<details>
  <summary><strong>Overview</strong></summary>

**ChemGraph** is an agentic framework that can automate molecular simulation workflows using large language models (LLMs). Built on top of `LangGraph` and `ASE`, ChemGraph allows users to perform complex computational chemistry tasks, from structure generation to thermochemistry calculations, with a natural language interface. 
ChemGraph supports diverse simulation backends, including ab initio quantum chemistry methods (e.g. coupled-cluster, DFT via NWChem, ORCA), semi-empirical methods (e.g., XTB via TBLite), and machine learning potentials (e.g, MACE, UMA) through a modular integration with `ASE`. 

</details>

<details>
  <summary><strong>Installation Instructions</strong></summary>

Ensure you have **Python 3.10 or higher** installed on your system. 
**Using pip (Recommended for most users)**

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

**Using Conda (Alternative)**

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
   - Python 3.10
   - Core packages (numpy, pandas, pytest, rich, toml)
   - Computational chemistry packages (nwchem, tblite)
   - All ChemGraph dependencies via pip
   

**Using uv (Alternative)**

1. Clone the repository:
   ```bash
   git clone https://github.com/Autonomous-Scientific-Agents/ChemGraph
   cd ChemGraph
   ```

2. Create and activate a virtual environment using uv:
    ```bash
    uv venv --python 3.11 chemgraph-env
    # uv venv --python 3.11 chemgraph-env # For specific python version

    source chemgraph-env/bin/activate # Unix/macos
    # OR
    .\chemgraph-env\Scripts\activate  # On Windows
   ```

3. Install ChemGraph using uv:
    ```bash
    uv pip install -e .
    ```

**Optional: Install with UMA support**

> **Note on e3nn Conflict for UMA Installation:** The `uma` extras (requiring `e3nn>=0.5`) conflict with the base `mace-torch` dependency (which pins `e3nn==0.4.4`). 
> If you need to install UMA support in an environment where `mace-torch` might cause this conflict, you can try the following workaround:
> 1. **Temporarily modify `pyproject.toml`**: Open the `pyproject.toml` file in the root of the ChemGraph project.
> 2. Find the line containing `"mace-torch>=0.3.13",` in the `dependencies` list.
> 3. Comment out this line by adding a `#` at the beginning (e.g., `#    "mace-torch>=0.3.13",`).
> 4. **Install UMA extras**: Run `pip install -e ".[uma]"`.
> 5. **(Optional) Restore `pyproject.toml`**: After installation, you can uncomment the `mace-torch` line if you still need it for other purposes in the same environment. Be aware that `mace-torch` might not function correctly due to the `e3nn` version mismatch (`e3nn>=0.5` will be present for UMA).
>
> **The most robust solution for using both MACE and UMA with their correct dependencies is to create separate Conda environments, as highlighted in the "Note on Compatibility" above.**

> **Important for UMA Model Access:** The `facebook/UMA` model is a gated model on Hugging Face. To use it, you must:
> 1. Visit the [facebook/UMA model page](https://huggingface.co/facebook/UMA) on Hugging Face.
> 2. Log in with your Hugging Face account.
> 3. Accept the model's terms and conditions if prompted.
> Your environment (local or CI) must also be authenticated with Hugging Face, typically by logging in via `huggingface-cli login` or ensuring `HF_TOKEN` is set and recognized.

```bash
pip install -e ".[uma]"
```
</details>

<details>
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

   - **[Multi-Agent System](notebooks/2_Demo-multi_agent.ipynb)**: This notebook demonstrates a multi-agent setup where different agents (Planner, Executor and Aggregator) handle various tasks exemplifying the collaborative potential of ChemGraph.

   - **[Model Context Protocol (MCP) Server](notebooks/3_MCP_server.ipynb)**: This notebook demonstrates how to run an MCP server and connect to ChemGraph.
   - **[Single-Agent System with gRASPA](notebooks/Demo_graspa_agent.ipynb)**: This notebook provides a sample guide on executing a gRASPA simulation using a single agent. For gRASPA-related installation instructions, visit the [gRASPA GitHub repository](https://github.com/snurr-group/gRASPA). The notebook's functionality has been validated on a single compute node at ALCF Polaris.

   - **[Infrared absorption spectrum prediction](notebooks/Demo_infrared_spectrum.ipynb)**: This notebook demonstrates how to calculate an infrared absorption spectrum.


</details>

<details>
  <summary><strong>Streamlit Web Interface</strong></summary>

ChemGraph includes a **Streamlit web interface** that provides an intuitive, chat-based UI for interacting with computational chemistry agents. The interface supports 3D molecular visualization, conversation history, and easy access to various ChemGraph workflows.

### Features

- **🧪 Interactive Chat Interface**: Natural language queries for computational chemistry tasks
- **🧬 3D Molecular Visualization**: Interactive molecular structure display using `stmol` and `py3Dmol`
- **📊 Report Integration**: Embedded HTML reports from computational calculations
- **💾 Data Export**: Download molecular structures as XYZ or JSON files
- **🔧 Multiple Workflows**: Support for single-agent, multi-agent, Python REPL, and gRASPA workflows
- **🎨 Modern UI**: Clean, responsive interface with conversation bubbles and molecular properties display

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
   streamlit run ui/app.py
   ```

3. **Access the interface**: Open your browser to `http://localhost:8501`

### Using the Interface

#### Configuration
- **Model Selection**: Choose from GPT-4o, GPT-4o-mini, or Claude models
- **Workflow Type**: Select single-agent, multi-agent, Python REPL, or gRASPA workflows


#### Interaction
1. **Initialize Agent**: Click "Initialize Agent" in the sidebar to set up your ChemGraph instance
2. **Ask Questions**: Use the text area to enter computational chemistry queries
3. **View Results**: See responses in chat bubbles with automatic structure detection
4. **3D Visualization**: When molecular structures are detected, they're automatically displayed in 3D
5. **Download Data**: Export structures and calculation results directly from the interface

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
- **Structure Detection**: Molecular structures are automatically extracted and visualized
- **Report Integration**: HTML reports from calculations are embedded directly in the interface
- **Debug Information**: Expandable sections show detailed message processing information

### Troubleshooting

**3D Visualization Issues:**
- Ensure `stmol` is installed: `pip install stmol`
- If 3D display fails, the interface falls back to table/text display
- Check browser compatibility for WebGL support

**Agent Initialization:**
- Verify API keys are set correctly
- Check that ChemGraph package is installed: `pip install -e .`
- Ensure all dependencies are available in your environment

**Performance:**
- For large molecular systems, visualization may take longer to load
- Use the refresh button if the interface becomes unresponsive
- Clear conversation history to improve performance with many queries

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
# Workflow type: single_agent, multi_agent, python_repl, graspa
workflow = "single_agent"
# Output format: state, last_message
output = "state"
# Enable structured output
structured = false
# Generate detailed reports
report = true

# Recursion limit for agent workflows
recursion_limit = 20
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

[api.anthropic]
base_url = "https://api.anthropic.com"
timeout = 30

[api.google]
base_url = "https://generativelanguage.googleapis.com/v1beta"
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

# Environment-specific configurations
[environments]
[environments.development]
model = "gpt-4o-mini"
temperature = 0.2
verbose = true
enable_cache = false

[environments.production]
model = "gpt-4o"
temperature = 0.1
verbose = false
enable_cache = true
rate_limit = true

[environments.testing]
model = "gpt-4o-mini"
temperature = 0.0
verbose = true
enable_cache = false
max_tokens = 1000
```

### Using Configuration Files

#### With the Command Line Interface

```bash
# Use configuration file
chemgraph --config config.toml -q "What is the SMILES string for water?"

# Override specific settings
chemgraph --config config.toml -q "Optimize methane" -m gpt-4o --verbose
```

#### Environment-Specific Configuration

Set the `CHEMGRAPH_ENV` environment variable to use environment-specific settings:

```bash
# Use development environment settings
export CHEMGRAPH_ENV=development
chemgraph --config config.toml -q "Your query"

# Use production environment settings
export CHEMGRAPH_ENV=production
chemgraph --config config.toml -q "Your query"
```

### Configuration Sections

| Section          | Description                                             |
| ---------------- | ------------------------------------------------------- |
| `[general]`      | Basic settings like model, workflow, and output format  |
| `[llm]`          | LLM-specific parameters (temperature, max_tokens, etc.) |
| `[api]`          | API endpoints and timeouts for different providers      |
| `[chemistry]`    | Chemistry-specific calculation settings                 |
| `[output]`       | Output file formats and visualization settings          |
| `[logging]`      | Logging configuration and verbosity levels              |
| `[features]`     | Feature flags and experimental settings                 |
| `[security]`     | Security settings and rate limiting                     |
| `[environments]` | Environment-specific configuration overrides            |

### Command Line Interface

ChemGraph includes a powerful command-line interface (CLI) that provides all the functionality of the web interface through the terminal. The CLI features rich formatting, interactive mode, and comprehensive configuration options.

#### Installation & Setup

The CLI is included by default when you install ChemGraph:

```bash
pip install -e .
```

#### Basic Usage

##### Quick Start

```bash
# Basic query
chemgraph -q "What is the SMILES string for water?"

# With model selection
chemgraph -q "Optimize methane geometry" -m gpt-4o

# With report generation
chemgraph -q "Calculate CO2 vibrational frequencies" -r

# Using configuration file
chemgraph --config config.toml -q "Your query here"
```

##### Command Syntax

```bash
chemgraph [OPTIONS] -q "YOUR_QUERY"
```

#### Command Line Options

**Core Arguments:**

| Option         | Short | Description                                  | Default        |
| -------------- | ----- | -------------------------------------------- | -------------- |
| `--query`      | `-q`  | The computational chemistry query to execute | Required       |
| `--model`      | `-m`  | LLM model to use                             | `gpt-4o-mini`  |
| `--workflow`   | `-w`  | Workflow type                                | `single_agent` |
| `--output`     | `-o`  | Output format (`state`, `last_message`)      | `state`        |
| `--structured` | `-s`  | Use structured output format                 | `False`        |
| `--report`     | `-r`  | Generate detailed report                     | `False`        |

**Model Selection:**

```bash
# OpenAI models
chemgraph -q "Your query" -m gpt-4o
chemgraph -q "Your query" -m gpt-4o-mini
chemgraph -q "Your query" -m o1-preview

# Anthropic models
chemgraph -q "Your query" -m claude-3-5-sonnet-20241022
chemgraph -q "Your query" -m claude-3-opus-20240229

# Google models
chemgraph -q "Your query" -m gemini-1.5-pro

# Local models (requires vLLM server)
chemgraph -q "Your query" -m llama-3.1-70b-instruct
```

**Workflow Types:**

```bash
# Single agent (default) - best for most tasks
chemgraph -q "Optimize water molecule" -w single_agent

# Multi-agent - complex tasks with planning
chemgraph -q "Complex analysis" -w multi_agent

# Python REPL - interactive coding
chemgraph -q "Write analysis code" -w python_repl

# gRASPA - molecular simulation
chemgraph -q "Run adsorption simulation" -w graspa
```

**Output Formats:**

```bash
# Full state output (default)
chemgraph -q "Your query" -o state

# Last message only
chemgraph -q "Your query" -o last_message

# Structured output
chemgraph -q "Your query" -s

# Generate detailed report
chemgraph -q "Your query" -r
```

#### Interactive Mode

Start an interactive session for continuous conversations:

```bash
chemgraph --interactive
```

**Interactive Features:**
- **Persistent conversation**: Maintain context across queries
- **Model switching**: Change models mid-conversation
- **Workflow switching**: Switch between different agent types
- **Built-in commands**: Help, clear, config, etc.

**Interactive Commands:**
```bash
# In interactive mode, type:
help                    # Show available commands
clear                   # Clear screen
config                  # Show current configuration
quit                    # Exit interactive mode
model gpt-4o           # Change model
workflow multi_agent   # Change workflow
```

#### Utility Commands

**List Available Models:**
```bash
chemgraph --list-models
```

**Check API Keys:**
```bash
chemgraph --check-keys
```

**Get Help:**
```bash
chemgraph --help
```

#### Configuration File Support

Use TOML configuration files for consistent settings:

```bash
chemgraph --config config.toml -q "Your query"
```

#### Environment Variables

Set environment-specific configurations:

```bash
# Use development settings
export CHEMGRAPH_ENV=development
chemgraph --config config.toml -q "Your query"

# Use production settings
export CHEMGRAPH_ENV=production
chemgraph --config config.toml -q "Your query"
```

#### Advanced Options

**Timeout and Error Handling:**
```bash
# Set recursion limit
chemgraph -q "Complex query" --recursion-limit 30

# Verbose output for debugging
chemgraph -q "Your query" -v

# Save output to file
chemgraph -q "Your query" --output-file results.txt
```



#### Example Workflows

**Basic Molecular Analysis:**
```bash
# Get molecular structure
chemgraph -q "What is the SMILES string for caffeine?"

# Optimize geometry
chemgraph -q "Optimize the geometry of caffeine using DFT" -m gpt-4o -r

# Calculate properties
chemgraph -q "Calculate the vibrational frequencies of optimized caffeine" -r
```

**Interactive Research Session:**
```bash
# Start interactive mode
chemgraph --interactive

# Select model and workflow
> model gpt-4o
> workflow single_agent

# Conduct analysis
> What is the structure of aspirin?
> Optimize its geometry using DFT
> Calculate its electronic properties
> Compare with ibuprofen
```

**Batch Processing:**
```bash
# Process multiple queries
chemgraph -q "Analyze water molecule" --output-file water_analysis.txt
chemgraph -q "Analyze methane molecule" --output-file methane_analysis.txt
chemgraph -q "Analyze ammonia molecule" --output-file ammonia_analysis.txt
```

#### API Key Setup

**Required API Keys:**
```bash
# OpenAI (for GPT models)
export OPENAI_API_KEY="your_openai_key_here"

# Anthropic (for Claude models)
export ANTHROPIC_API_KEY="your_anthropic_key_here"

# Google (for Gemini models)
export GEMINI_API_KEY="your_gemini_key_here"
```

**Getting API Keys:**
- **OpenAI**: Visit [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic**: Visit [console.anthropic.com](https://console.anthropic.com/)
- **Google**: Visit [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

#### Performance Tips

- Use `gpt-4o-mini` for faster, cost-effective queries
- Use `gpt-4o` for complex analysis requiring higher reasoning
- Enable `--report` for detailed documentation
- Use `--structured` output for programmatic parsing
- Leverage configuration files for consistent settings

#### Troubleshooting

**Common Issues:**
```bash
# Check API key status
chemgraph --check-keys

# Verify model availability
chemgraph --list-models

# Test with verbose output
chemgraph -q "test query" -v

# Check configuration
chemgraph --config config.toml -q "test" --verbose
```

**Error Messages:**
- **"Invalid model"**: Use `--list-models` to see available options
- **"API key not found"**: Use `--check-keys` to verify setup
- **"Query required"**: Use `-q` to specify your query
- **"Timeout"**: Increase `--recursion-limit` or simplify query

The CLI provides:
- **Beautiful terminal output** with colors and formatting powered by Rich
- **API key validation** before agent initialization
- **Timeout protection** to prevent hanging processes
- **Interactive mode** for continuous conversations
- **Configuration file support** with TOML format
- **Environment-specific settings** for development/production
- **Comprehensive help** and examples for all features

</details>

<details>
  <summary><strong>Project Structure</strong></summary>

```
chemgraph/
│
├── src/                       # Source code
│   ├── chemgraph/             # Top-level package
│   │   ├── agent/             # Agent-based task management
│   │   ├── graphs/            # Workflow graph utilities
│   │   ├── models/            # Different Pydantic models
│   │   ├── prompt/            # Agent prompt
│   │   ├── state/             # Agent state
│   │   ├── tools/             # Tools for molecular simulations
│   │   ├── utils/             # Other utility functions
│
├── pyproject.toml             # Project configuration
└── README.md                  # Project documentation
```

</details>

<details>
  <summary><strong>Running Local Models with vLLM</strong></summary>
This section describes how to set up and run local language models using the vLLM inference server.

### Inference Backend Setup (Remote/Local)

#### Virtual Python Environment
All instructions below must be executed within a Python virtual environment. Ensure the virtual environment uses the same Python version as your project (e.g., Python 3.11).

**Example 1: Using conda**
```bash
conda create -n vllm-env python=3.11 -y
conda activate vllm-env
```

**Example 2: Using python venv**
```bash
python3.11 -m venv vllm-env
source vllm-env/bin/activate  # On Windows use `vllm-env\\Scripts\\activate`
```

#### Install Inference Server (vLLM)
vLLM is recommended for serving many transformer models efficiently.

**Basic vLLM installation from source:**
Make sure your virtual environment is activated.
```bash
# Ensure git is installed
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```
For specific hardware acceleration (e.g., CUDA, ROCm), refer to the [official vLLM installation documentation](https://docs.vllm.ai/en/latest/getting_started/installation.html).

#### Running the vLLM Server (Standalone)

A script is provided at `scripts/run_vllm_server.sh` to help start a vLLM server with features like logging, retry attempts, and timeout. This is useful for running vLLM outside of Docker Compose, for example, directly on a machine with GPU access.

**Before running the script:**
1.  Ensure your vLLM Python virtual environment is activated.
    ```bash
    # Example: if you used conda
    # conda activate vllm-env 
    # Example: if you used python venv
    # source path/to/your/vllm-env/bin/activate
    ```
2.  Make the script executable:
    ```bash
    chmod +x scripts/run_vllm_server.sh
    ```

**To run the script:**

```bash
./scripts/run_vllm_server.sh [MODEL_IDENTIFIER] [PORT] [MAX_MODEL_LENGTH]
```

-   `[MODEL_IDENTIFIER]` (optional): The Hugging Face model identifier. Defaults to `facebook/opt-125m`.
-   `[PORT]` (optional): The port for the vLLM server. Defaults to `8001`.
-   `[MAX_MODEL_LENGTH]` (optional): The maximum model length. Defaults to `4096`.

**Example:**
```bash
./scripts/run_vllm_server.sh meta-llama/Meta-Llama-3-8B-Instruct 8001 8192
```

**Important Note on Gated Models (e.g., Llama 3):**
Many models, such as those from the Llama family by Meta, are gated and require you to accept their terms of use on Hugging Face and use an access token for download. 

To use such models with vLLM (either via the script or Docker Compose):
1.  **Hugging Face Account and Token**: Ensure you have a Hugging Face account and have generated an access token with `read` permissions. You can find this in your Hugging Face account settings under "Access Tokens".
2.  **Accept Model License**: Navigate to the Hugging Face page of the specific model you want to use (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`) and accept its license/terms if prompted.
3.  **Environment Variables**: Before running the vLLM server (either via the script or `docker-compose up`), you need to set the following environment variables in your terminal session or within your environment configuration (e.g., `.bashrc`, `.zshrc`, or by passing them to Docker Compose if applicable):
    ```bash
    export HF_TOKEN="your_hugging_face_token_here"
    # Optional: Specify a directory for Hugging Face to download models and cache.
    # export HF_HOME="/path/to/your/huggingface_cache_directory"
    ```
    vLLM will use these environment variables to authenticate with Hugging Face and download the model weights.

The script will:
- Attempt to start the vLLM OpenAI-compatible API server.
- Log output to a file in the `logs/` directory (created if it doesn't exist at the project root).
- The server runs in the background via `nohup`.

This standalone script is an alternative to running vLLM via Docker Compose and is primarily for users who manage their vLLM instances directly.
</details>

<details>
  <summary><strong>Docker Support with Docker Compose (Recommended for vLLM)</strong></summary>

This project uses Docker Compose to manage multi-container applications, providing a consistent development and deployment environment. This setup allows you to run the `chemgraph` (with JupyterLab) and a local vLLM model server as separate, inter-communicating services.

**Prerequisites**

- [Docker](https://docs.docker.com/get-docker/) installed on your system.
- [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.
- [vllm](https://github.com/vllm-project/vllm) cloned into the project root. `git clone https://github.com/vllm-project/vllm.git`

**Overview**

The `docker-compose.yml` file defines two main services:
1.  **`jupyter_lab`**: 
    *   Builds from the main `Dockerfile`.
    *   Runs JupyterLab, allowing you to interact with the notebooks and agent code.
    *   Is configured to communicate with the `vllm_server`.
2.  **`vllm_server`**:
    *   Builds from `Dockerfile.arm` by default (located in the project root), which is suitable for running vLLM on macOS (Apple Silicon / ARM-based CPUs). This Dockerfile is a modified version intended for CPU execution.
    *   For other operating systems or hardware (e.g., Linux with NVIDIA GPUs), you will need to use a different Dockerfile. The vLLM project provides a collection of Dockerfiles for various architectures (CPU, CUDA, ROCm, etc.) available at [https://github.com/vllm-project/vllm/tree/main/docker](https://github.com/vllm-project/vllm/tree/main/docker). You would need to adjust the `docker-compose.yml` to point to the appropriate Dockerfile and context (e.g., by cloning the vLLM repository locally and referencing a Dockerfile within it).
    *   Starts an OpenAI-compatible API server using vLLM, serving a pre-configured model (e.g., `meta-llama/Llama-3-8B-Instruct` as per the current `docker-compose.yml`).
    *   Listens on port 8000 within the Docker network (and is exposed to host port 8001 by default).

**Building and Running with Docker Compose**

Navigate to the root directory of the project (where `docker-compose.yml` is located) and run:

```bash
docker-compose up --build
```

**Note on Hugging Face Token (`HF_TOKEN`):**
Many models, including the default `meta-llama/Llama-3-8B-Instruct`, are gated and require Hugging Face authentication. To provide your Hugging Face token to the `vllm_server` service:

1.  **Create a `.env` file** in the root directory of the project (the same directory as `docker-compose.yml`).
2.  Add your Hugging Face token to this file:
    ```
    HF_TOKEN="your_actual_hugging_face_token_here"
    ```
    
Docker Compose will automatically load this variable when you run `docker-compose up`. The `vllm_server` in `docker-compose.yml` is configured to use this environment variable.

Breakdown of the command:
- `docker-compose up`: Starts or restarts all services defined in `docker-compose.yml`.
- `--build`: Forces Docker Compose to build the images before starting the containers. This is useful if you've made changes to `Dockerfile`, `Dockerfile.arm` (or other vLLM Dockerfiles), or project dependencies.

After running this command:
- The vLLM server will start, and its logs will be streamed to your terminal.
- JupyterLab will start, and its logs will also be streamed. JupyterLab will be accessible in your web browser at `http://localhost:8888`. No token is required by default.

To stop the services, press `Ctrl+C` in the terminal where `docker-compose up` is running. To stop and remove the containers, you can use `docker-compose down`.

### Configuring Notebooks to Use the Local vLLM Server

When you initialize `ChemGraph` in your Jupyter notebooks (running within the `jupyter_lab` service), you can now point to the local vLLM server:

1.  **Model Name**: Use the Hugging Face identifier of the model being served by vLLM (e.g., `meta-llama/Llama-3-8B-Instruct` as per default in `docker-compose.yml`).
2.  **Base URL & API Key**: These are automatically passed as environment variables (`VLLM_BASE_URL` and `OPENAI_API_KEY`) to the `jupyter_lab` service by `docker-compose.yml`. The agent code in `llm_agent.py` has been updated to automatically use these environment variables if a model name is provided that isn't in the pre-defined supported lists (OpenAI, Ollama, ALCF, Anthropic).

**Example in a notebook:**

```python
from chemgraph.agent.llm_agent import ChemGraph

# The model name should match what vLLM is serving.
# The base_url and api_key will be picked up from environment variables
# set in docker-compose.yml if this model_name is not a standard one.
agent = ChemGraph(
    model_name="meta-llama/Llama-3-8B-Instruct", # Or whatever model is configured in docker-compose.yml
    workflow_type="single_agent", 
    # No need to explicitly pass base_url or api_key here if using the docker-compose setup
)

# Now you can run the agent
# response = agent.run("What is the SMILES string for water?")
# print(response)
```

The `jupyter_lab` service will connect to `http://vllm_server:8000/v1` (as defined by `VLLM_BASE_URL` in `docker-compose.yml`) to make requests to the language model.

### GPU Support for vLLM (Advanced)

The provided `Dockerfile.arm` and the default `docker-compose.yml` setup are configured for CPU-based vLLM (suitable for macOS). To enable GPU support (typically on Linux with NVIDIA GPUs):

1.  **Choose the Correct vLLM Dockerfile**:
    *   Do **not** use `Dockerfile.arm`.
    *   You will need to use a Dockerfile from the official vLLM repository designed for CUDA. Clone the vLLM repository (e.g., into a `./vllm` subdirectory in your project) or use it as a submodule.
    *   A common choice is `vllm/docker/Dockerfile` (for CUDA) or a specific version like `vllm/docker/Dockerfile.cuda-12.1`. Refer to [vLLM Dockerfiles](https://github.com/vllm-project/vllm/tree/main/docker) for options.
2.  **Modify `docker-compose.yml`**:
    *   Change the `build.context` for the `vllm_server` service to point to your local clone of the vLLM repository (e.g., `./vllm`).
    *   Change the `build.dockerfile` to the path of the CUDA-enabled Dockerfile within that context (e.g., `docker/Dockerfile`).
    *   Uncomment and configure the `deploy.resources.reservations.devices` section for the `vllm_server` service to grant it GPU access.

    ```yaml
    # ... in docker-compose.yml, for vllm_server:
    # build:
    #   context: ./vllm  # Path to your local vLLM repo clone
    #   dockerfile: docker/Dockerfile # Path to the CUDA Dockerfile within the vLLM repo
    # ...
    # environment:
      # Remove or comment out:
      # - VLLM_CPU_ONLY=1 
      # ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1 # or 'all'
              capabilities: [gpu]
    ```
3.  **NVIDIA Container Toolkit**: Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your host system for Docker to recognize and use NVIDIA GPUs.
4.  **Build Arguments**: Some official vLLM Dockerfiles accept build arguments (e.g., `CUDA_VERSION`, `PYTHON_VERSION`). You might need to pass these via the `build.args` section in `docker-compose.yml`.

    ```yaml
    # ... in docker-compose.yml, for vllm_server build:
    # args:
    #   - CUDA_VERSION=12.1.0 
    #   - PYTHON_VERSION=3.10 
    ```
    Consult the specific vLLM Dockerfile you choose for available build arguments.

### Running Only JupyterLab (for External LLM Services)

If you prefer to use external LLM services like OpenAI, Claude, or other hosted providers instead of running a local vLLM server, you can run only the JupyterLab service:

```bash
docker-compose up jupyter_lab
```

This will start only the JupyterLab container without the vLLM server. In this setup:

1. **JupyterLab Access**: JupyterLab will be available at `http://localhost:8888`
2. **LLM Configuration**: In your notebooks, configure the agent to use external services by providing appropriate model names and API keys:

**Example for OpenAI:**
```python
import os
from chemgraph.agent.llm_agent import ChemGraph

# Set your OpenAI API key as an environment variable or pass it directly
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

agent = ChemGraph(
    model_name="gpt-4",  # or "gpt-3.5-turbo", "gpt-4o", etc.
    workflow_type="single_agent"
)
```

**Example for Anthropic Claude:**
```python
import os
from chemgraph.agent.llm_agent import ChemGraph

# Set your Anthropic API key
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key-here"

agent = ChemGraph(
    model_name="claude-3-sonnet-20240229",  # or other Claude models
    workflow_type="single_agent_ase"
)
```

**Available Environment Variables for External Services:**
- `OPENAI_API_KEY`: For OpenAI models
- `ANTHROPIC_API_KEY`: For Anthropic Claude models
- `GEMINI_API_KEY`: For Gemini models

### Working with Example Notebooks

Once JupyterLab is running (via `docker-compose up` or `docker-compose up jupyter_lab`), you can navigate to the `notebooks/` directory within the JupyterLab interface to open and run the example notebooks. Modify them as shown above to use either the locally served vLLM model or external LLM services.

### Notes on TBLite Python API

The `tblite` package is installed via pip within the `jupyter_lab` service. For the full Python API functionality of TBLite (especially for XTB), you might need to follow separate installation instructions as mentioned in the [TBLite documentation](https://tblite.readthedocs.io/en/latest/installation.html). If you require this, you may need to modify the main `Dockerfile` to include these additional installation steps or perform them inside a running container and commit the changes to a new image for the `jupyter_lab` service.

</details>

<details>
  <summary><strong>Model Context Protocol (MCP) Servers</strong></summary>

ChemGraph provides several **MCP servers** that expose its capabilities as standardized tools conforming to the [Model Context Protocol](https://modelcontextprotocol.io/). These can be connected to MCP-compliant clients or other agentic frameworks.

### Available Servers

The servers are located in `src/chemgraph/mcp/`:

*   **`mcp_tools.py`**: General-purpose chemistry tools powered by ASE. Supports:
    *   Geometry optimization (GetStructure, RunASE)
    *   Energy/Thermochemistry calculations
    *   File I/O (handling XYZ, JSON, etc.)
*   **`mace_mcp_parsl.py`**: Tools for running **MACE** (Machine Learning Potential) simulations.
    *   Supports single-structure calculations.
    *   Supports **ensemble** calculations (directories of structures) using **Parsl** for parallel execution on HPC systems (e.g., Polaris, Aurora).
*   **`graspa_mcp_parsl.py`**: Tools for **gRASPA** simulations (Gas Adsorption in MOFs).
    *   Supports single and ensemble runs via Parsl.
*   **`data_analysis_mcp.py`**: Tools for analyzing simulation results.
    *   Aggregating JSONL logs from ensemble runs into CSV/DataFrames.
    *   Plotting isotherms and other data.

### Running a Server

You can run the servers using Python. They support both `stdio` (default) and `streamable_http` (SSE) transports.

**Basic Usage (stdio)**
Connect this directly to your MCP client (e.g., Claude Desktop config):

```bash
python src/chemgraph/mcp/mcp_tools.py
```

**Using HTTP/SSE**
To run a server that listens for HTTP connections (useful for remote deployment or debugging):

```bash
python src/chemgraph/mcp/mcp_tools.py --transport streamable_http --port 8000
```

**Configuration via Arguments**
All servers in `src/chemgraph/mcp/` support the following arguments:
*   `--transport`: `stdio` (default) or `streamable_http`.
*   `--port`: Port number (for HTTP transport).
*   `--host`: Host address (default: 127.0.0.1).

**Note on HPC Servers:**
For `mace_mcp_parsl.py` and `graspa_mcp_parsl.py`, ensure your environment is configured for the target HPC system if running actual parallel jobs. They leverage `chemgraph.hpc_configs` to load system-specific Parsl configurations (like `polaris` or `aurora`).

</details>

<details>
  <summary><strong>Code Formatting & Linting</strong></summary>

This project uses [Ruff](https://github.com/astral-sh/ruff) for **both formatting and linting**. To ensure all code follows our style guidelines, install the pre-commit hook:

```sh
pip install pre-commit
pre-commit install
```
</details>

<details>
  <summary><strong>Citation</strong></summary>
    
  If you use ChemGraph in your research, please cite our work:
    
  ```bibtex
    @article{pham_chemgraph_2026,
      title = {{ChemGraph} as an agentic framework for computational chemistry workflows},
      issn = {2399-3669},
      url = {https://doi.org/10.1038/s42004-025-01776-9},
      doi = {10.1038/s42004-025-01776-9},
      journaltitle = {Communications Chemistry},
      shortjournal = {Communications Chemistry},
      author = {Pham, Thang D. and Tanikanti, Aditya and Keçeli, Murat},
      date = {2026-01-08},
    }
  ```
 </details>
<details>
  <summary><strong>Acknowledgments</strong></summary>
This research used resources of the Argonne Leadership Computing Facility, a U.S.
Department of Energy (DOE) Office of Science user facility at Argonne National
Laboratory and is based on research supported by the U.S. DOE Office of Science-
Advanced Scientific Computing Research Program, under Contract No. DE-AC02-
06CH11357. Our work leverages ALCF Inference Endpoints, which provide a robust API
for LLM inference on ALCF HPC clusters via Globus Compute. We are thankful to Serkan
Altuntaş for his contributions to the user interface of ChemGraph and for insightful
discussions on AIOps.
</details>

<details>
  <summary><strong>License</strong></summary>
This project is licensed under the Apache 2.0 License.
</details>
