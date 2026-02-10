!!! note
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
```

### Using Configuration Files

#### With the Command Line Interface

```bash
# Use configuration file
chemgraph --config config.toml -q "What is the SMILES string for water?"

# Override specific settings
chemgraph --config config.toml -q "Optimize methane" -m gpt-4o --verbose
```

#### Argo/OpenAI-Compatible Endpoints

For Argo or any OpenAI-compatible endpoint, set `api.openai.base_url` in `config.toml`.
Optional `api.openai.argo_user` can also be configured.

```toml
[api.openai]
base_url = "https://apps-dev.inside.anl.gov/argoapi/v1"
argo_user = "your_argo_username"
```

`ARGO_USER` is only used as a fallback when `argo_user` is not provided in `config.toml`.

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

# Local models (OpenAI-compatible local endpoint)
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

Provider keys and optional endpoint settings are read from environment variables
and `config.toml` (for example, `api.openai.base_url` and `api.openai.argo_user`).

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
