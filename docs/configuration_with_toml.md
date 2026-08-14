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
# Development-only workspace subagent for interactive main_agent sessions.
enable_deepagent = false
# deepagent_workspace = "."
# Durable main-agent checkpoint database
# checkpoint_db = "~/.chemgraph/checkpoints.db"
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

#### Argo/OpenAI-Compatible Endpoints

For Argo or any OpenAI-compatible endpoint, set `api.openai.base_url` in `config.toml`.
Optional `api.openai.argo_user` can also be configured.

```toml
[api.openai]
base_url = "https://apps-dev.inside.anl.gov/argoapi/v1"
argo_user = "your_argo_username"
```

`ARGO_USER` is only used as a fallback when `argo_user` is not provided in `config.toml`.

#### ALCF Inference Endpoints

ChemGraph supports [ALCF Inference Endpoints](https://docs.alcf.anl.gov/services/inference-endpoints/), which provide API access to open-source models running on dedicated ALCF hardware.

1. The endpoint is configured by default in `config.toml`:

```toml
[api.alcf]
base_url = "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
timeout = 30
```

2. Authenticate via Globus OAuth and set the access token:

```bash
pip install globus_sdk
wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py
python inference_auth_token.py authenticate
export ALCF_ACCESS_TOKEN=$(python inference_auth_token.py get_access_token)
```

3. Use an ALCF model (no prefix needed):

```bash
chemgraph --config config.toml -m meta-llama/Meta-Llama-3.1-70B-Instruct \
  -q "Calculate the energy of water using MACE"
```

Access tokens are valid for ~48 hours. See the
[ALCF docs](https://docs.alcf.anl.gov/services/inference-endpoints/#available-models) for available models.

#### Groq

ChemGraph supports [Groq](https://groq.com/) for fast LLM inference. Use the `groq:` prefix to route any model through Groq.

1. Set your API key:

```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

2. Use any Groq model with the `groq:` prefix:

```bash
chemgraph -q "What is the SMILES for water?" -m groq:llama-3.3-70b-versatile
chemgraph -q "Optimize methane" -m groq:openai/gpt-oss-120b
```

No curated model list is maintained -- any model available on the
[Groq console](https://console.groq.com/docs/models) can be used by prefixing
it with `groq:`. The prefix is stripped before sending to the Groq API.

#### LLM Provider Prefixes

For third-party providers that share model names with other services, ChemGraph
uses a prefix convention to route models unambiguously:

| Prefix | Provider | Auth Env Var | Example |
|--------|----------|--------------|---------|
| `argo:` | Argo API (Argonne internal) | `OPENAI_API_KEY` | `argo:gpt-4o` |
| `groq:` | Groq Cloud | `GROQ_API_KEY` | `groq:llama-3.3-70b-versatile` |

Direct model names (no prefix) are used for OpenAI, Anthropic, Google, ALCF, and Ollama.

### Configuration Sections

| Section          | Description                                             |
| ---------------- | ------------------------------------------------------- |
| `[general]`      | Basic settings like model, workflow, and output format  |
| `[llm]`          | Reserved/legacy LLM parameter documentation             |
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

| Option              | Short | Description                                           | Default        |
| ------------------- | ----- | ----------------------------------------------------- | -------------- |
| `--query`           | `-q`  | The computational chemistry query to execute          | Required       |
| `--model`           | `-m`  | LLM model to use                                     | `gpt-4o-mini`  |
| `--workflow`        | `-w`  | Workflow type                                        | `single_agent` |
| `--output`          | `-o`  | Output format (`state`, `last_message`)              | `state`        |
| `--structured`      | `-s`  | Use structured output format                         | `False`        |
| `--report`          | `-r`  | Generate detailed report                             | `False`        |
| `--deepagent`       |       | Enable the experimental `main_agent` workspace worker | `False`       |
| `--deepagent-workspace` |   | Root used by the experimental workspace worker       | Current directory |
| `--checkpoint-db`    |       | SQLite checkpoints for durable `main_agent` threads | `~/.chemgraph/checkpoints.db` |
| `--resume`          |       | Resume from a previous session ID (prefix supported) |                |
| `--list-sessions`   |       | List recent sessions from the memory database        |                |
| `--show-session`    |       | Show conversation for a session (prefix supported)   |                |
| `--delete-session`  |       | Delete a session from the memory database            |                |

**Model Selection:**

```bash
# OpenAI models
chemgraph -q "Your query" -m gpt-4o
chemgraph -q "Your query" -m gpt-4o-mini

# Anthropic models
chemgraph -q "Your query" -m claude-3-5-sonnet-20241022

# Google models
chemgraph -q "Your query" -m gemini-2.5-pro

# Argo models (Argonne internal, argo: prefix)
chemgraph -q "Your query" -m argo:gpt-4o
chemgraph -q "Your query" -m argo:claude-sonnet-4

# ALCF models (Globus auth required, no prefix)
chemgraph -q "Your query" -m meta-llama/Meta-Llama-3.1-70B-Instruct

# Groq models (groq: prefix, any Groq model)
chemgraph -q "Your query" -m groq:llama-3.3-70b-versatile

# Local models (Ollama)
chemgraph -q "Your query" -m llama3.2
```

**Workflow Types:**

```bash
# Single agent (default) - best for most tasks
chemgraph -q "Optimize water molecule" -w single_agent

# Long-lived supervisor - delegates chemistry tasks to the existing agent
chemgraph --interactive -w main_agent

# Multi-agent - complex tasks with planning
chemgraph -q "Complex analysis" -w multi_agent

# Python REPL - interactive coding
chemgraph -q "Write analysis code" -w python_repl

# gRASPA - molecular simulation
chemgraph -q "Run adsorption simulation" -w graspa

# Molecular docking - dock a candidate into a receptor (AutoDock Vina)
chemgraph -q "Dock aspirin into 'receptor.pdbqt'" -w molecular_docking
```

!!! note "Molecular docking workflow requirements"
    The `molecular_docking` workflow needs the optional docking dependency (`pip
    install -e ".[docking]"` for Meeko) plus AutoDock Vina from conda-forge
    (`conda install -c conda-forge vina`). The candidate may be a SMILES, a
    molecule name, or a PubChem CID; the receptor is a prepared rigid receptor
    `.pdbqt`, or a SMILES/name/CID for a small-molecule target. The search box is
    detected automatically.

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

To test the long-lived supervisor, select `main_agent` explicitly:

```bash
chemgraph --interactive -w main_agent
```

Each prompt runs on one durable thread, and chemistry work is delegated through
Deep Agents' `task` middleware tool. The CLI stores exact graph state in
`~/.chemgraph/checkpoints.db` and readable transcripts in
`~/.chemgraph/sessions.db`. Quitting or switching model/workflow leaves the old
thread available for exact recovery:

```bash
chemgraph --interactive -w main_agent --resume <session-id>
```

Use `/resume <session-id>` to switch threads inside the REPL. Pending
clarifications and approvals are presented immediately. Use `/retry` after a
recoverable failure; the original user message is not added again.

The checkpoint database is authoritative. The sessions database is a
best-effort readable projection: a projection write failure is logged but does
not change a completed graph result. Checkpoint serialization remains strict
and does not use pickle, so unsupported objects in graph state still fail the
authoritative graph operation. When a retry follows a different branch, the
readable transcript is replaced with the branch represented by the latest
checkpoint.

#### Experimental workspace Deep Agent

For development and testing, `main_agent` can register an additional
`deepagent` sibling alongside the existing `chemgraph` chemistry worker:

```bash
chemgraph --interactive -w main_agent --deepagent \
  --deepagent-workspace /absolute/path/to/workspace
```

The supervisor routes molecular simulations and calculator work to
`chemgraph`, while repository exploration, coding, file analysis, and test runs
can be delegated to `deepagent`. The equivalent TOML settings are
`general.enable_deepagent = true` and `general.deepagent_workspace = "..."`.
An explicit `--no-deepagent` overrides an enabled TOML setting.

!!! danger "Development-only host shell"
    The experimental CLI uses Deep Agents' `LocalShellBackend`. Its filesystem
    tools are rooted at the selected workspace, but shell commands are executed
    directly on the host and can access paths outside that root. ChemGraph
    displays a warning and requires startup confirmation, then requires an
    approve/reject decision before every shell command and every file
    write/edit/delete. Only a small environment allowlist is forwarded; API
    keys and token variables are not copied. Do not use this mode for deployed
    or untrusted workloads.

Python callers can instead pass any compatible Deep Agents backend without
using the host-shell CLI path:

```python
agent = ChemGraph(
    model_name="gpt-4o-mini",
    workflow_type="main_agent",
    enable_deepagent=True,
    deepagent_backend=my_sandbox_backend,
)
```

A production release must use an isolated sandbox backend with explicit user
approval, defined lifecycle and cleanup, artifact transfer, and network/secret
policies. The experimental local backend is not a production sandbox.

**Interactive Features:**
- **Persistent conversation**: Maintain context across queries
- **Session memory**: Standard workflows use summary-injection resume; `main_agent` additionally uses durable LangGraph checkpoints for exact recovery
- **Model switching**: Change models mid-conversation
- **Workflow switching**: Switch between different agent types
- **Built-in commands**: Help, clear, config, session management, etc.

**Interactive Commands:**
```bash
# In interactive mode, type:
/help                   # Show available commands
/clear                  # Clear screen
/config                 # Show current configuration and session ID
/quit                   # Exit interactive mode
/model gpt-4o           # Change model
/workflow multi_agent   # Change workflow

# Session management:
/history                # List recent sessions
/show <session_id>      # Show a session's conversation
/resume <session_id>    # Resume from a previous session
/retry                  # Retry a failed main_agent operation
```

Exact bare aliases for commands without arguments, such as `help`, `quit`, and
`history`, remain supported. Commands with arguments require `/`, so natural
prompts beginning with words such as `show`, `model`, or `workflow` are sent to
the active agent.

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

#### Session Memory

ChemGraph automatically saves every conversation to a local SQLite database at `~/.chemgraph/sessions.db`. This allows you to browse past sessions, review tool calls and results, and resume previous conversations with full context.

**List Recent Sessions:**
```bash
chemgraph --list-sessions
```

**View a Session's Conversation:**
```bash
# Full session ID or prefix (first few characters)
chemgraph --show-session a3b2
```

**Resume From a Previous Session:**
```bash
# Injects previous conversation context into the new query
chemgraph -q "Now optimize the geometry at 500K" --resume a3b2

# Restore exact main-agent state, including pending interrupts
chemgraph --interactive -w main_agent --resume a3b2
```

**Delete a Session:**
```bash
chemgraph --delete-session a3b2c1d4
```

Session IDs support prefix matching -- you only need to type enough characters to uniquely identify the session.

Main-agent checkpoints and full tool transcripts can contain sensitive prompts,
arguments, and outputs. They are unencrypted; ChemGraph restricts local file
permissions where the platform supports POSIX modes. SQLite is intended for
local use. Python deployments should inject an async production saver such as
`AsyncPostgresSaver` and retain ownership of its lifecycle.

Without an injected saver, Python `main_agent` sessions use process-local
memory checkpoints. Their readable records can be reviewed and deleted, but
they cannot be restored after the process that owns the checkpointer exits.

When injecting `AsyncSqliteSaver`, create, invoke, inspect, and close it on the
same event loop and pass it through `ChemGraph(..., checkpointer=saver)`. Exact
restore also requires recreating custom tools, prompts, credentials, MCP
bindings, and sandbox backends with the same graph topology.

```python
import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.agent.main_session import MainAgentSession


async def run_durable_thread():
    async with aiosqlite.connect("checkpoints.db") as connection:
        saver = AsyncSqliteSaver(
            connection,
            serde=JsonPlusSerializer(
                pickle_fallback=False,
                allowed_msgpack_modules=None,
            ),
        )
        await saver.setup()
        agent = ChemGraph(workflow_type="main_agent", checkpointer=saver)
        session = MainAgentSession(
            agent.workflow,
            thread_id="your-stable-thread-id",
            session_store=agent.session_store,
            session_metadata=agent.main_agent_metadata,
        )
        await session.run("Optimize a water molecule")
```

The CLI recreates its local-shell Deep Agent only after showing the host-access
warning and receiving fresh approval. Python callers must recreate their own
injected sandbox backend before restoring a thread.

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

# Groq (for groq: prefixed models)
export GROQ_API_KEY="your_groq_key_here"

# ALCF (Globus OAuth access token)
export ALCF_ACCESS_TOKEN=$(python inference_auth_token.py get_access_token)
```

**Getting API Keys:**
- **OpenAI**: Visit [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic**: Visit [console.anthropic.com](https://console.anthropic.com/)
- **Google**: Visit [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **Groq**: Visit [console.groq.com/keys](https://console.groq.com/keys)
- **ALCF**: See [ALCF Inference Endpoints docs](https://docs.alcf.anl.gov/services/inference-endpoints/#api-access)

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
