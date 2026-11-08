# ChemGraph

ChemGraph turns natural-language requests into computational chemistry and
materials-science workflows. It combines LangGraph and LangChain agents with
ASE, RDKit, MCP servers, and pluggable execution backends.

Use ChemGraph through the command line, Python, a Streamlit interface, or as an
MCP server. Start locally with the lightweight EMT calculator, then opt into
larger models, external simulation programs, or distributed execution when a
workflow needs them.

!!! warning "Review generated calculations"
    ChemGraph can launch calculations and write files. Check generated inputs,
    calculator settings, convergence, units, and scientific conclusions before
    relying on a result.

## First run

Install the CLI with [uv](https://docs.astral.sh/uv/guides/tools/), which manages
an isolated environment (Python 3.11 or newer required):

```bash
uv tool install chemgraph
```

To import ChemGraph from Python or a notebook, install it in a virtual
environment instead; see [Installation](installation.md).

Set one model-provider credential, then check the installation and run a query:

```bash
export OPENAI_API_KEY="..."
chemgraph run --check-keys
chemgraph run -q "What is the SMILES string for aspirin?"
```

The [quickstart](quickstart.md) adds a first EMT calculation and explains where
ChemGraph writes artifacts. Argonne users can follow the same guide with an
Argo or ALCF-hosted model.

## Find the right guide

| Goal | Guide |
| --- | --- |
| Install core or optional dependencies | [Installation](installation.md) |
| Configure a model provider | [Models and authentication](models.md) |
| Learn CLI commands and saved sessions | [Command-line interface](cli.md) |
| Select an agent architecture | [Workflows](workflows.md) |
| Choose a chemistry calculator | [Calculators](calculators.md) |
| Embed ChemGraph in Python | [Python API](python_api.md) |
| Use a browser interface | [Streamlit interface](streamlit_web_interface.md) |
| Connect MCP clients and servers | [MCP servers](mcp_servers.md) |
| Run with containers | [Docker](docker_support.md) |
| Deploy Streamlit and MCP to a cluster | [Kubernetes](kubernetes.md) |
| Scale across execution backends | [HPC and Academy](hpc_and_academy.md) |
| Evaluate models with structured ground truth | [Evaluation](evaluation.md) and the [ChemGraph Leaderboard](https://huggingface.co/spaces/Autonomous-Scientific-Agents/chemgraph-leaderboard) |
| Diagnose a failed first run | [Troubleshooting](troubleshooting.md) |

## Support levels

The single-agent CLI/Python path, core ASE tools, EMT, MACE, and the general MCP
server are the normal starting points. Optional integrations require their
documented extras or external programs. Codex subscription support requires its
optional SDK and a ChatGPT login; see [Codex subscription](codex_subscription.md).
Site-specific HPC servers, gRASPA workflows, docking, XANES, and some
distributed backends are advanced or experimental; validate them in your own
environment before production use.

## News and research

Read [News](news.md) for releases, presentations, and
media coverage. Citations and BibTeX for the framework, HPC orchestration,
ChemGraph-XANES, and ChemGraph-IR papers are on
[Papers and citations](citation.md).

## Community

- [Examples](example_usage.md)
- [Contributing](contributing.md)
- [News](news.md)
- [Papers and citations](citation.md)
- [GitHub issues](https://github.com/argonne-lcf/ChemGraph/issues)
