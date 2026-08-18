# Project structure

```text
ChemGraph/
├── src/
│   ├── chemgraph/
│   │   ├── agent/        # ChemGraph agent, durable sessions, run events
│   │   ├── graphs/       # Single-, multi-, and specialized LangGraph graphs
│   │   ├── tools/        # Chemistry, ASE, analysis, and file tools
│   │   ├── mcp/          # General and HPC MCP servers
│   │   ├── execution/    # Local, Parsl, Ensemble, Globus backends
│   │   ├── academy/      # Persistent multi-agent campaigns and dashboard
│   │   ├── models/       # Model registry, provider loaders, normalization
│   │   ├── schemas/      # Pydantic input/output models
│   │   ├── eval/         # Dataset-driven evaluation and judges
│   │   ├── cli/          # Command-line parser and commands
│   │   ├── memory/       # SQLite sessions and checkpoints
│   │   ├── prompt/       # Agent prompts
│   │   ├── hpc_configs/  # Facility Parsl configurations
│   │   └── utils/        # Shared utilities
│   └── ui/               # Streamlit application
├── tests/                 # Pytest suite
├── docs/                  # MkDocs site
├── examples/              # Specialized runnable examples
├── scripts/               # MCP, smoke, demo, and helper scripts
├── notebooks/             # Interactive examples
├── config.toml            # Example configuration
├── pyproject.toml         # Package metadata and tooling
└── mkdocs.yml             # Documentation navigation/theme
```

The installable package is `chemgraph`; the `src/ui` package contains the
source-checkout Streamlit entry point. Package version and optional extras are
single-sourced in `pyproject.toml`.

For implementation conventions and validation commands, see
[Contributing](contributing.md).
