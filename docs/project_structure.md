```
chemgraph/
│
├── src/                       # Source code
│   ├── chemgraph/             # Top-level package
│   │   ├── agent/             # Agent-based task management
│   │   ├── eval/              # Evaluation & benchmarking (LLM-as-judge)
│   │   ├── graphs/            # Workflow graph utilities
│   │   ├── mcp/               # MCP servers (stdio/streamable HTTP)
│   │   ├── memory/            # Session memory (SQLite-backed persistence)
│   │   ├── models/            # LLM provider integrations
│   │   ├── prompt/            # Agent prompt templates
│   │   ├── schemas/           # Pydantic data models
│   │   ├── state/             # Agent state definitions
│   │   ├── tools/             # Tools for molecular simulations
│   │   ├── utils/             # Other utility functions
│   ├── ui/                    # CLI and Streamlit UI package
│
├── scripts/                   # Utility & evaluation scripts
│   ├── new_evaluation/        # Ground-truth dataset generation
├── docs/                      # MkDocs documentation
├── pyproject.toml             # Project configuration
└── README.md                  # Project documentation
```
