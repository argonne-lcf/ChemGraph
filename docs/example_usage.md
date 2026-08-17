# Examples

Start with the [quickstart](quickstart.md), then choose an example closest to
your intended deployment. Examples that use live models, external scientific
programs, or HPC services require their own credentials and site setup.

## Small local tasks

```bash
# Lookup
chemgraph run -q "What is the SMILES string for aspirin?"

# Geometry optimization with the lightweight bundled calculator
chemgraph run -q "Build water from SMILES O and optimize it with EMT."

# Frequencies
chemgraph run -q "Calculate water vibrational frequencies with EMT."

# Save only the final response
chemgraph run --output last_message --output-file result.txt \
  -q "Build methane and report its formula."
```

Review calculator suitability before interpreting the result. EMT examples are
setup checks, not general high-accuracy chemistry recommendations.

## Interactive and Python examples

```bash
chemgraph run --interactive
```

For application code, begin with the async example in [Python API](python_api.md).

Repository notebooks and example directories cover richer use cases:

- [`notebooks/`](https://github.com/argonne-lcf/ChemGraph/tree/main/notebooks)
- [`examples/`](https://github.com/argonne-lcf/ChemGraph/tree/main/examples)
- [`scripts/`](https://github.com/argonne-lcf/ChemGraph/tree/main/scripts)

## MCP examples

- [General stdio and HTTP examples](https://github.com/argonne-lcf/ChemGraph/tree/main/scripts/mcp_example)
- [Parsl MCP example](https://github.com/argonne-lcf/ChemGraph/tree/main/scripts/mcp_parsl_example)
- [XANES MCP examples](https://github.com/argonne-lcf/ChemGraph/tree/main/examples/xanes_mcp)
- [OpenCode client example](https://github.com/argonne-lcf/ChemGraph/tree/main/examples/chemgraph_opencode)

Read [MCP servers](mcp_servers.md) first to choose stdio or streamable HTTP.

## Docking and XANES

- [Docking example](https://github.com/argonne-lcf/ChemGraph/tree/main/examples/docking)
- [XANES examples](https://github.com/argonne-lcf/ChemGraph/tree/main/examples/xanes_mcp)

These are specialized workflows. Docking needs Meeko and Vina; XANES may need
Materials Project access and/or FDMNES.

## Distributed execution and Academy

- [Execution backend demos](https://github.com/argonne-lcf/ChemGraph/tree/main/scripts/demo)
- [Academy MACE ensemble screening](https://github.com/argonne-lcf/ChemGraph/tree/main/examples/academy/example-002-mace-ensemble-screening)
- [Connect to Argo from an ALCF compute node](https://github.com/argonne-lcf/ChemGraph/tree/main/examples/connecting_to_argo)

These examples deliberately separate direct backend calls from agent-driven
calls, which is useful for diagnosing infrastructure before adding an LLM.

## Evaluation

Evaluation is dataset-driven rather than bundled with a default benchmark.
Follow [Evaluation](evaluation.md) to create the supported JSON schema, select a
judge, resume interrupted runs, and compare workflows.
