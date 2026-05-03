# Evaluation & Benchmarking

ChemGraph includes a built-in evaluation module (`chemgraph.eval`) for benchmarking LLM tool-calling accuracy across multiple models and workflows. The module uses an **LLM-as-judge** strategy where a separate judge LLM compares the agent's tool-call sequence and final answer against ground-truth results using binary scoring (1 = correct, 0 = wrong).

## Overview

The evaluation pipeline works as follows:

1. **Load dataset** -- A ground-truth JSON file containing queries, expected tool-call sequences, and actual results.
2. **Run agent** -- For each `(model, workflow, query)` combination, initialize a `ChemGraph` agent, execute the query, and capture tool calls and the final answer.
3. **Judge** -- A separate judge LLM compares the agent's output against the ground truth and assigns a binary score.
4. **Report** -- Aggregate scores are written as JSON, Markdown, and console reports.

```
Dataset (14 queries)
    │
    ▼
┌──────────────────┐     ┌──────────────┐     ┌───────────┐
│  ChemGraph Agent │ ──▶ │  LLM Judge   │ ──▶ │  Reports  │
│  (model under    │     │  (separate   │     │  (JSON,   │
│   test)          │     │   model)     │     │   MD,     │
└──────────────────┘     └──────────────┘     │   console)│
                                              └───────────┘
```

## Bundled Dataset

A default dataset of **14 queries** across 4 categories is shipped with the package at `src/chemgraph/eval/data/ground_truth.json` and used automatically when no explicit dataset is provided.

### Categories

| Category | IDs | Description | Tool Chain |
|----------|-----|-------------|------------|
| **A** Single tool calls | 1--4 | Name-to-SMILES, SMILES-to-coordinates (1 or 2 molecules) | `molecule_name_to_smiles` or `smiles_to_coordinate_file` |
| **B** Multi-step from name | 5--9 | Full pipeline from molecule name to ASE simulation | `molecule_name_to_smiles` → `smiles_to_coordinate_file` → `run_ase` |
| **C** Multi-step from SMILES | 10--11 | Pipeline from SMILES string to ASE simulation | `smiles_to_coordinate_file` → `run_ase` |
| **D** Reaction Gibbs energy | 12--14 | Multi-species thermochemistry with stoichiometric calculation | `molecule_name_to_smiles` → `smiles_to_coordinate_file` → `run_ase` (per species) → `calculator` |

## Running Evaluations

### CLI

The evaluation module provides a standalone CLI command (`chemgraph-eval`) as well as a subcommand (`chemgraph eval`).

#### Minimal Invocation

```bash
# Uses the bundled 14-query dataset, single_agent workflow
chemgraph-eval --models gpt-4o-mini --judge-model gpt-4o
```

#### Multiple Models

```bash
chemgraph-eval \
    --models gpt-4o-mini gemini-2.5-flash claude-3-5-haiku-20241022 \
    --judge-model gpt-4o
```

#### With TOML Config

When a `config.toml` is provided, the evaluation module resolves `base_url` and `argo_user` for each model from the `[api.*]` sections, matching the behaviour of the main CLI.

```bash
chemgraph-eval --models gpt-4o-mini --judge-model gpt-4o --config config.toml
```

#### Profile-Based

Profiles are defined under `[eval.profiles.*]` in `config.toml` and provide reusable configurations:

```bash
chemgraph-eval --profile quick --models gpt-4o-mini --judge-model gpt-4o --config config.toml
```

#### Custom Dataset & Limits

```bash
chemgraph-eval \
    --models gpt-4o-mini \
    --judge-model gpt-4o \
    --dataset path/to/custom_ground_truth.json \
    --workflows single_agent \
    --max-queries 5 \
    --output-dir eval_results
```

### Python API

```python
import asyncio
from chemgraph.eval import ModelBenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(
    models=["gpt-4o-mini", "gemini-2.5-flash"],
    judge_model="gpt-4o",
    # dataset defaults to bundled 14-query dataset
    # workflow_types defaults to ["single_agent"]
)
runner = ModelBenchmarkRunner(config)
results = asyncio.run(runner.run_all())
runner.report()  # generates JSON + Markdown + console output
```

You can also control report format:

```python
runner.report(format="json")      # JSON only
runner.report(format="markdown")  # Markdown only
runner.report(format="console")   # Console table only
runner.report(format="all")       # All formats (default)
```

## CLI Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--models` | LLM model names to evaluate (required, space-separated) | — |
| `--judge-model` | LLM model name for the judge (required) | — |
| `--profile` | Eval profile name from `[eval.profiles.*]` in config.toml | None |
| `--dataset` | Path to ground-truth JSON file | Bundled dataset |
| `--workflows` | Workflow types to test (space-separated) | `single_agent` |
| `--output-dir` | Output directory for results | `eval_results` |
| `--max-queries` | Max queries to evaluate (0 = all) | 0 |
| `--recursion-limit` | Max LangGraph recursion steps per query | 50 |
| `--config` | Path to TOML config file | None |
| `--tags` | Free-form tags for run metadata (space-separated) | — |
| `--no-structured-output` | Disable structured output on the agent | — |
| `--report` | Report format: `json`, `markdown`, `console`, `all` | `all` |

**Valid workflow types**: `single_agent`, `multi_agent`, `single_agent_mcp`

## Configuration

### BenchmarkConfig

The `BenchmarkConfig` Pydantic model holds all settings for a benchmark run:

```python
from chemgraph.eval import BenchmarkConfig

config = BenchmarkConfig(
    models=["gpt-4o-mini"],           # Required: models to evaluate
    judge_model="gpt-4o",             # Required: judge model
    workflow_types=["single_agent"],   # Default: ["single_agent"]
    dataset="path/to/gt.json",        # Default: bundled dataset
    output_dir="eval_results",        # Default: "eval_results"
    structured_output=True,           # Default: True
    recursion_limit=50,               # Default: 50
    max_queries=0,                    # Default: 0 (all queries)
    config_file="config.toml",        # Default: None
)
```

### TOML Profiles

Define reusable profiles in your `config.toml`:

```toml
[eval]
default_profile = "standard"

[eval.profiles.standard]
judge_model = "gpt-4o"
workflow_types = ["single_agent", "multi_agent"]
recursion_limit = 50
```

Profiles are loaded via `BenchmarkConfig.from_profile()` or the `--profile` CLI flag. CLI arguments always override profile values.

When `--config` is provided without `--profile`, the `[eval] default_profile` is used automatically if defined.

List available profiles:

```python
from chemgraph.eval import BenchmarkConfig
profiles = BenchmarkConfig.list_profiles("config.toml")
```

## LLM Judge

The judge is implemented in `chemgraph.eval.llm_judge` and uses the following evaluation rubric:

### Scoring Rules

- **Binary scoring**: 1 = correct, 0 = wrong
- **Numeric tolerance**: Values must match within **5% relative tolerance**
- **Minor formatting**: Extra explanation, rounding, or formatting differences are acceptable
- **File paths**: Minor path/name differences are acceptable if the expected output is produced
- **Tool calls**: Missing tool calls are acceptable if the final answer is correct and the dependency chain is preserved
- **Key arguments must match**: calculator type, driver, SMILES strings, molecule names, temperature, method
- **Optional parameters**: Differences in default/optional parameter values are acceptable
- **Final verdict**: Correct (1) only if **both** the tool-call sequence and final result are substantially correct

### Using a Different Judge

The judge model should ideally be a capable model (e.g., `gpt-4o`) that is different from the model under test to avoid self-evaluation bias:

```bash
# Evaluate gpt-4o-mini, judged by gpt-4o
chemgraph-eval --models gpt-4o-mini --judge-model gpt-4o
```

## Ground-Truth Generation

The ground-truth dataset is generated by the script `scripts/evaluations/generate_ground_truth.py`, which programmatically builds and executes tool-call chains for each query category.

### Input Format

The input file (`input_data.json`) contains molecules and reactions:

```json
{
    "molecules": [
        {
            "name": "water",
            "number_of_atoms": 3,
            "smiles": "O"
        }
    ],
    "reactions": [
        {
            "reaction_name": "Methane Combustion",
            "reactants": [
                {"name": "Methane", "smiles": "C", "coefficient": 1},
                {"name": "Oxygen", "smiles": "O=O", "coefficient": 2}
            ],
            "products": [
                {"name": "Carbon dioxide", "smiles": "O=C=O", "coefficient": 1},
                {"name": "Water", "smiles": "O", "coefficient": 2}
            ]
        }
    ]
}
```

### Running the Generator

```bash
cd scripts/evaluations

# Full execution (runs all tool chains end-to-end, captures results)
python generate_ground_truth.py --input_file input_data.json

# Skip execution (produces entries with empty results -- faster for testing)
python generate_ground_truth.py --input_file input_data.json --skip_execution

# Custom output path
python generate_ground_truth.py --input_file input_data.json -o my_ground_truth.json
```

### Output Format

Each entry in the generated `ground_truth.json` has this structure:

```json
{
    "id": "5",
    "query": "Calculate the geometry optimization of sulfur dioxide using mace_mp",
    "answer": {
        "tool_calls": [
            {"molecule_name_to_smiles": {"name": "sulfur dioxide"}},
            {"smiles_to_coordinate_file": {"smiles": "O=S=O"}},
            {"run_ase": {"input_structure_file": "...", "calculator_type": "mace_mp", "driver": "opt"}}
        ],
        "result": {
            "energy": -14.523,
            "positions": [[...], ...],
            "...": "..."
        }
    }
}
```

### Custom Datasets

You can create your own ground-truth dataset by following either of two supported JSON formats:

**List format** (recommended):

```json
[
    {
        "id": "1",
        "query": "Your natural language query",
        "answer": {
            "tool_calls": [...],
            "result": {...}
        }
    }
]
```

**Legacy dict format** (also supported):

```json
{
    "molecule_name": {
        "query": "Your query",
        "answer": {...}
    }
}
```

Both formats are auto-detected by `load_dataset()`.

## Output & Reports

Evaluation runs produce output in the `eval_results/` directory (configurable via `--output-dir`):

### JSON Report

`benchmark_<timestamp>.json` -- Machine-readable aggregate results:

- Run metadata (timestamp, models, workflows, tags)
- Per-model, per-workflow accuracy scores
- Per-query judge scores and reasoning

### Markdown Report

`benchmark_<timestamp>.md` -- Human-readable summary with accuracy tables:

```
| Model          | Workflow     | Queries | Correct | Accuracy | Parse Errors |
|----------------|-------------|---------|---------|----------|--------------|
| gpt-4o-mini    | single_agent | 14      | 11      | 78.6%    | 0            |
| gemini-2.5-flash | single_agent | 14    | 12      | 85.7%    | 1            |
```

### Per-Model Detail Files

`<model>_<workflow>_detail.json` -- Full detail for each query including the agent's tool calls, final answer, judge score, and judge reasoning.

### Console Summary

A Rich-formatted table printed to the console during the run showing real-time accuracy per model and workflow.

## Testing

The evaluation module has a comprehensive test suite:

```bash
# Run all eval tests
pytest tests/test_eval.py -v

# Run specific test classes
pytest tests/test_eval.py::TestBenchmarkConfig -v
pytest tests/test_eval.py::TestLLMJudge -v
pytest tests/test_eval.py::TestCLI -v
```

## Module Structure

```
src/chemgraph/eval/
├── __init__.py          # Public API exports
├── cli.py               # CLI entry point (chemgraph-eval command)
├── config.py            # BenchmarkConfig (Pydantic model)
├── datasets.py          # Dataset loading & GroundTruthItem schema
├── llm_judge.py         # LLM-as-judge evaluator (binary scoring)
├── reporter.py          # JSON/Markdown/console report generators
├── runner.py            # ModelBenchmarkRunner orchestration
└── data/
    └── ground_truth.json  # Bundled default dataset (14 queries)
```
