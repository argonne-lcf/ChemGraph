# Evaluation

`chemgraph.eval` compares models and workflows against an explicit ground-truth
JSON dataset. It supports an LLM judge, deterministic structured-output
comparison, or both.

No default benchmark dataset is bundled. Supply `--dataset` or define `dataset`
in a selected TOML evaluation profile.

## Dataset format

The preferred format is a JSON array. Each item needs a unique ID, query, and
answer object:

```json
[
  {
    "id": "water-smiles",
    "category": "lookup",
    "query": "What is the SMILES string for water?",
    "answer": {
      "tool_calls": [
        {
          "name": "molecule_name_to_smiles",
          "args": {"name": "water"}
        }
      ],
      "result": "O"
    }
  }
]
```

For `--judge-type structured` or `both`, add `structured_output` under `answer`
using the same response fields the agent is expected to produce. The loader also
accepts a legacy object format whose entries contain `manual_workflow` or
`llm_workflow`, but new datasets should use the list format.

Keep ground truth under version control when licensing permits, document how it
was produced, and review it independently of the model being measured.

## Deterministic structured judge

This mode needs no separate judge model:

```bash
chemgraph eval \
  --models gpt-4o-mini \
  --dataset evaluation/questions.json \
  --workflows single_agent \
  --judge-type structured \
  --output-dir eval_results
```

Dataset entries without `structured_output` cannot receive a structured field
comparison.

## LLM-as-judge

The LLM judge compares expected and observed tool calls/results. Choose a judge
model separately from the model under test when practical:

```bash
chemgraph eval \
  --models gpt-4o-mini gemini-2.5-flash \
  --dataset evaluation/questions.json \
  --workflows single_agent multi_agent \
  --judge-type llm \
  --judge-model gpt-4o \
  --output-dir eval_results
```

Use `--judge-type both` to run both strategies. LLM judging introduces model
cost, nondeterminism, and potential bias; report judge identity and settings.

## Profiles

Put reusable settings in `config.toml`:

```toml
[eval]
default_profile = "quick"

[eval.profiles.quick]
dataset = "evaluation/questions.json"
workflow_types = ["single_agent"]
judge_type = "structured"
structured_output = true
recursion_limit = 50
max_queries = 5
```

Then run:

```bash
chemgraph eval \
  --config config.toml \
  --profile quick \
  --models gpt-4o-mini
```

`--profile` requires `--config`. CLI values override the corresponding profile
values. If the config defines `[eval] default_profile`, providing `--config`
without `--profile` selects it automatically.

## Resume and reports

The runner writes per-query checkpoints even when resume is disabled. Continue
an interrupted run with the same output directory:

```bash
chemgraph eval \
  --models gpt-4o-mini \
  --dataset evaluation/questions.json \
  --judge-type structured \
  --output-dir eval_results \
  --resume
```

Select `--report json`, `markdown`, `console`, or `all` (the default). Use
`--max-queries` for a smoke test, `--recursion-limit` to bound graph execution,
and `--tags` for run metadata. Only `single_agent` and `multi_agent` are accepted
as evaluation workflows.

## Python API

```python
import asyncio

from chemgraph.eval import BenchmarkConfig, ModelBenchmarkRunner


config = BenchmarkConfig(
    models=["gpt-4o-mini"],
    dataset="evaluation/questions.json",
    workflow_types=["single_agent"],
    judge_type="structured",
    output_dir="eval_results",
)
runner = ModelBenchmarkRunner(config)
asyncio.run(runner.run_all())
runner.report(format="all")
```

Evaluation runs invoke real model/tool workflows unless your test setup replaces
them. Estimate cost, isolate credentials, and begin with a small dataset.
