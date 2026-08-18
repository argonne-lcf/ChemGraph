# Evaluation

ChemGraph's primary evaluation pipeline uses the **deterministic structured-output
judge**. It asks each agent for a `ResponseFormatter` result, compares that result
field-by-field with structured ground truth, and produces reproducible binary
scores without a second judge model.

View published results on the
[ChemGraph Leaderboard](https://huggingface.co/spaces/Autonomous-Scientific-Agents/chemgraph-leaderboard).

The LLM-as-judge path remains available for qualitative or legacy comparisons,
but it is secondary because it adds cost, nondeterminism, and judge-model bias.

No default benchmark dataset is bundled. Supply `--dataset` or define `dataset`
in a selected TOML evaluation profile.

## Main pipeline

```text
Ground-truth structured_output
              ↓
Agent ResponseFormatter JSON
              ↓
Deterministic field comparators
              ↓
Per-query score: all expected fields pass = 1, otherwise 0
              ↓
Accuracy and per-field diagnostics
```

For every model, workflow, and query, the runner:

1. starts ChemGraph with structured output enabled;
2. captures the agent's tool calls, final result, and `ResponseFormatter` JSON;
3. compares each non-null expected field with the matching actual field;
4. records a binary score, per-field pass/fail values, and a rationale;
5. aggregates correct queries into structured-output accuracy.

Formatter/JSON parse failures score as incorrect and remain in the accuracy
denominator. Ground truth with no non-null structured fields is also invalid and
scores as a failure instead of receiving a trivial pass.

## Quickstart

Run the structured judge explicitly. A judge model is not needed:

```bash
chemgraph eval \
  --models gpt-4o-mini \
  --dataset evaluation/questions.json \
  --workflows single_agent \
  --judge-type structured \
  --output-dir eval_results
```

Always include `--judge-type structured` unless a selected profile already sets
it. The CLI's compatibility fallback is still `llm` when no judge type is
specified.

## Structured ground truth

The preferred dataset format is a JSON array. Each item needs a unique ID,
query, and `answer.structured_output` object:

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
      "result": "O",
      "structured_output": {
        "smiles": ["O"]
      }
    }
  }
]
```

Only non-null fields in the expected `structured_output` are judged, so include
every result modality that the query is required to return.

| Structured field | Expected shape | Comparison |
| --- | --- | --- |
| `smiles` | List of SMILES strings | RDKit canonicalization; expected values matched independently of order |
| `scalar_answer` | `{value, property, unit}` | Value within 5% relative tolerance; normalized property category; unit match |
| `dipole` | `{value: [dx, dy, dz], unit}` | Components within 5% relative tolerance; normalized unit match |
| `vibrational_answer` | `{frequency_cm1: [...]}` | Imaginary modes removed; sorted real modes compared within 5% |
| `ir_spectrum` | `{frequency_cm1: [...], intensity: [...]}` | Frequencies and intensities compared within 5% |
| `atoms_data` | `{numbers: [...], positions: [...]}` | Atomic numbers exact; each coordinate within 0.1 Å |

The default numeric tolerances come from `judge_structured_output()`. The CLI
runner currently uses those defaults. The loader also accepts the legacy object
format containing `manual_workflow` or `llm_workflow`, but new datasets should
use the list format above.

Keep ground truth under version control when licensing permits, document how it
was produced, and review it independently of every model being measured.

## Profiles

Make structured judging the explicit default in a reusable profile:

```toml
[eval]
default_profile = "standard"

[eval.profiles.standard]
dataset = "evaluation/questions.json"
workflow_types = ["single_agent"]
judge_type = "structured"
structured_output = true
recursion_limit = 50
max_queries = 0
```

Then run:

```bash
chemgraph eval \
  --config config.toml \
  --profile standard \
  --models gpt-4o-mini
```

`--profile` requires `--config`. CLI values override matching profile values.
If `[eval] default_profile` exists, providing `--config` without `--profile`
selects it automatically.

## Reports and diagnostics

Select `--report json`, `markdown`, `console`, or `all` (the default). Structured
results include:

- `structured_judge_aggregate`: query count, correct count, accuracy, and parse
  errors for each model/workflow pair;
- `structured_judge_details`: per-query score, `field_scores`, rationale, query
  metadata, and parse error;
- raw tool calls and final agent results for debugging failures.

Use the per-field rationale to distinguish a wrong scientific value from a
missing field, unit/property mismatch, geometry difference, or formatter parse
failure.

## Resume interrupted runs

The runner writes per-query checkpoints even when resume is disabled. Continue
with the same dataset, model/workflow configuration, and output directory:

```bash
chemgraph eval \
  --models gpt-4o-mini \
  --dataset evaluation/questions.json \
  --judge-type structured \
  --output-dir eval_results \
  --resume
```

Use `--max-queries` for a smoke test, `--recursion-limit` to bound graph
execution, and `--tags` for run metadata. Evaluation currently accepts
`single_agent` and `multi_agent` workflows.

## Python API

```python
import asyncio

from chemgraph.eval import BenchmarkConfig, ModelBenchmarkRunner


config = BenchmarkConfig(
    models=["gpt-4o-mini"],
    dataset="evaluation/questions.json",
    workflow_types=["single_agent"],
    structured_output=True,
    judge_type="structured",
    output_dir="eval_results",
)
runner = ModelBenchmarkRunner(config)
asyncio.run(runner.run_all())
runner.report(format="all")
```

## Optional LLM judge

Use the LLM judge only when a deterministic structured comparison does not
capture the evaluation goal or when reproducing a legacy benchmark:

```bash
chemgraph eval \
  --models gpt-4o-mini \
  --dataset evaluation/questions.json \
  --judge-type llm \
  --judge-model gpt-4o \
  --output-dir eval_results
```

Use `--judge-type both` to record deterministic and LLM-judge results side by
side. Report the judge model and settings, and do not substitute the LLM score
for the structured score on leaderboard-style comparisons.

Evaluation runs invoke real models and tools unless your test setup replaces
them. Estimate cost, isolate credentials, and begin with a small dataset.
