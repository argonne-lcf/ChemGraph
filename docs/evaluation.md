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
recursion_limit = 200
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

The default recursion limit is 200 graph steps per query, increased from 50.
A looping query can therefore run for four times as many steps before stopping,
which can increase runtime and model/tool costs. Steps include middleware and
tool execution; they are not a token or monetary budget, so cost does not scale
by a fixed factor. Use `--recursion-limit 50` to retain the previous budget, or
set `recursion_limit = 50` in an evaluation profile. Limits must be at least 1.

## Deep Agent evaluation

The evaluator runs the existing `deep_agent` workflow with bundled skills,
chemistry tool discovery, and one output directory per model/thread. File tools,
shell commands, chemistry tools, and logs share real host paths under
`<output-dir>/logs/<model>/deep_agent/thread_<query-id>/` (model `/` and `:` become
`_`; query IDs are URL-escaped). This directory is also the shell working
directory and `CHEMGRAPH_LOG_DIR`; there is no `/workspace` alias. The agent is
told to keep generated inputs, scripts, and results here.
On Windows, built-in file tools use forward-slash paths without the drive prefix
(for example, `/Users/name/results/file.xyz`); the backend supplies the workspace
drive. Chemistry tools continue to use native host paths. The evaluation prompt
provides both directory forms.

`--output-dir` is sufficient; `--deepagent-workspace` optionally replaces the
`<output-dir>/logs` root. Queries require empty thread directories. Before loading
models or changing checkpoints and reports, evaluation rejects any selected query
whose directory is nonempty, unless a compatible checkpoint already records that
query and `--resume` will reuse it. This also rejects artifacts from interrupted
queries that have no checkpoint. Existing files are preserved. Use a new output
directory for a fresh benchmark, and a new `--deepagent-workspace` root if that
override is configured. A second check immediately before each query rejects
files that appeared after preflight as an initialization error.
Persistent memory and personal/project skill discovery are disabled. These
directories separate artifacts; they are **not operating-system sandboxes**.
Deep Agent evaluation requires `--deepagent-auto-approve`, or
`deepagent_auto_approve = true` in the evaluation profile
(`deepagent_auto_approve=True` in Python). Evaluation has no interactive approval
handler, so missing approval is rejected before any models are initialized,
including when Deep Agent is selected alongside other workflows. Use this option
only in a trusted execution environment: it allows unattended tool execution,
file changes, and shell commands.

```bash
chemgraph eval \
  --config config.toml \
  --models argo:gpt-4.1-mini \
  --dataset /path/to/groundtruth.json \
  --workflows deep_agent \
  --judge-type structured \
  --deepagent-auto-approve \
  --query-ids 3 5 6 7 8 9 31 \
  --output-dir eval_results/deepagent-smoke
```

For a proxy, add `--base-url http://127.0.0.1:20219/v1` to the same command.
It overrides provider URLs from `config.toml` for both execution and formatting
(and the optional LLM judge). Profiles can also set `base_url`. When splitting
a shell command across lines, each continuation backslash must be the last
character on its line.

Download `output_data.json` from the
[ground-truth dataset](https://huggingface.co/datasets/Autonomous-Scientific-Agents/groundtruth)
at a fixed revision, and keep it outside the agent workspaces. The initial
40-query benchmark uses revision `8fe726352eb0437e4958282ca741d064fcfe6959`.
Omit `--query-ids` for the full dataset. IDs are selected in dataset order before
`--max-queries` is applied. These options also work as profile keys:
`deepagent_workspace`, `deepagent_auto_approve`, and `query_ids`.

After a completed run, evaluation uses the same model and provider settings,
including the agent's resolved reasoning effort,
to format the recorded messages into `ResponseFormatter` JSON, with the existing
formatter prompt and one retry. This formatting call has no tools or ground
truth. It does not change the Deep Agent graph. The original answer and state
are preserved separately from `structured_output` and `formatter_responses`.
Formatter replies are retained even if a later retry raises an exception.
The deterministic judge and its tolerances are unchanged, including the
orientation-sensitive dipole component comparison.

Each raw query record includes status, errors, workspace, elapsed seconds, and
separate execution, formatting, and combined usage. Usage includes retries;
unreported tokens remain unknown. Initialization and execution failures count as
incorrect. Formatting failures, including invalid JSON and formatter model
errors, count as incorrect for the structured judge. The LLM judge still scores
the completed original answer and tool calls independently. Formatting failures
retain their error status and diagnostics in raw records and remain included in
the error counts in JSON and Markdown execution summaries.

Dataset and code checksums, dependency versions, and settings identify the run.
Deep Agent resume requires a matching fingerprint, including the dataset,
`query_ids`, `max_queries`, judge and execution settings, configured model/judge
endpoints, recorded dependency versions, and hashed package source (`.py`, `.md`,
and `.template` files).
Provider environment variables are not fingerprinted; resume assumes the
provider environment remains unchanged.
Changing query-selection options invalidates checkpoints even if they select the
same queries; raising `--max-queries` to extend a smoke test also requires a new
output directory. Package source changes, including evaluator fixes, invalidate
old checkpoints. There is no checkpoint migration. Use separate output
directories for smoke tests and full runs; `--resume` reuses completed query
records, including failures, when the fingerprint matches.

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
`single_agent`, `multi_agent`, and `deep_agent` workflows.

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

## Optional isolated Podman evaluation

The `podman` runtime runs each DeepAgent query, its tools, shell commands, and
answer formatter in a fresh container. The controller alone reads the dataset,
scores answers, and writes checkpoints. Local evaluation remains the default.
This first version requires **native Linux x86-64, rootless Podman with Netavark,
and cgroup v2 with delegated CPU, memory, and process controllers**. It supports
CPU chemistry and hosted Argo endpoints only. It fails at startup if its
isolation probes fail; it never falls back to local execution.

Install Podman, Netavark/Aardvark DNS, the rootless networking helpers, and
`newuidmap`/`newgidmap` using your distribution's supported packages. An
administrator must configure subordinate UID/GID ranges and cgroup delegation
when these are absent. Verify `podman info --format=json`: `host.security.rootless`
must be true, `host.networkBackend` must be `netavark`, and
`host.cgroupControllers` must contain `cpu`, `memory`, and `pids`. Do not run the
benchmark with `sudo`. Older distribution Podman packages may need upgrading.

### Install the host prerequisites

For a fresh **Ubuntu 24.04 x86-64** host, an administrator can install the
container tools and basic command-line utilities with:

```bash
sudo apt-get update
sudo apt-get install -y podman netavark aardvark-dns uidmap \
  slirp4netns fuse-overlayfs crun jq curl openssl tmux
```

Ubuntu 22.04's standard repository provides Podman 3.4, which predates this
Netavark configuration. On that release, have an administrator provide a newer
supported Podman/Netavark/Aardvark stack, or use a suitable newer Linux host.
Do not mix packages from different Ubuntu releases. See the
[Podman installation guide](https://podman.io/docs/installation) and Ubuntu's
[22.04](https://packages.ubuntu.com/jammy/admin/podman) and
[24.04](https://packages.ubuntu.com/noble/admin/podman) package versions.
The startup probes, rather than a version number alone, determine compatibility.

Run the remaining commands as your ordinary login user:

```bash
# Both files should contain a subordinate-ID allocation for your username.
awk -F: -v user="$(id -un)" '$1 == user { print }' /etc/subuid /etc/subgid

# Required: true, "netavark", "v2", and controllers including cpu/memory/pids.
podman info --format=json | jq '.host | {
  rootless: .security.rootless,
  networkBackend,
  cgroupVersion,
  cgroupControllers
}'
```

If the ID allocations or resource controllers are missing, ask your system
administrator to configure them using the
[rootless setup guide](https://github.com/containers/podman/blob/main/docs/tutorials/rootless_tutorial.md),
then log out and back in. Existing Podman installations using CNI need a planned
network-backend migration; do not reset existing container storage to run this
benchmark. On a fresh installation, Netavark is the expected backend.

Use a local filesystem for rootless container storage. The host needs room for
Python/CPU PyTorch images, downloaded weights, and collected artifacts, plus
memory beyond the worker's default 8 GiB limit for the controller and proxy.
Hosted Argo must be reachable from this host (including any required Argonne
network/VPN connection).

Set up the host controller from the checked-out PR branch using `uv`:

```bash
# In a fresh checkout; reuse an existing suitable virtualenv if one is present.
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .
uv pip check
```

### Prepare images

From this checkout, using the normal editable ChemGraph environment:

```bash
python evaluation/build.py \
  --worker-image localhost/chemgraph-eval:latest \
  --proxy-image localhost/chemgraph-eval-proxy:latest
export PODMAN_IMAGE=localhost/chemgraph-eval:latest
export PODMAN_PROXY_IMAGE=localhost/chemgraph-eval-proxy:latest
export ARGO_USER=your-argo-identity
```

Preparation needs internet access. The builder creates a temporary context
from explicitly selected package sources and metadata, installs a wheel using
`evaluation/requirements.lock`, and preloads `medium-mpa-0`. It performs real
CPU MACE and TBLite water calculations and records dependency versions, weight
SHA-256 digests, and energies in `/opt/chemgraph-evaluation-manifest.json`.
Datasets, tests, personal configuration, archives, and historical outputs never
enter either build stage. `python evaluation/lock.py` intentionally refreshes
the Linux/Python 3.11 dependency lock with `uv`. Base OS images and apt packages
are resolved at preparation time; record and retain the resulting local images.
No registry images are published by this workflow.

Every run resolves the prepared worker and proxy tags to immutable local image
IDs, uses `--pull=never`, and records both IDs and the manifest in its metadata.
The runtime applies a read-only root filesystem, UID 1000, dropped capabilities,
`no-new-privileges`, default seccomp, and a fixed process limit of 256. Each query
gets its own private named volume at `/workspace` and private temporary storage.
No host paths or container sockets are mounted. Persistent agent memory and
personal/project skill discovery remain disabled.

Workers join only a run-specific internal bridge. Squid joins that bridge and
a separate outbound bridge; neither publishes host ports. It permits HTTPS
CONNECT only to **apps.inside.anl.gov:443** and
**pubchem.ncbi.nlm.nih.gov:443**, without caching. TLS verification remains on.
Explicit proxy settings reach both Python tools and shell subprocesses, and
host proxy variables are not inherited. Startup probes check cgroup limits,
capabilities, filesystem permissions, direct routing, external DNS rejection,
proxy denials, and verified TLS connectivity to both allowed services. See the
[Podman run documentation](https://docs.podman.io/en/latest/markdown/podman-run.1.html)
and [internal network behavior](https://docs.podman.io/en/latest/markdown/podman-network-create.1.html).
This is an evaluation boundary, not a general-purpose sandbox. It prevents
runtime access to host answers; training-data memorization still requires a
separate held-out benchmark.

### Smoke, full runs, and resume

Download the existing 40-query `output_data.json` at the fixed revision above to
`evaluation/output_data.json`, or set `DATASET` to its existing host location.
The build helper excludes this file even when it lives in `evaluation/`.
For a reproducible download, keep the dataset on the controller host:

```bash
mkdir -p "$HOME/chemgraph-benchmark-data"
export DATASET="$HOME/chemgraph-benchmark-data/output_data.json"
curl --fail --location --output "$DATASET" \
  https://huggingface.co/datasets/Autonomous-Scientific-Agents/groundtruth/resolve/8fe726352eb0437e4958282ca741d064fcfe6959/output_data.json
python - <<'PYDATA'
import os
from chemgraph.eval.datasets import load_dataset
items = load_dataset(os.environ["DATASET"])
assert len(items) == 40, f"Expected 40 queries, got {len(items)}"
assert {"3", "5", "6", "7", "8", "9", "31"}.issubset({item.id for item in items})
print(f"Validated {len(items)} queries")
PYDATA
```

Run the opt-in container integration test below before starting live evaluation.
Use an output directory that has not already been used for another run:

```bash
export DATASET=/absolute/path/to/output_data.json
OUTPUT_DIR=eval_results/podman-smoke bash run_eval.sh --query-ids 3 5 6 7 8 9 31
# Inspect execution errors, scores, artifacts and usage before the full run.
OUTPUT_DIR=eval_results/podman-full bash run_eval.sh
# Resume only the same dataset, images, selection, and execution settings:
OUTPUT_DIR=eval_results/podman-full bash run_eval.sh --resume
```

For a long full run, start `tmux new -s chemgraph-eval`, activate the environment
and set the variables inside that session, then run the full command. Detach
with **Ctrl-b**, then **d**; reconnect with `tmux attach -t chemgraph-eval`.
To change limits, append e.g. `--podman-cpus 4 --podman-memory-gb 8
--query-timeout-seconds 1800`, and reuse the same values when resuming.

After the smoke run, inspect `benchmark_*.json`/`.md`, model detail files,
`checkpoints/*.jsonl`, and `artifacts/` inside its output directory. Confirm all
28 records are present; investigate `timeout`, `worker_error`, `invalid_response`,
initialization, execution, or formatting errors before the full run. Review
accuracy separately from runtime health: an incorrect answer is a benchmark
result, whereas an infrastructure failure needs fixing. Resume reuses failures;
to retry them after a fix, use a new output directory. Metadata in
`podman-run.json` records the pinned images and isolation policy.

The launcher selects `argo:gpt-5.6-sol`, `argo:gpt-5.6-luna`,
`argo:claude-opus-5`, and `argo:claude-opus-4.8`, DeepAgent, structured judging,
and recursion limit 200. GPT reasoning effort retains its `none` default.
The smoke set has 28 combinations; the full set has 160 and uses a separate
output directory. `ARGO_USER`, `PODMAN_IMAGE`, and `PODMAN_PROXY_IMAGE` are
required. The launcher requires an explicit `OUTPUT_DIR` or `--output-dir` for
resume and accepts normal CLI overrides, including `--dataset`.

`--deepagent-auto-approve` authorizes unattended tools and shell execution inside
the worker. It remains required in Podman mode. CLI/profile settings are:

| CLI flag / profile key | Default |
| --- | --- |
| `--eval-runtime` / `eval_runtime` | `local`; accepts `local`, `podman` |
| `--podman-image` / `podman_image` | required in Podman mode |
| `--podman-proxy-image` / `podman_proxy_image` | required in Podman mode |
| `--podman-cpus` / `podman_cpus` | `4` |
| `--podman-memory-gb` / `podman_memory_gb` | `8` GiB |
| `--query-timeout-seconds` / `query_timeout_seconds` | `1800` in Podman mode |

All resource settings must be finite and positive. `deepagent_workspace` is
rejected in Podman mode. The versioned stdin request rejects unexpected fields
and carries only one question, its ID, model, resolved Argo settings, and
execution settings. It contains no benchmark configuration or ground truth.

The controller stops each worker before collecting its result and artifacts.
It validates JSON and rejects archive traversal, symbolic/hard links, special
files, duplicate entries, and excessive sizes. Artifacts are extracted to a
fresh directory under `OUTPUT_DIR/artifacts`; `artifact_location` records that
host path. Traces keep their original `/workspace` paths. Timeout, crash, or
invalid-response records score as failures. Completed failure records are reused
on resume. Image IDs, dependency/weight manifests, policy, resources, dataset,
and evaluator source fingerprints must match; old checkpoints are not migrated.
An exclusive output-directory lock prevents concurrent runs from writing there.

### Offline integration checks and cleanup

Unit tests run in the ordinary offline suite. Real Podman tests are opt-in and
use prebuilt test images, a temporary TLS certificate, fake Argo/PubChem servers,
and EMT. They make no live LLM calls or runtime downloads:

```bash
python evaluation/integration/build.py --worker-image "$PODMAN_IMAGE"
CHEMGRAPH_TEST_PODMAN=1 \
CHEMGRAPH_TEST_PODMAN_IMAGE=localhost/chemgraph-eval-test:latest \
CHEMGRAPH_TEST_PODMAN_PROXY_IMAGE="$PODMAN_PROXY_IMAGE" \
CHEMGRAPH_TEST_PODMAN_SERVICE_IMAGE=localhost/chemgraph-eval-test-services:latest \
pytest tests/test_eval_podman_integration.py -q
```

The test-only worker trusts the test certificate; use the production worker for
live evaluation. The test plants host and sibling-worker sentinels, exercises
file/shell/Python/chemistry readers and symlinks, and checks direct-IP, DNS,
proxy-bypass, host-service, and destination restrictions.

Normal completion, timeout, and cancellation remove containers and volumes.
After a hard kill or a reported cleanup failure, use the exact `run_id` from
`OUTPUT_DIR/podman-run.json` or the error message:

```bash
python evaluation/cleanup.py --run-id YOUR_32_CHARACTER_RUN_ID
```

Cleanup requires both the exact run label and matching resource-name prefix.
It does not prune unrelated Podman resources. Collected host artifacts remain.
