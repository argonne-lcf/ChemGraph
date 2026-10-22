# Scoped MACE-small optimization

This interactive example optimizes H2 with MACE-MP small on CPU. It demonstrates
the scoped main agent, exact-action approval, operation records, and saved
artifacts introduced in the [scoped runtime](../../docs/scoped_runtime.md).

## Setup and run

From the ChemGraph checkout containing the scoped runtime, install into your
Python environment:

```bash
python -m pip install -e ".[codex]"
```

The default LLM is `codex:gpt-5.6-luna`. Configure authentication as described in
[Codex subscription support](../../docs/codex_subscription.md). To use another
configured ChemGraph model, pass `--model`.

```bash
python -u examples/scoped_runtime/run_mace_small.py
python -u examples/scoped_runtime/run_mace_small.py --model gpt-4o-mini
```

MACE automatically downloads the `small` weights on first use and reuses its
normal cache afterward. You do not need to supply a model path. The first run
needs network access; `XDG_CACHE_HOME` can select another cache location.

Scoped execution needs the weights checksum before approving a calculation.
The example therefore calls MACE's own checkpoint downloader during setup and
binds the returned file to `mace_mp:small` in its policy. The agent requests
`model="small"`; preparation resolves it to the exact cached file before review.
This keeps the approved model identity stable through execution.

## Review and results

The script prints its run directory immediately, then reports dependency
loading, cache/download progress, and agent events. Initial imports, model
loading, and LLM turns can take time. `--help` works without loading ChemGraph
or downloading weights.

At `Approve this action? [y/N]`, inspect the prepared arguments, input snapshot,
model checksum, and operation digest. Enter `y` or `yes` to approve. Any other
answer rejects the action; optional feedback lets the agent propose a revision
for a new review. An empty rejection message asks it to stop retrying.

The script records the decision through `journal.decide()` before resuming the
graph. The resume value itself does not authorize execution. Approval expires
after ten minutes.

Each run keeps a fresh temporary directory, printed as `WORKSPACE`, containing:

- `ws/h2.xyz`: the generated input, with an initial bond length of 0.74 angstrom.
- `ws/h2_opt.json` and the optimization trajectory: calculation outputs.
- `ws/.chemgraph-artifacts/`: immutable artifact copies printed by the script.
- `ops/`: durable JSON operation records.
- `logs/`: agent logs.

The prompt gives the agent the absolute input filename because checkpoint-file
helpers such as `ls` inspect a different filesystem from native chemistry tools.
The input path is generated automatically; no user-supplied paths are required.

The calculation uses `float64`, `fmax=0.01` eV/angstrom, and at most 200 steps.
Exit code 0 requires a finalized optimization with `success=true`, convergence,
a finite saved energy, and the expected MACE weights. Rejection without a
successful revision, missing results, and failed verification return exit code 1.
The assistant's text alone is not used to determine success.

This is a local example with in-memory graph checkpoints and a new workspace on
every run. It exercises approval and artifact handling; restart recovery and
the persistent MCP service require separate demonstrations.
