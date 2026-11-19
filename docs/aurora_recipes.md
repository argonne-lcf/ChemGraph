# Aurora LLM Inference Recipes for ChemGraph

Full-XPU, maximum-context llama.cpp inference on ALCF Aurora for four models, served to ChemGraph via
the [`aurora:`](running_local_models.md#aurora-on-node-inference-aurora-models) provider. Every recipe
keeps **all model weights on the GPU** (no MoE→CPU offload) and runs at each model's **native maximum
context length**. GPU tiles are partitioned so the LLM and the scientific workload never share tiles,
and both CPU sockets stay free for the science process.

| Model | `aurora:` id | Quant / GGUF | LLM tiles | Free for science | Max context |
|-------|--------------|--------------|-----------|-----------------|-------------|
| gpt-oss-120b | `aurora:gpt-oss-120b` | MXFP4 (~60 GB) | 2 (1 GPU) | 10 tiles + both CPUs | 131 072 |
| Nemotron-3-Ultra-550B | `aurora:nemotron-3-ultra` | UD-IQ2_M (~181 GB) | 6 (3 GPUs) | 6 tiles + both CPUs | 262 144 |
| Inkling | `aurora:inkling` | UD-IQ1_S (~270 GB) | 8 (4 GPUs) | 4 tiles + both CPUs | 131 072 |
| Nemotron-4-340B | `aurora:nemotron-4-340b` | i1-Q4_K_M (~196 GB) | 6 (3 GPUs) | 6 tiles + both CPUs | 4 096 |

Each Aurora node has 6 GPUs × 2 tiles = **12 tiles** of ~64 GB HBM. The KV cache is small relative to
weights (few KV heads), so max context fits the listed tile counts with headroom — no extra tiles are
needed to go from a short context to the maximum.

---

## Prerequisites

### 1. Build llama.cpp with the SYCL backend

**There is no facility-wide llama.cpp module on Aurora — you must build it once.** A self-contained PBS
script is provided; it uses only Aurora modules (oneapi, cmake, ninja) and system git (no conda), and
merges the Inkling PR so a single build serves all four models.

```bash
qsub -q debug -v LLAMA_ROOT=$HOME/llamacpp-sycl scripts/aurora/build_llamacpp_sycl.pbs
# result: $HOME/llamacpp-sycl/build/bin/llama-server  (+ llama-cli, llama-bench)
```

The serve scripts default to `BIN=$HOME/llamacpp-sycl/build/bin`; override `BIN` if you build elsewhere.
Key CMake flags (already set in the script): `-DGGML_SYCL=ON -DGGML_SYCL_F16=ON
-DGGML_SYCL_DEVICE_ARCH=pvc -DCMAKE_CXX_COMPILER=icpx -DLLAMA_BUILD_SERVER=ON`.

Architecture notes: `gpt-oss`, `nemotron_h` (Nemotron-3-Ultra), and `nemotron` (Nemotron-4-340B) are on
recent llama.cpp `master`; the `inkling` architecture needs PR #25731, which the build script merges
(`BUILD_INKLING=1`, default).

### 2. Download the GGUF weights

Point each recipe's `GGUF` at the model's first shard. The `alcf-aurora-llm` campaign contains the
download scripts; place the GGUFs where the serve script's `GGUF` variable points (or override it).

---

## Common launch pattern

All models use the same unified serve script, which selects tiles, GGUF, max context, and alias from
the model key:

```bash
# on the LLM node:
BIN=$HOME/llamacpp-sycl/build/bin \
  scripts/aurora/serve_llamacpp_maxctx.sh <model-key> [port]
#   model-key ∈ { gpt-oss-120b | nemotron-3-ultra | inkling | nemotron-4-340b }
```

It runs `llama-server` with: `-ngl 99` (all layers on GPU), `--split-mode layer` (row split crashes on
`nemotron_h`), **no `-ncmoe`**, `-fa on`, `--jinja` (tool calls), `-c <model max>`,
`--parallel 4 --cont-batching`, `--host 0.0.0.0`. `ZE_AFFINITY_MASK` pins the model's tile group;
`GGML_SYCL_ENABLE_VMM=0` is required on PVC.

Then point ChemGraph at it (co-located, or from another node via the intra-cluster IP / SSH tunnel):

```bash
export AURORA_BASE_URL="http://<llm_node_ip>:8000/v1"
chemgraph run --model aurora:<model-key> -q "..."
# or:  chemgraph run --config config/aurora_<model>.toml -q "..."
```

For the scientific workload, set **its own** `ZE_AFFINITY_MASK` to the complementary (free) tiles so
the two processes never share a tile.

---

## Recipe 1 — gpt-oss-120b (2 tiles, same node)

LLM on tiles 0–1 (1 GPU); science on tiles 2–11 + both CPU sockets.

```bash
scripts/aurora/serve_llamacpp_maxctx.sh gpt-oss-120b 8000
# ZE_AFFINITY_MASK=0,1  -c 131072  --split-mode layer  (weights ~60 GB + KV ~19 GB @ max)
# Science: export ZE_AFFINITY_MASK=2,3,4,5,6,7,8,9,10,11
chemgraph run --model aurora:gpt-oss-120b -q "Build water from SMILES O and optimize with EMT."
```
gpt-oss uses `openai_harmony`, which fetches its tiktoken vocab; the serve script sets the ALCF proxy
(+ `no_proxy=127.0.0.1`) and caches it under `TIKTOKEN_RS_CACHE_DIR`.
Reference: decode ~34 tok/s (2-tile), TTFT ~50 ms; ~8% decode penalty at 131 072 vs short context.

## Recipe 2 — Nemotron-3-Ultra-550B (6 tiles, same node)

LLM on tiles 0–5 (3 GPUs); science on tiles 6–11 + both CPU sockets.

```bash
scripts/aurora/serve_llamacpp_maxctx.sh nemotron-3-ultra 8000
# ZE_AFFINITY_MASK=0,1,2,3,4,5  -c 262144  --split-mode layer  (weights ~181 GB + KV ~3 GB @ max)
# Science: export ZE_AFFINITY_MASK=6,7,8,9,10,11
chemgraph run --model aurora:nemotron-3-ultra -q "..."
```
MoE experts are on GPU (no CPU offload), so the LLM uses little CPU/DDR during decode. Layer split is
pipeline-serial → decode ~5.8 tok/s (vs ~7 tok/s for the older MoE→CPU 1-tile path — same order of
magnitude, and it keeps both CPU sockets free). Min tiles for max context = 4; 6 is used for headroom.

KV note: Nemotron-3-Ultra is a **hybrid** model — only its 12 attention layers carry context-scaling
KV (the 48 Mamba2 layers hold a small constant state; the 48 MoE layers none), and `kv_heads=2`. So
even at 262 144 the KV cache is only ~3 GB — it is weights-bound, not KV-bound. (Full-XPU layer-split
was validated at 8 tiles / short context; at max context KV adds ~3 GB, so 6 tiles remain ample.)

## Recipe 3 — Inkling (8 tiles, or full node with ChemGraph on a second node)

LLM on tiles 0–7 (4 GPUs); science on tiles 8–11 + both CPUs. For maximum isolation, run Inkling on
one node and ChemGraph on a second node (two-node PBS below).

```bash
scripts/aurora/serve_llamacpp_maxctx.sh inkling 8000
# ZE_AFFINITY_MASK=0,1,...,7  -c 131072  --split-mode layer  (weights ~270 GB + KV ~33 GB @ max)
# Science: export ZE_AFFINITY_MASK=8,9,10,11
chemgraph run --model aurora:inkling -q "..."
```
Reference: decode ~6.7 tok/s. Min tiles for max context = 6; 8 is used for headroom / better split.

## Recipe 4 — Nemotron-4-340B (6 tiles, same node)

LLM on tiles 0–5 (3 GPUs); science on tiles 6–11 + both CPUs. Dense model (no MoE).

```bash
scripts/aurora/serve_llamacpp_maxctx.sh nemotron-4-340b 8000
# ZE_AFFINITY_MASK=0,1,2,3,4,5  -c 4096  --split-mode layer  (weights ~196 GB + KV ~1.6 GB @ max)
# Science: export ZE_AFFINITY_MASK=6,7,8,9,10,11
chemgraph run --model aurora:nemotron-4-340b -q "..."
```
Reference: decode ~2.3 tok/s, TTFT ~2.4 s (dense 340B does full-model FLOPs/token). Max context for
the Q4_K_M quant is 4 096 (GGUF n_ctx_train=4096; llama.cpp clamps to it). Min tiles = 4; 6 used for headroom.

---

## Two-node deployment (LLM on one node, ChemGraph on another)

For full compute isolation, run the LLM server on one node and the ChemGraph scientific workload on a
separate node. The nodes communicate via node 0's intra-cluster IP (written to `ENDPOINT.txt`).

```bash
qsub -q debug-scaling scripts/aurora/two_node_inkling.pbs           # Inkling + ChemGraph
qsub -q debug-scaling scripts/aurora/two_node_nemotron_ultra.pbs    # Nemotron-3-Ultra + ChemGraph
```

Both scripts run `select=2`: node 0 serves the model full-XPU (via SSH), node 1 runs `chemgraph run`
against `AURORA_BASE_URL=http://<node0_ip>:8000/v1`.

---

## Context length note

ChemGraph prepends the full conversation history on every LLM call, so context grows per agent step; a
short `-c 4096` truncates a simple multi-step workflow (`finish_reason: length`). These recipes use each
model's **native maximum context**, which fits the listed tile counts (KV cache is small vs weights),
so ChemGraph runs of any realistic length are safe. If you need to trade context for a bit more decode
speed, lower `-c` via the `CTX` env var (e.g. `CTX=8192`).

## Known caveats

- **Nemotron-4-340B is limited to 4096 context (model architecture).** Nemotron-4-340B-Instruct was
  pretrained with `max_position_embeddings = 4096` and **no RoPE scaling** (confirmed in both the
  upstream HF config and the GGUF header, `nemotron.context_length = 4096`). llama.cpp clamps any larger
  `-c` down to 4096 (`n_ctx_seq > n_ctx_train` warning). This is a hard architectural limit, not a
  quantization or checkpoint artifact — there is no official long-context variant, and forcing longer
  context via RoPE extension degrades quality (and can break tool-calling, which ChemGraph relies on).
  4096 is sufficient for typical single-agent tasks (the co-located EMT test passed), but is marginal
  for long multi-tool chains. **For long-context ChemGraph workflows, use gpt-oss-120b (131072),
  Inkling (131072), or Nemotron-3-Ultra (262144) instead.**

## File locations

| File | Purpose |
|------|---------|
| `scripts/aurora/build_llamacpp_sycl.pbs` | Build llama.cpp SYCL for Aurora (all 4 models) |
| `scripts/aurora/serve_llamacpp_maxctx.sh` | Unified full-XPU max-context serve (pick model by key) |
| `scripts/aurora/two_node_inkling.pbs` | Two-node: Inkling LLM + ChemGraph |
| `scripts/aurora/two_node_nemotron_ultra.pbs` | Two-node: Nemotron-3-Ultra LLM + ChemGraph |
| `scripts/aurora/chemgraph_aurora_env.sh` | Export `AURORA_BASE_URL` from `ENDPOINT.txt` |
| `config/aurora_gpt_oss.toml` | ChemGraph config: `aurora:gpt-oss-120b` |
| `config/aurora_nemotron_ultra.toml` | ChemGraph config: `aurora:nemotron-3-ultra` |
| `config/aurora_inkling.toml` | ChemGraph config: `aurora:inkling` |
| `config/aurora_nemotron4.toml` | ChemGraph config: `aurora:nemotron-4-340b` |
