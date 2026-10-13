#!/bin/bash
# Serve one of the four Aurora models with llama.cpp SYCL, FULL-XPU (all weights on GPU, no MoE
# offload), at each model's MAXIMUM context length. Source-able / callable; picks tiles, GGUF, and
# context automatically from MODEL. Exposes an OpenAI-compatible /v1 endpoint for ChemGraph's
# `aurora:` provider.
#
# Usage:
#   scripts/aurora/serve_llamacpp_maxctx.sh gpt-oss-120b        [PORT]
#   scripts/aurora/serve_llamacpp_maxctx.sh nemotron-3-ultra    [PORT]
#   scripts/aurora/serve_llamacpp_maxctx.sh inkling             [PORT]
#   scripts/aurora/serve_llamacpp_maxctx.sh nemotron-4-340b     [PORT]
#
# All models: full-XPU (-ngl 99, no -ncmoe), layer split across the model's tile group, -fa on,
# --jinja (tool calls). Context is each model's native maximum; KV cache is small vs weights so it
# fits the listed tile counts. Tiles not used by the LLM (and both CPU sockets) are free for the
# scientific workload — set that process's own ZE_AFFINITY_MASK to the complementary tiles.

set -o pipefail

MODEL_KEY=${1:?"usage: serve_llamacpp_maxctx.sh <gpt-oss-120b|nemotron-3-ultra|inkling|nemotron-4-340b> [port]"}
PORT=${2:-8000}

ALCF=/lus/flare/projects/MatSciAI/xiaoliyan/workdir/alcf-aurora-llm
BIN=${BIN:-$HOME/llamacpp-sycl/build/bin}   # your llama.cpp SYCL build (see build_llamacpp_sycl.pbs)
NThreads=${NThreads:-64}
NPAR=${NPAR:-4}

# Per-model: GGUF (first shard), tile mask (full-XPU), native max context, served alias.
case "$MODEL_KEY" in
  gpt-oss-120b)
    GGUF=$ALCF/gpt-oss-120b/models/ggml-org-gpt-oss-120b-GGUF/gpt-oss-120b-MXFP4.gguf
    TILES=${TILES:-0,1}                     # 2 tiles (1 GPU); free: 2-11
    CTX=${CTX:-131072}
    SERVED=gpt-oss-120b
    # gpt-oss needs the openai_harmony vocab (fetched via proxy on first run, then cached)
    export http_proxy=${http_proxy:-http://proxy.alcf.anl.gov:3128}
    export https_proxy=${https_proxy:-http://proxy.alcf.anl.gov:3128}
    export no_proxy=${no_proxy:-127.0.0.1,localhost} NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
    export TIKTOKEN_RS_CACHE_DIR=${TIKTOKEN_RS_CACHE_DIR:-$HOME/.cache/tiktoken-rs}
    mkdir -p "$TIKTOKEN_RS_CACHE_DIR"
    ;;
  nemotron-3-ultra)
    GGUF=$ALCF/nemotron/models/gguf/UD-IQ2_M/NVIDIA-Nemotron-3-Ultra-550B-A55B-UD-IQ2_M-00001-of-00005.gguf
    TILES=${TILES:-0,1,2,3,4,5}             # 6 tiles (3 GPUs); free: 6-11.  (min 4 for max ctx)
    CTX=${CTX:-262144}
    SERVED=nemotron-3-ultra
    ;;
  inkling)
    GGUF=$ALCF/inkling/models/unsloth-Inkling-GGUF/UD-IQ1_S/inkling-UD-IQ1_S-00001-of-00007.gguf
    TILES=${TILES:-0,1,2,3,4,5,6,7}         # 8 tiles (4 GPUs); free: 8-11.  (min 6 for max ctx)
    CTX=${CTX:-131072}
    SERVED=inkling
    ;;
  nemotron-4-340b)
    GGUF=$ALCF/nemotron/models/gguf-n4/Nemotron-4-340B-Instruct-hf.i1-Q4_K_M.gguf
    TILES=${TILES:-0,1,2,3,4,5}             # 6 tiles (3 GPUs); free: 6-11.  (min 4 for max ctx)
    CTX=${CTX:-4096}   # i1-Q4_K_M GGUF n_ctx_train=4096
    SERVED=nemotron-4-340b
    ;;
  *) echo "unknown MODEL_KEY '$MODEL_KEY'"; exit 2 ;;
esac

echo "START $(date -Is) host=$(hostname) MODEL=$MODEL_KEY TILES=$TILES CTX=$CTX PORT=$PORT"
[ -e "$GGUF" ] || { echo "ERROR: GGUF not found ($GGUF)"; exit 2; }

module load oneapi/release/2025.3.1
# Full-XPU: do NOT set ONEAPI_DEVICE_SELECTOR/ZE_AFFINITY conflicts; ZE_AFFINITY_MASK selects the
# model's tiles, layer split spreads weights across them. GGML_SYCL_ENABLE_VMM=0 required on PVC.
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=$TILES
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZES_ENABLE_SYSMAN=1
export GGML_SYCL_ENABLE_VMM=0
JT=${PBS_JOBID:-$$}
export TRITON_CACHE_DIR=/tmp/tc_${JT} SYCL_CACHE_DIR=/tmp/sc_${JT}
mkdir -p "$TRITON_CACHE_DIR" "$SYCL_CACHE_DIR"

# -sm layer (row split crashes on nemotron_h_moe); -ngl 99 all layers on GPU; NO -ncmoe.
exec numactl --interleave=all \
  "$BIN/llama-server" \
    -m "$GGUF" \
    -c "$CTX" -ngl 99 --split-mode layer \
    -fa on -t "$NThreads" --no-mmap \
    --jinja --parallel "$NPAR" --cont-batching \
    --alias "$SERVED" \
    --host 0.0.0.0 --port "$PORT"
