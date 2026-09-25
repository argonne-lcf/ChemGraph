#!/bin/bash
# Start MCP and the planner/executor client inside an existing Aurora PBS allocation.
set -eo pipefail
abort() { echo "[ABORT] $*" >&2; exit 2; }
CG_INTERACTIVE=0
case "${1:-}" in
    --interactive) CG_INTERACTIVE=1 ;;
    "") ;;
    -h|--help)
        cat <<'USAGE'
Usage: bash examples/graspa_scaling/run.sh [--interactive]

Run inside an existing Aurora PBS allocation. --interactive selects the
20-CIF Parsl smoke run: 10,000 cycles per phase, H2O at 298 K and 960/320 Pa,
and alcf:openai/gpt-oss-120b. It streams agent.log to the terminal and uses a
fresh output directory for each invocation. Export ALCF_ACCESS_TOKEN first.

Override defaults with CG_ENV (or CG_SETUP_FILE), CG_MODEL, CG_CIF_DIR,
CG_LIMIT, N_CYCLES, CG_RUN_DIR, CG_STARTUP_TIMEOUT, CG_WAIT_TIMEOUT, and
CG_AGENT_TIMEOUT (overall client deadline, including MCP readiness).
Without --interactive, the batch/reference defaults apply.
USAGE
        exit 0 ;;
    *) abort "Unknown argument: $1; use --help." ;;
esac
(( $# <= 1 )) || abort "Expected at most one argument; use --help."
[[ -f "${PBS_NODEFILE:-}" ]] || abort "Run inside an Aurora PBS allocation."
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CG_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$CG_REPO"

if (( CG_INTERACTIVE )); then
    export CG_ENV="${CG_ENV:-${VIRTUAL_ENV:-/lus/flare/projects/ChemGraph/thang/ChemGraph/venv}}"
    export CG_CIF_DIR="${CG_CIF_DIR:-/lus/flare/projects/IQC/thang/ChemGraph_parsl/coremof_database/databases}"
    export CG_LIMIT="${CG_LIMIT:-20}"
    export N_CYCLES="${N_CYCLES:-10000}"
    export CG_WAIT_TIMEOUT="${CG_WAIT_TIMEOUT:-2700}"
    export CG_AGENT_TIMEOUT="${CG_AGENT_TIMEOUT:-3000}"
    CG_RUN_DIR="${CG_RUN_DIR:-$CG_REPO/graspa_scaling_runs/interactive-${PBS_JOBID:-allocation}-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
fi

# Module/conda setup scripts may use unset shell variables.
if [[ -n "${CG_SETUP_FILE:-}" ]]; then
    [[ "$CG_SETUP_FILE" == /* && -r "$CG_SETUP_FILE" ]] || abort "CG_SETUP_FILE must be a readable absolute path."
    source "$CG_SETUP_FILE"
    printf -v ENV_INIT 'source %q' "$CG_SETUP_FILE"
else
    [[ "${CG_ENV:-}" == /* && -r "${CG_ENV:-}/bin/activate" ]] || abort "Set CG_ENV to your existing venv, or pass CG_SETUP_FILE."
    read -r -a CG_MODULE_LIST <<< "${CG_MODULES-frameworks}"
    ENV_INIT=""
    if (( ${#CG_MODULE_LIST[@]} )); then
        module load "${CG_MODULE_LIST[@]}"
        printf -v ENV_INIT 'module load'
        for name in "${CG_MODULE_LIST[@]}"; do
            printf -v ENV_INIT '%s %q' "$ENV_INIT" "$name"
        done
        ENV_INIT+=" && "
    fi
    source "$CG_ENV/bin/activate"
    printf -v ENV_INIT '%ssource %q' "$ENV_INIT" "$CG_ENV/bin/activate"
fi
set -u

# Shared defaults; interactive mode changes only workload/environment defaults above.
CG_MODEL="${CG_MODEL:-alcf:openai/gpt-oss-120b}"
CG_CIF_DIR="${CG_CIF_DIR:-/lus/flare/projects/IQC/thang/ChemGraph_parsl/weak_scaling_rerun/random_sampling/512_nodes/cif_files}"
CG_LIMIT="${CG_LIMIT:-0}"
N_CYCLES="${N_CYCLES:-2000000}"
CG_STARTUP_TIMEOUT="${CG_STARTUP_TIMEOUT:-300}"
CG_WAIT_TIMEOUT="${CG_WAIT_TIMEOUT:-9900}"
CG_AGENT_TIMEOUT="${CG_AGENT_TIMEOUT:-10200}"
CG_RECURSION_LIMIT="${CG_RECURSION_LIMIT:-100}"
CG_MCP_PORT="${CG_MCP_PORT:-9001}"
CG_RUN_DIR="${CG_RUN_DIR:-$CG_REPO/graspa_scaling_runs/${PBS_JOBID:-interactive-$$}}"

export CG_PYTHON="${CG_PYTHON:-python}"
export PYTHONPATH="$CG_REPO/src"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE="${CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE:-9}"
export COMPUTE_SYSTEM=aurora
export CHEMGRAPH_EXECUTION_BACKEND=parsl
export CHEMGRAPH_GRASPA_EXECUTABLE="${CHEMGRAPH_GRASPA_EXECUTABLE:-/lus/flare/projects/ChemGraph/thang/soft/gRASPA/graspa-sycl/bin/sycl.out}"
[[ -x "$CHEMGRAPH_GRASPA_EXECUTABLE" ]] || abort "gRASPA executable is not executable: $CHEMGRAPH_GRASPA_EXECUTABLE"
if [[ "$CG_MODEL" == alcf:* && -z "${ALCF_ACCESS_TOKEN:-}" ]]; then
    abort "Export ALCF_ACCESS_TOKEN (pass with qsub -v for batch jobs), or load it in CG_SETUP_FILE."
fi
export OMP_NUM_THREADS="${CG_OMP_NUM_THREADS:-1}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
printf -v WORKER_EXPORTS 'export PYTHONPATH=%q CHEMGRAPH_GRASPA_EXECUTABLE=%q OMP_NUM_THREADS=%q PYTHONNOUSERSITE=1 ZE_FLAT_DEVICE_HIERARCHY=FLAT' \
    "$PYTHONPATH" "$CHEMGRAPH_GRASPA_EXECUTABLE" "$OMP_NUM_THREADS"
export CHEMGRAPH_WORKER_INIT="${CHEMGRAPH_WORKER_INIT:-$ENV_INIT} && $WORKER_EXPORTS"

# LLM requests leave the compute node through the proxy; MCP stays on loopback.
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="$NO_PROXY"
export GRASPA_MCP_URL="http://127.0.0.1:$CG_MCP_PORT/mcp/"

[[ -d "$CG_CIF_DIR" ]] || abort "Missing CG_CIF_DIR: $CG_CIF_DIR"
CG_CIF_DIR="$(cd "$CG_CIF_DIR" && pwd)"
[[ ! -e "$CG_RUN_DIR" ]] || abort "Output already exists; choose a fresh CG_RUN_DIR: $CG_RUN_DIR"
mkdir -p "$CG_RUN_DIR"
CG_RUN_DIR="$(cd "$CG_RUN_DIR" && pwd)"
export CHEMGRAPH_LOG_DIR="$CG_RUN_DIR"
cd "$CG_RUN_DIR"
printf 'Checkout: %s\nOutput: %s\nMCP: %s\n' "$CG_REPO" "$CG_RUN_DIR" "$GRASPA_MCP_URL"
printf 'Input: %s\n' "$CG_CIF_DIR"
if (( CG_INTERACTIVE )); then
    printf 'Interactive Parsl: CIF limit=%s, cycles/phase=%s, model=%s, workers/node=%s\n' \
        "$CG_LIMIT" "$N_CYCLES" "$CG_MODEL" "$CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE"
    printf 'Starting MCP; startup logs: %s/mcp.log\n' "$CG_RUN_DIR"
fi
"$CG_PYTHON" -c 'import chemgraph; print("ChemGraph source:", chemgraph.__file__)'

MCP_PID=""
CLIENT_PID=""
TAIL_PID=""
cleanup() {
    trap - EXIT INT TERM
    for pid in ${TAIL_PID:-} ${CLIENT_PID:-} ${MCP_PID:-}; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in ${TAIL_PID:-} ${CLIENT_PID:-} ${MCP_PID:-}; do
        timeout 30s tail --pid="$pid" -f /dev/null 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Refuse to attach to a server already using this port.
"$CG_PYTHON" - "$CG_MCP_PORT" <<'PY'
import socket
import sys
with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY
"$CG_PYTHON" -m chemgraph.mcp.graspa_mcp_hpc \
    --transport streamable_http --host 127.0.0.1 --port "$CG_MCP_PORT" \
    >"$CG_RUN_DIR/mcp.log" 2>&1 &
MCP_PID=$!

EXTRA_ARGS=()
[[ -z "${CG_BASE_URL:-}" ]] || EXTRA_ARGS+=(--base-url "$CG_BASE_URL")
[[ -z "${CG_SIMULATION_TIMEOUT:-}" ]] || EXTRA_ARGS+=(--simulation-timeout "$CG_SIMULATION_TIMEOUT")
# The client waits for MCP readiness before running one ensemble with both conditions.
: >"$CG_RUN_DIR/agent.log"  # Ensure the interactive tail can open it immediately.
timeout --kill-after=30s "$CG_AGENT_TIMEOUT" \
    "$CG_PYTHON" "$CG_REPO/examples/graspa_scaling/run_graspa.py" \
    --input-dir "$CG_CIF_DIR" --limit "$CG_LIMIT" --mcp-url "$GRASPA_MCP_URL" --output-dir "$CG_RUN_DIR" \
    --ads-temp "${ADS_TEMP_K:-298}" --ads-pressure "${ADS_PRESSURE_PA:-960}" \
    --des-temp "${DES_TEMP_K:-298}" --des-pressure "${DES_PRESSURE_PA:-320}" \
    --n-cycles "$N_CYCLES" --startup-timeout "$CG_STARTUP_TIMEOUT" --wait-timeout "$CG_WAIT_TIMEOUT" \
    --model "$CG_MODEL" --recursion-limit "$CG_RECURSION_LIMIT" \
    "${EXTRA_ARGS[@]}" >"$CG_RUN_DIR/agent.log" 2>&1 &
CLIENT_PID=$!
if (( CG_INTERACTIVE )); then
    tail --pid="$CLIENT_PID" -n +1 -f "$CG_RUN_DIR/agent.log" &
    TAIL_PID=$!
fi
status=0
wait -n "$MCP_PID" "$CLIENT_PID" || status=$?
if ! kill -0 "$MCP_PID" 2>/dev/null; then
    echo "MCP exited before cleanup; inspect mcp.log." >&2
    status=1
fi
printf 'Agent exit status: %s\nResults and logs: %s\n' "$status" "$CG_RUN_DIR"
exit "$status"
