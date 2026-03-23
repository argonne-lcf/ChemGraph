#!/bin/bash
# ==============================================================================
# start_mcp_server_interactive.sh
#
# Start the ChemGraph MCP server on an ALCF compute node via HTTP.
#
# Usage (after getting an interactive session via qsub -I):
#   ./start_mcp_server_interactive.sh [OPTIONS]
#
# Options:
#   --port PORT         Port for the MCP HTTP server (default: 9003)
#   --venv PATH         Path to virtual environment to activate
#   --log-dir PATH      Directory for MCP logs (default: ./chemgraph_mcp_logs)
#   --mcp-module MOD    Python module to run (default: chemgraph.mcp.mcp_tools)
#   --help              Show this help message
#
# Example:
#   # 1. Get an interactive compute node
#   qsub -I -l select=1 -l walltime=01:00:00 -l filesystems=home:flare -q debug -A myproject
#
#   # 2. Run the script on the compute node
#   ./start_mcp_server_interactive.sh --venv /path/to/venv --port 9003
#
#   # 3. Set up an SSH tunnel from login node to connect:
#   #    ssh -L 9003:COMPUTE_NODE:9003 COMPUTE_NODE
#   #    Then: http://localhost:9003/mcp/
# ==============================================================================

set -eo pipefail

# --------------- Default configuration ---------------
MCP_PORT=9003
VENV_PATH=""
LOG_DIR="./chemgraph_mcp_logs"
MCP_MODULE="chemgraph.mcp.mcp_tools"

# --------------- Parse arguments ---------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            MCP_PORT="$2"; shift 2 ;;
        --venv)
            VENV_PATH="$2"; shift 2 ;;
        --log-dir)
            LOG_DIR="$2"; shift 2 ;;
        --mcp-module)
            MCP_MODULE="$2"; shift 2 ;;
        --help)
            head -n 27 "$0" | tail -n +2 | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1 ;;
    esac
done

# --------------- Helper functions ---------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cleanup() {
    log "Shutting down..."
    if [[ -n "${MCP_PID:-}" ]] && kill -0 "$MCP_PID" 2>/dev/null; then
        log "Stopping MCP server (PID: $MCP_PID)"
        kill "$MCP_PID" 2>/dev/null || true
    fi
    log "Cleanup complete."
}

trap cleanup EXIT INT TERM

# --------------- Detect environment ---------------
COMPUTE_NODE=$(hostname)
log "Compute node: $COMPUTE_NODE"

# --------------- Set up environment ---------------
# ALCF proxy settings (needed for PubChem lookups, etc.)
export http_proxy="proxy.alcf.anl.gov:3128"
export https_proxy="proxy.alcf.anl.gov:3128"
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=127.0.0.1,localhost,::1

# Load ALCF frameworks module
if command -v module &>/dev/null; then
    log "Loading frameworks module..."
    module load frameworks 2>/dev/null || log "WARNING: 'module load frameworks' failed (may not be needed)"
fi

# Activate virtual environment if specified
if [[ -n "$VENV_PATH" ]]; then
    log "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate" || { log "ERROR: Failed to activate venv: $VENV_PATH"; exit 1; }
fi

# Resolve the python binary
if [[ -n "$VENV_PATH" && -x "$VENV_PATH/bin/python" ]]; then
    PYTHON="$VENV_PATH/bin/python"
elif command -v python &>/dev/null; then
    PYTHON="$(command -v python)"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
else
    log "ERROR: No python or python3 found on PATH"
    exit 1
fi
log "Python: $PYTHON"

# Set up log directory
export CHEMGRAPH_LOG_DIR="$LOG_DIR"
mkdir -p "$CHEMGRAPH_LOG_DIR"
MCP_LOG_FILE="$CHEMGRAPH_LOG_DIR/mcp_server_$(date '+%Y%m%d_%H%M%S').log"

log "MCP module:    $MCP_MODULE"
log "MCP port:      $MCP_PORT"
log "Log directory: $CHEMGRAPH_LOG_DIR"
log "Log file:      $MCP_LOG_FILE"

# --------------- Start the MCP server ---------------
log "Starting MCP server on $COMPUTE_NODE:$MCP_PORT ..."

"$PYTHON" -m "$MCP_MODULE" \
    --transport streamable_http \
    --port "$MCP_PORT" \
    > "$MCP_LOG_FILE" 2>&1 &

MCP_PID=$!
log "MCP server started with PID: $MCP_PID"

# Wait for the server to be ready
log "Waiting for MCP server to become ready..."
MAX_WAIT=120
WAITED=0
while [[ $WAITED -lt $MAX_WAIT ]]; do
    if ! kill -0 "$MCP_PID" 2>/dev/null; then
        log "ERROR: MCP server process exited unexpectedly. Check logs:"
        tail -n 20 "$MCP_LOG_FILE"
        exit 1
    fi
    if grep -q "Uvicorn running on\|Application startup complete\|Started server" "$MCP_LOG_FILE" 2>/dev/null; then
        log "MCP server is ready!"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

if [[ $WAITED -ge $MAX_WAIT ]]; then
    log "WARNING: Timed out waiting for server ready signal (${MAX_WAIT}s)."
    log "The server may still be starting. Last log lines:"
    tail -n 10 "$MCP_LOG_FILE"
fi

# --------------- Print connection info ---------------
log ""
log "============================================================"
log "  MCP server is running at:"
log "    http://${COMPUTE_NODE}:${MCP_PORT}/mcp/"
log ""
log "  To connect from the login node, set up an SSH tunnel:"
log "    ssh -L ${MCP_PORT}:${COMPUTE_NODE}:${MCP_PORT} ${COMPUTE_NODE}"
log "    Then: http://localhost:${MCP_PORT}/mcp/"
log "============================================================"
log ""

# --------------- Keep alive ---------------
log "Server is running. Press Ctrl+C to stop."
log "Tailing server log (${MCP_LOG_FILE}):"
log ""

# Wait for the MCP server process; tail the log in the foreground
tail -f "$MCP_LOG_FILE" &
TAIL_PID=$!

wait "$MCP_PID" 2>/dev/null
EXIT_CODE=$?

kill "$TAIL_PID" 2>/dev/null || true
log "MCP server exited with code: $EXIT_CODE"
exit $EXIT_CODE

