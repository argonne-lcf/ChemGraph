#!/bin/bash
# ==============================================================================
# start_mcp_server_interactive.sh
#
# Start the ChemGraph XANES MCP server on a compute node via HTTP
# during an interactive session.
#
# Usage (after getting an interactive session via qsub -I or locally):
#   ./start_mcp_server_interactive.sh [OPTIONS]
#
# Options:
#   --port PORT         Port for the MCP HTTP server (default: 9007)
#   --venv PATH         Path to virtual environment to activate
#   --fdmnes PATH       Path to FDMNES executable
#   --log-dir PATH      Directory for MCP logs (default: ./chemgraph_xanes_logs)
#   --help              Show this help message
#
# Example:
#   ./start_mcp_server_interactive.sh --venv /path/to/venv --fdmnes /path/to/fdmnes
#
#   # If on a remote compute node, set up an SSH tunnel from the login node:
#   #   ssh -L 9007:COMPUTE_NODE:9007 COMPUTE_NODE
#   #   Then: http://localhost:9007/mcp/
# ==============================================================================

set -eo pipefail

# --------------- Default configuration ---------------
MCP_PORT=9007
VENV_PATH=""
FDMNES_PATH=""
LOG_DIR="./chemgraph_xanes_logs"
MCP_MODULE="chemgraph.mcp.xanes_mcp"

# --------------- Parse arguments ---------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            MCP_PORT="$2"; shift 2 ;;
        --venv)
            VENV_PATH="$2"; shift 2 ;;
        --fdmnes)
            FDMNES_PATH="$2"; shift 2 ;;
        --log-dir)
            LOG_DIR="$2"; shift 2 ;;
        --help)
            head -n 23 "$0" | tail -n +2 | sed 's/^# \?//'
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
# Proxy settings (for Materials Project API calls on HPC)
if [[ -n "${http_proxy:-}" ]]; then
    export NO_PROXY=127.0.0.1,localhost,::1
    export no_proxy=127.0.0.1,localhost,::1
fi

# Load frameworks module if available (ALCF HPC)
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

# Set FDMNES executable
if [[ -n "$FDMNES_PATH" ]]; then
    export FDMNES_EXE="$FDMNES_PATH"
fi
if [[ -z "${FDMNES_EXE:-}" ]]; then
    log "WARNING: FDMNES_EXE is not set. run_xanes_single will fail."
    log "  Pass --fdmnes /path/to/fdmnes or export FDMNES_EXE."
fi

# Set up log directory
export CHEMGRAPH_LOG_DIR="$LOG_DIR"
mkdir -p "$CHEMGRAPH_LOG_DIR"
MCP_LOG_FILE="$CHEMGRAPH_LOG_DIR/xanes_mcp_$(date '+%Y%m%d_%H%M%S').log"

log "MCP module:      $MCP_MODULE"
log "MCP port:        $MCP_PORT"
log "FDMNES_EXE:      ${FDMNES_EXE:-NOT SET}"
log "MP_API_KEY:      ${MP_API_KEY:+SET (hidden)}"
log "Log file:        $MCP_LOG_FILE"

# --------------- Start the MCP server ---------------
log "Starting XANES MCP server on $COMPUTE_NODE:$MCP_PORT ..."

"$PYTHON" -u -m "$MCP_MODULE" \
    --transport streamable_http \
    --host 0.0.0.0 \
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
log "  XANES MCP server is running at:"
log "    http://${COMPUTE_NODE}:${MCP_PORT}/mcp/"
log ""
log "  To connect from a remote host, set up an SSH tunnel:"
log "    ssh -L ${MCP_PORT}:${COMPUTE_NODE}:${MCP_PORT} ${COMPUTE_NODE}"
log "    Then: http://localhost:${MCP_PORT}/mcp/"
log "============================================================"
log ""

# --------------- Keep alive ---------------
log "Server is running. Press Ctrl+C to stop."
log "Tailing server log (${MCP_LOG_FILE}):"
log ""

tail -f "$MCP_LOG_FILE" &
TAIL_PID=$!

wait "$MCP_PID" 2>/dev/null
EXIT_CODE=$?

kill "$TAIL_PID" 2>/dev/null || true
log "MCP server exited with code: $EXIT_CODE"
exit $EXIT_CODE
