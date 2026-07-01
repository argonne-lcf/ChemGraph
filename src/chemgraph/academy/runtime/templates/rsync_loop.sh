#!/bin/bash
set -euo pipefail

HOST="$1"
CONTROL_PATH="$2"
REMOTE_RUN_DIR="$3"
LOCAL_RUN_DIR="$4"
INTERVAL_S="$5"
LOG_PATH="$6"

mkdir -p "${LOCAL_RUN_DIR}"
while true; do
  rsync -az --delete \
    -e "ssh -o BatchMode=yes -o ControlMaster=auto -o ControlPath=${CONTROL_PATH} -o ControlPersist=yes" \
    "${HOST}:${REMOTE_RUN_DIR}/" \
    "${LOCAL_RUN_DIR}/" \
    >> "${LOG_PATH}" 2>&1 || true
  sleep "${INTERVAL_S}"
done
