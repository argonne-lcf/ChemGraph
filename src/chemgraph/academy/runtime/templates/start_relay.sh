#!/bin/bash
set -euo pipefail

REMOTE_ROOT="$1"
RELAY_SCRIPT="$2"
RELAY_HOST_FILE="$3"
RELAY_PID_FILE="$4"
RELAY_LOG_FILE="$5"
RELAY_PORT="$6"
REVERSE_PORT="$7"
RELAY_PYTHON="$8"

cd "${REMOTE_ROOT}"
UAN_HOST="$(hostname -f)"
printf '%s\n' "${UAN_HOST}" > "${RELAY_HOST_FILE}"

if [ -f "${RELAY_PID_FILE}" ]; then
  OLD_PID="$(cat "${RELAY_PID_FILE}" 2>/dev/null || true)"
  case "${OLD_PID}" in
    ''|*[!0-9]*) ;;
    *) kill "${OLD_PID}" 2>/dev/null || true ;;
  esac
fi

"${RELAY_PYTHON}" "${RELAY_SCRIPT}" \
  --listen-host 0.0.0.0 \
  --listen-port "${RELAY_PORT}" \
  --target-host 127.0.0.1 \
  --target-port "${REVERSE_PORT}" \
  > "${RELAY_LOG_FILE}" 2>&1 &
RELAY_PID="$!"
printf '%s\n' "${RELAY_PID}" > "${RELAY_PID_FILE}"

cleanup_remote() {
  kill "${RELAY_PID}" 2>/dev/null || true
}
trap cleanup_remote EXIT

deadline=$((SECONDS + 45))
while ! curl -fsS "http://${UAN_HOST}:${RELAY_PORT}/v1/models" >/dev/null; do
  if ! kill -0 "${RELAY_PID}" 2>/dev/null; then
    echo "UAN relay process exited before readiness. Last relay log lines:" >&2
    tail -n 80 "${RELAY_LOG_FILE}" >&2 || true
    exit 1
  fi
  if [ "${SECONDS}" -gt "${deadline}" ]; then
    echo "UAN relay did not become ready. Last relay log lines:" >&2
    tail -n 80 "${RELAY_LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "UAN_RELAY_HOST=${UAN_HOST}"
echo "UAN relay ready at http://${UAN_HOST}:${RELAY_PORT}/argoapi/v1"

while true; do
  sleep 3600
done
