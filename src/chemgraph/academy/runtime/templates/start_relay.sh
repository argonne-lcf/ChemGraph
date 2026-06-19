#!/bin/bash
set -uo pipefail

REMOTE_ROOT="$1"
RELAY_SCRIPT="$2"
RELAY_HOST_FILE="$3"
RELAY_PID_FILE="$4"
RELAY_LOG_FILE="$5"
RELAY_PORT="$6"
REVERSE_PORT="$7"
RELAY_PYTHON="$8"

# Trace every line so the local relay log captures exactly which step
# fails. Without xtrace, a bash failure (e.g. cd, redirect) produces
# zero diagnostics and the launcher just reports "Local relay log:
# <empty>". The output goes through the SSH stdout pipe so it lands
# in the Mac-side relay log -- no remote-side tail needed.
exec 2>&1
echo "[start_relay] uan=$(hostname -f) port=${RELAY_PORT} reverse=${REVERSE_PORT} pid=$$"
set -x

cd "${REMOTE_ROOT}"
UAN_HOST="$(hostname -f)"
UAN_SHORT="$(hostname -s)"
printf '%s\n' "${UAN_HOST}" > "${RELAY_HOST_FILE}"

# Pid bookkeeping has to be per-UAN. Aurora's login alias round-robins
# across uan-0001..uan-0010, the shared filesystem is the same, but a
# pid only means something on the UAN that holds the process. Without
# per-host scoping, the launcher would happily kill the wrong pid (or
# none at all) on a sibling UAN and leave an orphan holding the port.
PER_HOST_PID_FILE="${RELAY_PID_FILE%.pid}.${UAN_SHORT}.pid"

find_orphan_pids() {
  # Match python processes whose first argv after the interpreter is
  # the relay script path. Using `comm=python` + an argv prefix is far
  # more precise than `pgrep -f <path>`, which would also match this
  # very bash script (because the path is in OUR argv too) and the
  # subsequent `pgrep` invocation itself -- killing them all and
  # taking the whole start_relay session down with them. That bug
  # produced the silent "log shows kill <self-pid> then nothing"
  # failure mode.
  local self_pid="$$"
  local parent_pid="${PPID:-0}"
  ps -u "${USER}" -o pid=,comm=,args= 2>/dev/null \
    | awk -v rs="${RELAY_SCRIPT}" -v me="${self_pid}" -v pp="${parent_pid}" '
        $1 == me || $1 == pp { next }
        $2 ~ /python/ {
          for (i = 3; i <= NF; i++) if ($i == rs) { print $1; next }
        }
      '
}

kill_local_orphans() {
  # Kill prior relay processes on THIS UAN. Scope: only python
  # processes that have the relay script as an argv element, owned
  # by us, excluding our own pid/ppid.
  local pids
  pids="$(find_orphan_pids)"
  if [ -n "${pids}" ]; then
    echo "[start_relay] killing prior relay pids on $(hostname -s): ${pids}"
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
    sleep 1
    pids="$(find_orphan_pids)"
    if [ -n "${pids}" ]; then
      echo "[start_relay] forcing kill -9 on stubborn pids: ${pids}"
      # shellcheck disable=SC2086
      kill -9 ${pids} 2>/dev/null || true
      sleep 1
    fi
  fi
  # Also try the per-host pid file in case the process was renamed or
  # something matched a previous launch's bookkeeping that the ps
  # scan didn't see. Best-effort.
  if [ -f "${PER_HOST_PID_FILE}" ]; then
    local old_pid
    old_pid="$(cat "${PER_HOST_PID_FILE}" 2>/dev/null || true)"
    case "${old_pid}" in
      ''|*[!0-9]*) ;;
      *)
        # Don't kill ourselves or our parent even if a stale file
        # happens to record our pid (shouldn't happen, but cheap).
        if [ "${old_pid}" != "$$" ] && [ "${old_pid}" != "${PPID:-0}" ]; then
          kill "${old_pid}" 2>/dev/null || true
        fi
        ;;
    esac
  fi
}

kill_local_orphans

# After reaping local orphans, fail fast and clearly if the port is
# still held -- it means another user (or another UAN's process via
# some unusual route) owns it and we can't take it over.
if command -v ss >/dev/null 2>&1; then
  if ss -tln 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${RELAY_PORT}\$"; then
    echo "ERROR: port ${RELAY_PORT} on ${UAN_SHORT} is still in use after" >&2
    echo "       cleaning up our own relays. Inspect with: ss -tlnp | grep ${RELAY_PORT}" >&2
    ss -tlnp 2>/dev/null | grep -E "[:.]${RELAY_PORT}\\b" >&2 || true
    exit 1
  fi
fi

"${RELAY_PYTHON}" "${RELAY_SCRIPT}" \
  --listen-host 0.0.0.0 \
  --listen-port "${RELAY_PORT}" \
  --target-host 127.0.0.1 \
  --target-port "${REVERSE_PORT}" \
  > "${RELAY_LOG_FILE}" 2>&1 &
RELAY_PID="$!"
printf '%s\n' "${RELAY_PID}" > "${PER_HOST_PID_FILE}"
# Also write the legacy pid path so older launchers / debugging scripts
# that look for the bare uan-relay-<port>.pid see *something* sensible.
printf '%s\n' "${RELAY_PID}" > "${RELAY_PID_FILE}"

cleanup_remote() {
  kill "${RELAY_PID}" 2>/dev/null || true
  rm -f "${PER_HOST_PID_FILE}" 2>/dev/null || true
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
