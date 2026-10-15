#!/bin/bash
# Invoke with bash: staged assets need not have executable permissions.
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: bash pbs-launch.sh COMMAND [ARGUMENTS...]" >&2
    exit 1
fi
if [[ -z "${PBS_JOBID:-}" || -z "${PBS_NODEFILE:-}" ||
      ! -f "$PBS_NODEFILE" || ! -r "$PBS_NODEFILE" ]]; then
    echo "A PBS job and readable allocation nodefile are required." >&2
    exit 1
fi

host=$(hostname)
host=${host%%.*}
allocated=false
while read -r -a nodes || [[ ${#nodes[@]} -gt 0 ]]; do
    for node in "${nodes[@]:-}"; do
        if [[ "${node%%.*}" == "$host" ]]; then
            allocated=true
        fi
    done
done < "$PBS_NODEFILE"
if [[ "$allocated" != true ]]; then
    echo "This host is not in the PBS allocation." >&2
    exit 1
fi

printf 'PBS allocation: job=%s host=%s\n' "$PBS_JOBID" "$host" >&2
exec "$@"
