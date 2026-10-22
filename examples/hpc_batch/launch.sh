#!/bin/bash
set -euo pipefail
: "${PBS_JOBID:?Run this script as a PBS job}"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

{
    printf 'pbs_job_id=%s\n' "$PBS_JOBID"
    printf 'hostname=%s\n' "$(hostname)"
    printf 'input_sha256=%s\n' "$(sha256sum inputs/numbers.txt | cut -d ' ' -f 1)"
    awk '{ total += $1 } END { printf "sum=%d\n", total }' inputs/numbers.txt
} > result.txt
cat result.txt
