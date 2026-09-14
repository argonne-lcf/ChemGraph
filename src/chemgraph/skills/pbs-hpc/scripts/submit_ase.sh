#!/bin/bash
# Direct batch example: run from its staged directory. Keep the marker on failure.
set -euo pipefail
test -f input.json
bash -n job.pbs
if grep -nE '\{\{[A-Z_]+\}\}' job.pbs input.json; then
    echo "Replace all template placeholders before submitting." >&2
    exit 2
fi
if ! (set -C; : > submission.started) 2>/dev/null; then
    echo "Submission already attempted; inspect job.id, qsub.stderr and PBS history." >&2
    exit 2
fi
if ! qsub job.pbs > job.id 2> qsub.stderr; then
    cat qsub.stderr >&2
    echo "Submission outcome unresolved. Inspect PBS before attempting another job." >&2
    exit 1
fi
if ! test -s job.id; then
    echo "qsub returned no job ID. Inspect PBS; do not resubmit automatically." >&2
    exit 1
fi
cat job.id
