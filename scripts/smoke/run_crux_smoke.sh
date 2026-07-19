#!/usr/bin/env bash
# Run Parsl + EnsembleLauncher smoke tests on a Crux compute node (MACE on CPU).
#
# Must be executed INSIDE an interactive PBS allocation on Crux:
#   qsub -I -A <proj> -l select=1 -l walltime=00:30:00 -q debug
#   cd /lus/eagle/projects/ChemGraph/thang/ChemGraph
#   bash scripts/smoke/run_crux_smoke.sh             # both backends + MACE
#   bash scripts/smoke/run_crux_smoke.sh --quick     # skip MACE
#   bash scripts/smoke/run_crux_smoke.sh --parsl-only
#   bash scripts/smoke/run_crux_smoke.sh --el-only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

abort() {
    echo "[ABORT] $*" >&2
    exit 2
}

QUICK=""
RUN_PARSL=1
RUN_EL=1
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK="--quick" ;;
        --parsl-only) RUN_EL=0 ;;
        --el-only) RUN_PARSL=0 ;;
        -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) abort "Unknown argument: $arg" ;;
    esac
done

[[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE}" ]] \
    || abort "PBS_NODEFILE not set or missing -- run inside 'qsub -I' on Crux."

VENV_ACTIVATE="$REPO_ROOT/.cg_crux_hpc/bin/activate"
[[ -f "$VENV_ACTIVATE" ]] || abort "Missing venv activate script: $VENV_ACTIVATE"

if [[ "${VIRTUAL_ENV:-}" != "$REPO_ROOT/.cg_crux_hpc" ]]; then
    module load conda 2>/dev/null || true
    # shellcheck disable=SC1090
    source "$VENV_ACTIVATE"
fi

export COMPUTE_SYSTEM=crux
RUN_DIR="${PBS_O_WORKDIR:-$PWD}/parsl_runs_smoke_crux"
mkdir -p "$RUN_DIR"

echo "REPO_ROOT=$REPO_ROOT"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-<none>}"
echo "PBS_NODEFILE=$PBS_NODEFILE  ($(wc -l <"$PBS_NODEFILE") node(s))"
echo "RUN_DIR=$RUN_DIR"
echo

parsl_rc=0
el_rc=0

if (( RUN_PARSL )); then
    echo "=== Parsl smoke (system=crux, device=cpu) ==="
    python "$REPO_ROOT/scripts/smoke/smoke_parsl_in_job.py" \
        --system crux --device cpu --run-dir "$RUN_DIR" $QUICK \
        || parsl_rc=$?
    echo
fi

if (( RUN_EL )); then
    echo "=== EnsembleLauncher smoke (managed, system=crux, device=cpu) ==="
    python "$REPO_ROOT/scripts/smoke/smoke_ensemble_launcher_in_job.py" \
        --mode managed --system crux --device cpu $QUICK \
        || el_rc=$?
    echo
fi

verdict() { (( $1 == 0 )) && echo PASS || echo "FAIL(rc=$1)"; }
echo "=== Summary ==="
(( RUN_PARSL )) && echo "parsl = $(verdict $parsl_rc)"
(( RUN_EL ))    && echo "el    = $(verdict $el_rc)"

(( parsl_rc > el_rc )) && exit "$parsl_rc" || exit "$el_rc"
