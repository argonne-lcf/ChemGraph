#!/usr/bin/env bash
# Run Parsl + EnsembleLauncher demo (5-molecule thermo screen, MACE on CPU)
# on a Crux compute node.
#
# Must be executed INSIDE an interactive PBS allocation on Crux:
#   qsub -I -A <proj> -l select=1 -l walltime=01:00:00 -q debug
#   cd /lus/eagle/projects/ChemGraph/thang/ChemGraph
#   bash scripts/demo/run_crux_demo.sh                  # both backends (MACE thermo)
#   bash scripts/demo/run_crux_demo.sh --parsl-only
#   bash scripts/demo/run_crux_demo.sh --el-only
#   bash scripts/demo/run_crux_demo.sh --molecules water methane
#   bash scripts/demo/run_crux_demo.sh --workload fairchem
#   bash scripts/demo/run_crux_demo.sh --workload pacmof2 --pacmof2-cifs a.cif b.cif

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

abort() {
    echo "[ABORT] $*" >&2
    exit 2
}

RUN_PARSL=1
RUN_EL=1
PASSTHROUGH=()
while (( $# )); do
    case "$1" in
        --parsl-only) RUN_EL=0; shift ;;
        --el-only) RUN_PARSL=0; shift ;;
        --molecules) shift; MOLS=(); while (( $# )) && [[ "$1" != --* ]]; do MOLS+=("$1"); shift; done; PASSTHROUGH+=(--molecules "${MOLS[@]}") ;;
        --pacmof2-cifs) shift; CIFS=(); while (( $# )) && [[ "$1" != --* ]]; do CIFS+=("$1"); shift; done; PASSTHROUGH+=(--pacmof2-cifs "${CIFS[@]}") ;;
        --workload) PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --model-name) PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --net-charge) PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        --timeout) PASSTHROUGH+=("$1" "$2"); shift 2 ;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) abort "Unknown argument: $1" ;;
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
RUN_DIR="${PBS_O_WORKDIR:-$PWD}/parsl_demo_runs_crux"
PARSL_OUT="${PBS_O_WORKDIR:-$PWD}/demo_parsl_out_crux"
EL_OUT="${PBS_O_WORKDIR:-$PWD}/demo_el_out_crux"
mkdir -p "$RUN_DIR" "$PARSL_OUT" "$EL_OUT"

echo "REPO_ROOT=$REPO_ROOT"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-<none>}"
echo "PBS_NODEFILE=$PBS_NODEFILE  ($(wc -l <"$PBS_NODEFILE") node(s))"
echo "RUN_DIR=$RUN_DIR"
echo "PARSL_OUT=$PARSL_OUT  EL_OUT=$EL_OUT"
echo

parsl_rc=0
el_rc=0

if (( RUN_PARSL )); then
    echo "=== Parsl demo (system=crux, device=cpu) ==="
    python "$REPO_ROOT/scripts/demo/demo_parsl_in_job_direct.py" \
        --system crux --device cpu --run-dir "$RUN_DIR" \
        --output-dir "$PARSL_OUT" "${PASSTHROUGH[@]}" \
        || parsl_rc=$?
    echo
fi

if (( RUN_EL )); then
    echo "=== EnsembleLauncher demo (managed, system=crux, device=cpu) ==="
    python "$REPO_ROOT/scripts/demo/demo_ensemble_launcher_in_job_direct.py" \
        --system crux --device cpu \
        --output-dir "$EL_OUT" "${PASSTHROUGH[@]}" \
        || el_rc=$?
    echo
fi

verdict() { (( $1 == 0 )) && echo PASS || echo "FAIL(rc=$1)"; }
echo "=== Summary ==="
(( RUN_PARSL )) && echo "parsl = $(verdict $parsl_rc)  (output: $PARSL_OUT)"
(( RUN_EL ))    && echo "el    = $(verdict $el_rc)  (output: $EL_OUT)"

(( parsl_rc > el_rc )) && exit "$parsl_rc" || exit "$el_rc"
