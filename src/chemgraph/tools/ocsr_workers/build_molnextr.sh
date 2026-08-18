#!/bin/bash
# Build the MolNexTR conda env + download weights (Aurora UAN or any host with conda).
# MolNexTR pins old deps (OpenNMT-py==2.2.0, timm==0.4.12, albumentations==1.1.0)
# that want py3.8 + an older torch. CPU-only (Aurora GPUs are Intel, not CUDA).
set -eo pipefail

# --- host setup -------------------------------------------------------------
# Written for and tested on an ALCF Aurora UAN node, where conda comes from a
# module and outbound traffic needs the site proxy. Every line is guarded so the
# script also runs on an ordinary machine that already has conda on PATH.
#
# The proxy guard matters more than it looks: proxy.alcf.anl.gov is reachable
# only from inside ANL, so exporting it elsewhere makes every pip and
# HuggingFace fetch HANG rather than fail.
[ -f /usr/share/lmod/lmod/init/bash ] && source /usr/share/lmod/lmod/init/bash >/dev/null 2>&1
if type module >/dev/null 2>&1; then
  module load oneapi/release/2025.3.1 >/dev/null 2>&1 || true
  module load miniforge3/25.11.0-1 >/dev/null 2>&1 || true
fi
case "$(hostname -f 2>/dev/null)" in
  *.alcf.anl.gov)
    export https_proxy=http://proxy.alcf.anl.gov:3128
    export http_proxy=http://proxy.alcf.anl.gov:3128
    export no_proxy=localhost,127.0.0.1
    ;;
esac
command -v conda >/dev/null 2>&1 || {
  echo "conda not found. Install miniforge (https://github.com/conda-forge/miniforge)" >&2
  echo "or load your site's conda module, then re-run this script." >&2
  exit 1
}
# ----------------------------------------------------------------------------

ENV=$HOME/ocsr-molnextr
SRC=$HOME/opt/MolNexTR
WDIR=$HOME/ocsr-weights/molnextr
PY=$ENV/bin/python

# An interrupted build leaves a half-populated env that the "is python present?"
# guard below would then treat as complete, so the next run skips straight to
# inference and fails somewhere far less legible. Remove it on failure, but only if
# this run is the one that created it.
_ocsr_created_env=0
_ocsr_build_cleanup() {
  if [ "$_ocsr_created_env" = "1" ]; then
    echo "[molnextr] build interrupted; removing the incomplete env $ENV" >&2
    rm -rf "$ENV"
  fi
}
trap _ocsr_build_cleanup ERR INT TERM

echo "[molnextr] === create env (py3.8) ==="
if [ ! -x "$PY" ]; then
  _ocsr_created_env=1
  conda create -p "$ENV" python=3.8 -y
fi

echo "[molnextr] === clone repo (pinned commit) ==="
if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/CYF2000127/MolNexTR "$SRC"
fi
# Pinned so a rebuild reproduces the accuracy this model is documented with (83.5%
# exact). An unpinned clone would silently track upstream main, and a changed
# prediction does not raise, it just moves the number.
git -C "$SRC" checkout -q 6f6502b4ed9733dba8b1ee45d2da474576683194

echo "[molnextr] === pip install (CPU torch + pinned deps) ==="
"$PY" -m pip install --quiet --upgrade "pip<24" setuptools wheel
# torch 1.12.1 CPU: last line that plays well with OpenNMT-py 2.2.0 / timm 0.4.12 on py3.8.
"$PY" -m pip install --quiet torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cpu
# Repo requirements (pinned versions inside requirements.txt).
"$PY" -m pip install --quiet -r "$SRC/requirements.txt"
# Install the package itself so `import MolNexTR` works.
"$PY" -m pip install --quiet -e "$SRC" || echo "[molnextr] WARN: pip install -e failed; will import from repo dir via PYTHONPATH"
# huggingface_hub for the weights download (not a repo dep).
"$PY" -m pip install --quiet huggingface_hub
# CRITICAL (same lesson as MolScribe): torch 1.12 + rdkit are numpy-1.x builds;
# numpy 2.x makes torch.from_numpy raise "Numpy is not available". Force numpy<2
# as the LAST pip step so nothing pulls numpy 2 back in.
"$PY" -m pip install --quiet "numpy<2"

echo "[molnextr] === download weights (1.13 GB) ==="
mkdir -p "$WDIR"
"$PY" - <<PYEOF
from huggingface_hub import hf_hub_download
p = hf_hub_download('CYF200127/MolNexTR', 'molnextr_best.pth',
                    repo_type='dataset', local_dir='$WDIR')
print('weights ->', p)
PYEOF

echo "[molnextr] === verify import ==="
PYTHONPATH="$SRC:$PYTHONPATH" "$PY" -c "import torch; import MolNexTR; print('MolNexTR import OK, torch', torch.__version__)" \
  || echo "[molnextr] WARN: import check failed; inspect worker import path"
echo "[molnextr] DONE"

trap - ERR INT TERM
