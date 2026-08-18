#!/bin/bash
# Build the MolScribe conda env + download weights (Aurora UAN or any host with conda).
# Self-contained: loads the module chain + proxy every run (module state does
# not persist across shells). Idempotent-ish: skips env create if it exists.
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

ENV=$HOME/ocsr-molscribe
WDIR=$HOME/ocsr-weights/molscribe
PY=$ENV/bin/python

# An interrupted build leaves a half-populated env that the "is python present?"
# guard below would then treat as complete, so the next run skips straight to
# inference and fails somewhere far less legible. Remove it on failure, but only if
# this run is the one that created it.
_ocsr_created_env=0
_ocsr_build_cleanup() {
  if [ "$_ocsr_created_env" = "1" ]; then
    echo "[molscribe] build interrupted; removing the incomplete env $ENV" >&2
    rm -rf "$ENV"
  fi
}
trap _ocsr_build_cleanup ERR INT TERM

echo "[molscribe] === create env (py3.10) ==="
if [ ! -x "$PY" ]; then
  _ocsr_created_env=1
  conda create -p "$ENV" python=3.10 -y
fi

echo "[molscribe] === pip install (torch + MolScribe) ==="
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet MolScribe
# MolScribe pulls a CUDA torch build; it runs on CPU fine (cuda.is_available()=False
# on Aurora's Intel GPUs). huggingface_hub is needed for the weights download but
# is not a MolScribe dependency, so add it explicitly.
"$PY" -m pip install --quiet huggingface_hub
# CRITICAL: torch/rdkit here are compiled against numpy 1.x. numpy 2.x makes
# torch.from_numpy raise "Numpy is not available" at inference time. opencv>=4.11
# drags numpy 2 back in, so pin opencv 4.10 (last that accepts numpy<2) and force
# numpy 1.26. Order matters: this MUST be the last pip step.
"$PY" -m pip install --quiet "opencv-python==4.10.0.84" "opencv-python-headless==4.10.0.84" "numpy==1.26.4"

echo "[molscribe] === download weights ==="
mkdir -p "$WDIR"
"$PY" - <<PYEOF
from huggingface_hub import hf_hub_download
p = hf_hub_download('yujieq/MolScribe', 'swin_base_char_aux_1m.pth', local_dir='$WDIR')
print('weights ->', p)
PYEOF

echo "[molscribe] === verify import ==="
"$PY" -c "import torch, molscribe; print('molscribe import OK, torch', torch.__version__)"
echo "[molscribe] DONE"

trap - ERR INT TERM
