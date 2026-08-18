#!/bin/bash
# Build the OCSRGlyph conda env + download weights (Aurora UAN or any host with conda).
# Self-contained: loads the module chain + proxy every run (module state does
# not persist across shells). Idempotent: skips env create, clone, and the
# weights download if they already exist.
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
# Keep ~/.local/lib/pythonX.Y/site-packages off sys.path so a stray user-site
# install can never shadow this env's torch.
export PYTHONNOUSERSITE=1

ENV=$HOME/ocsr-glyph
SRC=$HOME/opt/glyph
WDIR=$HOME/ocsr-weights/ocsrglyph
PY=$ENV/bin/python

# An interrupted build leaves a half-populated env that the "is python present?"
# guard below would then treat as complete, so the next run skips straight to
# inference and fails somewhere far less legible. Remove it on failure, but only if
# this run is the one that created it.
_ocsr_created_env=0
_ocsr_build_cleanup() {
  if [ "$_ocsr_created_env" = "1" ]; then
    echo "[ocsrglyph] build interrupted; removing the incomplete env $ENV" >&2
    rm -rf "$ENV"
  fi
}
trap _ocsr_build_cleanup ERR INT TERM

echo "[ocsrglyph] === create env (py3.11) ==="
# py3.11: glyph's pyproject sets requires-python >=3.11.
if [ ! -x "$PY" ]; then
  _ocsr_created_env=1
  conda create -p "$ENV" python=3.11 -y
fi
"$PY" -m pip install --quiet --upgrade pip

echo "[ocsrglyph] === pip install CPU torch + torchvision (ordering matters) ==="
# CRITICAL ORDERING: torch AND torchvision together from the CPU index FIRST.
# timm depends on torchvision; if torchvision is absent when timm installs, pip
# pulls it from PyPI and drags in torch 2.13+cu130 plus ~15 nvidia-* CUDA
# packages (verified failure mode, not hypothetical). Aurora GPUs are Intel, so
# a CUDA torch is pure bloat here.
"$PY" -m pip install --quiet torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu

echo "[ocsrglyph] === pip install glyph runtime deps ==="
"$PY" -m pip install --quiet "timm>=1.0" "rdkit>=2024.3.1" "SmilesPE>=0.0.3" \
    "pyarrow>=15.0" "Pillow>=10.3" "PyYAML>=6.0" "tqdm>=4.66" \
    "huggingface-hub>=0.24" "typer>=0.20"

echo "[ocsrglyph] === clone + install glyph (pinned commit) ==="
if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/EdisonScientific/glyph "$SRC"
fi
git -C "$SRC" checkout -q 0bf782f863d26b041ace157668928ef07c38b972
# --no-deps: deps are installed above with the CPU torch already pinned in
# place; letting pip resolve the pyproject here would re-open the CUDA trap.
"$PY" -m pip install --quiet --no-deps -e "$SRC"

echo "[ocsrglyph] === guard: no CUDA packages ==="
# Guard: the ordering trap above must not have fired.
if "$PY" -m pip list 2>/dev/null | grep -qi nvidia; then
  echo "ERROR: nvidia-* packages present; torch install order was wrong" >&2
  exit 1
fi

echo "[ocsrglyph] === download weights (357 MB) ==="
mkdir -p "$WDIR"
if [ -f "$WDIR/model.pth" ]; then
  echo "[ocsrglyph] weights already present -> $WDIR/model.pth (skipping)"
else
  "$PY" - <<PYEOF
from huggingface_hub import hf_hub_download
p = hf_hub_download('EdisonScientific/OCSRGlyph', 'model.pth', local_dir='$WDIR')
print('weights ->', p)
PYEOF
fi

echo "[ocsrglyph] === verify import ==="
"$PY" -c "import torch, timm; from glyph.ocsr.predict import OCSRPredictor; print('OK', torch.__version__, timm.__version__)"
echo "[ocsrglyph] DONE"

trap - ERR INT TERM
