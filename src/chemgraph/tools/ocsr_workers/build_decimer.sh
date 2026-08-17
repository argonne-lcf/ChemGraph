#!/bin/bash
# Build the DECIMER conda env + pre-fetch Zenodo weights (Aurora UAN or any host with conda).
# DECIMER is TensorFlow-based (py3.10). CPU by default (Aurora GPUs are Intel).
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

ENV=$HOME/ocsr-decimer
PY=$ENV/bin/python

# An interrupted build leaves a half-populated env that the "is python present?"
# guard below would then treat as complete, so the next run skips straight to
# inference and fails somewhere far less legible. Remove it on failure, but only if
# this run is the one that created it.
_ocsr_created_env=0
_ocsr_build_cleanup() {
  if [ "$_ocsr_created_env" = "1" ]; then
    echo "[decimer] build interrupted; removing the incomplete env $ENV" >&2
    rm -rf "$ENV"
  fi
}
trap _ocsr_build_cleanup ERR INT TERM

echo "[decimer] === create env (py3.10) ==="
if [ ! -x "$PY" ]; then
  _ocsr_created_env=1
  conda create -p "$ENV" python=3.10 -y
fi

echo "[decimer] === pip install decimer (pulls tensorflow) ==="
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet decimer

echo "[decimer] === warm-up: pre-download Zenodo weights ==="
# First predict_SMILES() downloads the model into ~/.data/DECIMER-V2. Do it here
# on the UAN node (with proxy) so the benchmark never pays the download.
# Render a trivial molecule image to feed the warm-up call.
"$PY" - <<'PYEOF'
import os, tempfile
# Make a tiny PNG with PIL (DECIMER pulls pillow); content need not be a real molecule
# for the download to happen, but a blank image is enough to trigger model build.
from PIL import Image
img = Image.new("RGB", (224, 224), "white")
fd, path = tempfile.mkstemp(suffix=".png"); os.close(fd)
img.save(path)
print("warming up DECIMER (downloads weights on first call)...")
from DECIMER import predict_SMILES
smi = predict_SMILES(path)
print("warmup prediction:", repr(smi))
os.unlink(path)
PYEOF

echo "[decimer] === verify ==="
ls -la "$HOME/.data/DECIMER-V2" 2>/dev/null | head || echo "[decimer] NOTE: weights dir not at ~/.data/DECIMER-V2 (check DECIMER default cache)"
echo "[decimer] DONE"

trap - ERR INT TERM
