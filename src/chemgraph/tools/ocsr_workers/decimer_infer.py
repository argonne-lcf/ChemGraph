#!/usr/bin/env python
"""DECIMER OCSR worker, run INSIDE the DECIMER conda env (py3.10 + tensorflow).

Loads the DECIMER model once (first call triggers the Zenodo weight download if
not cached), then serves image->SMILES requests over the line protocol in
``_protocol.py``. Launched as a subprocess by
``chemgraph.tools.ocsr_worker_client``; never imported by ChemGraph itself.

    python decimer_infer.py [--weights /path/DECIMER-V2] [--device auto]

Model API (per the DECIMER repo):
    from DECIMER import predict_SMILES
    smiles = predict_SMILES(image_path)

Weights auto-download to ~/.data/DECIMER-V2 on first use. ``--weights`` is
advisory: if given, it is exported as the DECIMER data dir so a pre-fetched
cache is reused instead of re-downloading (helpful on egress-less compute nodes).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as P  # noqa: E402


def _select_backend(want: str) -> str:
    """Best-effort TF device selection; CPU is the safe default on Aurora.

    TF picks GPUs automatically when a compatible plugin is present. When the
    caller asks for cpu (or no GPU is visible) we hide GPUs so TF stays on CPU.
    Returns the backend label for the ready line.
    """
    want = (want or "auto").lower()
    if want == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "cpu"
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return "gpu"
    except Exception:
        pass
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    # Point DECIMER at a pre-fetched cache if provided (avoids re-download).
    if args.weights:
        parent = os.path.dirname(args.weights.rstrip("/")) or args.weights
        os.environ.setdefault("DECIMER_DATA_DIR", args.weights)
        os.environ.setdefault("XDG_DATA_HOME", parent)

    start = time.monotonic()
    with P.redirect_c_stdout_to_stderr():
        backend = _select_backend(args.device)
        from DECIMER import predict_SMILES

        # Warm up so the (slow) model build happens before we report ready.
        _ = predict_SMILES  # ensure symbol resolved
    P.emit_ready("decimer", backend, time.monotonic() - start)

    def infer(image_path: str):
        return predict_SMILES(image_path)

    return P.serve("decimer", infer)


if __name__ == "__main__":
    raise SystemExit(main())
