#!/usr/bin/env python
"""MolNexTR OCSR worker, run INSIDE the MolNexTR conda env (py3.8 + torch).

Loads the MolNexTR model once, then serves image->SMILES requests over the line
protocol in ``_protocol.py``. Launched as a subprocess by
``chemgraph.tools.ocsr_worker_client``; never imported by ChemGraph itself.

    python molnextr_infer.py --weights /path/molnextr_best.pth [--device auto]

Model API (per the MolNexTR repo, https://github.com/CYF2000127/MolNexTR):
    from MolNexTR import molnextr
    model = molnextr(weights_path, device)
    res = model.predict_final_results(image_path)  # dict with "predicted_smiles"

The exact result key can vary by repo revision; we probe a few likely keys and
fall back to the module-level ``MolNexTR.get_predictions`` helper if present.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as P  # noqa: E402

# SMILES key candidates seen across MolNexTR revisions / helpers.
_SMILES_KEYS = ("predicted_smiles", "smiles", "pred_smiles", "SMILES")


def _resolve_device(want: str):
    """Return (torch.device, label). MolNexTR's constructor wants a torch.device.

    Honor the requested device with a safe CPU fallback (Aurora GPUs are Intel,
    not CUDA).
    """
    import torch

    want = (want or "auto").lower()
    if want == "cpu":
        return torch.device("cpu"), "cpu"
    if want in ("cuda", "gpu") and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if want in ("xpu", "auto") and getattr(torch, "xpu", None) is not None:
        try:
            if torch.xpu.is_available():  # type: ignore[attr-defined]
                return torch.device("xpu"), "xpu"
        except Exception:
            pass
    if want == "auto" and torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def _extract_smiles(res):
    """Pull a SMILES string out of MolNexTR's prediction result (dict/str/tuple)."""
    if res is None:
        return None
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        for k in _SMILES_KEYS:
            if k in res and res[k]:
                return res[k]
        return None
    if isinstance(res, (list, tuple)):
        for elem in res:
            smi = _extract_smiles(elem)
            if smi:
                return smi
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    start = time.monotonic()
    with P.redirect_c_stdout_to_stderr():
        import torch  # noqa: F401
        # The top-level ``MolNexTR.molnextr`` is a SUBMODULE; the model class of
        # the same name lives inside it. Import the class explicitly.
        from MolNexTR.molnextr import molnextr as MolNexTRModel

        device, backend = _resolve_device(args.device)
        # Constructor signature: molnextr(model_path, device=torch.device(...)).
        model = MolNexTRModel(args.weights, device=device)

    P.emit_ready("molnextr", backend, time.monotonic() - start)

    def infer(image_path: str):
        res = model.predict_final_results(image_path)
        return _extract_smiles(res)

    return P.serve("molnextr", infer)


if __name__ == "__main__":
    raise SystemExit(main())
