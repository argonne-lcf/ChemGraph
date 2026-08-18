#!/usr/bin/env python
"""OCSRGlyph OCSR worker: runs INSIDE the OCSRGlyph conda env (py3.11 + torch).

Loads the OCSRGlyph model once, then serves image->SMILES requests over the
line protocol in ``_protocol.py``. Never imported by the bench; launched as a
subprocess by ``chemgraph.tools.ocsr_worker_client``.

    python ocsrglyph_infer.py --weights /path/model.pth [--device auto] [--threads 16]

Model API (per the glyph repo, https://github.com/EdisonScientific/glyph):
    from glyph.ocsr.predict import OCSRPredictor
    model = OCSRPredictor(weights_path, device="cpu")
    smiles = model.predict(image_path)

Two differences from the sibling workers, both verified on this node:
  * ``device`` is a plain STRING ("cpu"/"cuda"/"xpu"), not a torch.device, so we
    reuse the usual resolver and hand over ``device.type``.
  * ``predict`` returns a BARE SMILES STRING, not a dict, so there is no key to
    dig out (contrast MolScribe's ``{"smiles": ...}``).

``predict`` defaults to ``postprocess=True`` and we deliberately leave it there:
that default already applies exactly the paper's postprocessing (drop isolated
``[H]``/``[HH]`` fragments without touching ``[nH]``/``[NH3+]``, then RDKit
re-canonicalize, falling back to the raw string if it will not parse). Keeping
the default is what makes our numbers comparable to the published ones; do not
reimplement it here and do not pass postprocess=False.

Threading: nothing in the bench sets OMP_NUM_THREADS for workers, so the thread
count is ours to pick. 16 measured fastest on this node (32 was ~25% slower).
Inference is strictly one image per call: ``predict_batch`` measured 3x SLOWER
per image, because greedy decoding pads every sequence to the batch's longest.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as P  # noqa: E402


def _resolve_device(want: str):
    """Return (torch.device, label) honoring the request with a CPU fallback."""
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=16,
                    help="torch intra-op threads (16 measured optimal on Aurora CPU)")
    args = ap.parse_args()

    start = time.monotonic()
    with P.redirect_c_stdout_to_stderr():
        import torch
        from glyph.ocsr.predict import OCSRPredictor

        if args.threads > 0:
            torch.set_num_threads(args.threads)

        device, backend = _resolve_device(args.device)
        # Constructor wants the device as a str, hence ``device.type``.
        model = OCSRPredictor(args.weights, device=device.type)
    P.emit_ready("ocsrglyph", backend, time.monotonic() - start)

    def infer(image_path: str):
        # Returns the SMILES string directly; postprocess=True stays default.
        # Image prep (RGB convert, 384x384 bilinear, normalize) is internal, and
        # RGBA vs RGB inputs were verified to predict identically, so our PNGs
        # need no conversion here.
        return model.predict(image_path)

    return P.serve("ocsrglyph", infer)


if __name__ == "__main__":
    raise SystemExit(main())
