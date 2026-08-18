#!/usr/bin/env python
"""MolScribe OCSR worker, run INSIDE the MolScribe conda env.

Loads the MolScribe model once, then serves image->SMILES requests over the
line protocol in ``_protocol.py``. Never imported by the bench; launched as a
subprocess by ``chemgraph.tools.ocsr_worker_client``.

    python molscribe_infer.py --weights /path/swin_base_char_aux_1m.pth [--device auto]

Model API (per the MolScribe repo):
    from molscribe import MolScribe
    model = MolScribe(ckpt_path, device=torch.device(dev))
    out = model.predict_image_file(path)  # -> {"smiles": ..., ...}
"""

from __future__ import annotations

import argparse
import sys
import time

# _protocol lives next to this file; ensure it's importable regardless of cwd.
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import _protocol as P  # noqa: E402


def _resolve_device(want: str):
    """Return a torch.device honoring the requested device with safe fallback."""
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
    args = ap.parse_args()

    start = time.monotonic()
    with P.redirect_c_stdout_to_stderr():
        import torch  # noqa: F401
        from molscribe import MolScribe

        device, backend = _resolve_device(args.device)
        model = MolScribe(args.weights, device=device)
    P.emit_ready("molscribe", backend, time.monotonic() - start)

    def infer(image_path: str):
        out = model.predict_image_file(image_path)
        # predict_image_file returns a dict; smiles under "smiles".
        return (out or {}).get("smiles")

    return P.serve("molscribe", infer)


if __name__ == "__main__":
    raise SystemExit(main())
