"""Restore torch.load's pre-2.6 weights_only=False default.

Why this exists
---------------
PyTorch 2.6 flipped ``torch.load``'s ``weights_only`` default to ``True``
as a security hardening (arbitrary pickle in a downloaded checkpoint =
arbitrary code execution). Pure-tensor checkpoints load fine under the
new default; checkpoints that pickle Python objects raise
``UnpicklingError: Weights only load failed``.

Every MLIP shipper we depend on still pickles Python objects in their
released checkpoints:

* ``fairchem-core`` (UMA) -- config dicts + custom module classes.
* ``mace-torch`` (MACE) -- same shape.

The MLIP shippers are the only ones who can fix this upstream by either
(a) re-releasing checkpoints as safetensors, or (b) calling
``torch.load(..., weights_only=False)`` explicitly at every load site.
Neither has landed at time of writing, so every downstream that touches
these checkpoints needs the same shim. This is not something PyTorch
will revert -- the new default is intentional.

Remove this file when
---------------------
Both of the following hold:

* ``fairchem-core`` loads UMA checkpoints without needing
  ``weights_only=False`` (safetensors packaging or explicit kwarg).
* ``mace-torch`` does the same for MACE.

Delivery
--------
Kept in its own tiny module so MCP subprocess launchers can prepend it
via ``python -c "import chemgraph.academy.runtime.torch_patch; ..."``
without pulling in the rest of the academy runtime. See
``mcp_supervisor._wrap_with_torch_patch`` for the caller.
"""

from __future__ import annotations


def _patch() -> None:
    try:
        import torch
    except ImportError:
        return
    if getattr(torch.load, "__swarm_patched__", False):
        return
    _orig = torch.load

    def _patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    _patched.__swarm_patched__ = True  # type: ignore[attr-defined]
    torch.load = _patched  # type: ignore[assignment]


_patch()
