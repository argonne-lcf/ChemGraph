"""Restore torch.load's pre-2.6 weights_only=False default.

Imported at the top of any process that may torch.load a pickled MLIP
checkpoint (fairchem/UMA/MACE ship pickle, not the safe tensor-only
format that torch>=2.6 requires by default).

Kept in its own tiny module so MCP subprocess launchers can prepend it
via ``python -c "import chemgraph.academy.runtime.torch_patch; ..."`` without pulling
in the rest of swarm.runtime.
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
