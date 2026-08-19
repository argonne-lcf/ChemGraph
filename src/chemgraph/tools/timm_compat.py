"""Let MolNexTR and MolScribe run on timm 1.x instead of their pinned 0.4.12.

Both vendor a Swin Transformer written against timm 0.4.12 internals, and the two
copies share 98% of their lines, so they break on exactly the same imports:

    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg
    from timm.models.layers import Mlp, DropPath, to_2tuple, trunc_normal_
    from timm.models.registry import register_model
    from timm.models.vision_transformer import checkpoint_filter_fn, _init_vit_weights

Call :func:`install` before importing either package and those paths resolve again,
which leaves both projects' own source untouched. One shim covers both, and it is
what lets all four specialists share ChemGraph's environment.

Where each piece went in timm 1.x:

===============================================  ==============================
``timm.data``                                    unchanged
``models.helpers.build_model_with_cfg``          moved to ``models._builder``;
                                                 ``default_cfg=`` renamed
                                                 ``pretrained_cfg=``
``models.helpers.overlay_external_default_cfg``  removed, reimplemented here
``models.layers``                                moved to ``timm.layers``
``models.registry``                              moved to ``models._registry``
``vision_transformer.checkpoint_filter_fn``      unchanged
``vision_transformer._init_vit_weights``         split into
                                                 ``init_weights_vit_{jax,timm}``
===============================================  ==============================

Four of those are renames and two need real code, so the 0.4.12 pins were stale
instead of load-bearing.

Verified on timm 1.0.28 with torch 2.13.0 and numpy 2.2.6: MolNexTR reproduces its
timm 0.4.12 predictions exactly over 60 PubChem eval images, MolScribe returns the
expected SMILES, and all four specialists import and infer in a single process.

On timm 0.4.12 itself :func:`install` is a no-op, so callers can invoke it
unconditionally.
"""

from __future__ import annotations

import sys
import types

import timm

# timm 1.x private module layout. Absent on 0.4.12, where the old public paths this
# module restores are the ones that already exist, so there is nothing to shim.
try:
    import timm.layers as _layers
    import timm.models._builder as _builder
    import timm.models._registry as _registry
    import timm.models.vision_transformer as _vit

    _NEEDED = True
except ImportError:  # pragma: no cover - only on timm < 1.0
    _NEEDED = False


def overlay_external_default_cfg(default_cfg: dict, kwargs: dict) -> None:
    """Fold ``kwargs['external_default_cfg']`` into ``default_cfg``, as 0.4.12 did.

    Mutates both arguments: the key is popped from ``kwargs`` and the config is
    updated in place, dropping any ``url``/``hf_hub`` the caller is overriding.
    """
    external = kwargs.pop("external_default_cfg", None)
    if external:
        default_cfg.pop("url", None)
        default_cfg.pop("hf_hub", None)
        default_cfg.update(external)


def build_model_with_cfg(model_cls, variant, pretrained, default_cfg=None, **kwargs):
    """Forward to timm 1.x, translating the ``default_cfg`` to ``pretrained_cfg`` rename."""
    if default_cfg is not None:
        kwargs["pretrained_cfg"] = default_cfg
    return _builder.build_model_with_cfg(model_cls, variant, pretrained, **kwargs)


def _init_vit_weights(module, name: str = "", head_bias: float = 0.0, jax_impl: bool = False):
    """Reunite timm 1.x's two initialisers behind the single 0.4.12 entry point."""
    if jax_impl:
        _vit.init_weights_vit_jax(module, name, head_bias=head_bias)
    else:
        _vit.init_weights_vit_timm(module, name)


def install() -> bool:
    """Restore the 0.4.12 import paths. Returns True if anything was patched.

    Idempotent, and a no-op on timm 0.4.12 where the old paths already resolve.
    """
    if not _NEEDED:
        return False

    helpers = sys.modules.get("timm.models.helpers")
    if helpers is None:
        helpers = types.ModuleType("timm.models.helpers")
        sys.modules["timm.models.helpers"] = helpers
        timm.models.helpers = helpers
    helpers.overlay_external_default_cfg = overlay_external_default_cfg
    helpers.build_model_with_cfg = build_model_with_cfg

    sys.modules["timm.models.registry"] = _registry
    timm.models.registry = _registry

    sys.modules["timm.models.layers"] = _layers
    timm.models.layers = _layers

    if not hasattr(_vit, "_init_vit_weights"):
        _vit._init_vit_weights = _init_vit_weights

    return True
