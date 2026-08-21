"""Tests for the timm 0.4.12 compatibility shim.

The shim exists so MolNexTR and MolScribe can share ChemGraph's environment instead
of each needing their own. These tests cover the parts that do not require the
models themselves: the import paths it restores, and the two functions it
reimplements. Loading MolNexTR or MolScribe needs their checkpoints, so that check
lives in the OCSR backend tests and is skipped when a model is not installed.
"""

import importlib
import sys

import pytest

timm = pytest.importorskip("timm")

from chemgraph.tools import timm_compat  # noqa: E402


@pytest.fixture(autouse=True)
def _installed():
    """Every test runs against an installed shim; install() is idempotent."""
    timm_compat.install()


def test_install_is_idempotent():
    assert timm_compat.install() == timm_compat.install()


def test_restores_the_paths_molnextr_and_molscribe_import():
    """The five imports both vendored Swin copies open with."""
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD  # noqa: F401
    from timm.models.helpers import (  # noqa: F401
        build_model_with_cfg,
        overlay_external_default_cfg,
    )
    from timm.models.layers import (  # noqa: F401
        DropPath,
        Mlp,
        to_2tuple,
        trunc_normal_,
    )
    from timm.models.registry import register_model  # noqa: F401
    from timm.models.vision_transformer import (  # noqa: F401
        _init_vit_weights,
        checkpoint_filter_fn,
    )


def test_restored_modules_are_the_real_ones():
    """The old names are aliases onto timm's current modules, not empty stubs."""
    layers = importlib.import_module("timm.models.layers")
    registry = importlib.import_module("timm.models.registry")
    assert layers is sys.modules["timm.layers"]
    if timm_compat._NEEDED:
        assert registry is sys.modules["timm.models._registry"]


def test_overlay_external_default_cfg_folds_and_pops():
    cfg = {"num_classes": 1000, "url": "https://example.invalid/weights.pth"}
    kwargs = {"external_default_cfg": {"num_classes": 21841}, "img_size": 384}

    timm_compat.overlay_external_default_cfg(cfg, kwargs)

    assert cfg["num_classes"] == 21841
    assert "url" not in cfg, "an overridden cfg must not keep the old download URL"
    assert "external_default_cfg" not in kwargs, "the key is consumed, not passed on"
    assert kwargs["img_size"] == 384, "unrelated kwargs survive"


def test_overlay_external_default_cfg_without_an_override_changes_nothing():
    cfg = {"num_classes": 1000, "url": "https://example.invalid/weights.pth"}
    before = dict(cfg)

    timm_compat.overlay_external_default_cfg(cfg, {"img_size": 384})

    assert cfg == before


def test_init_vit_weights_reaches_both_timm_initialisers(monkeypatch):
    """0.4.12 had one entry point; 1.x split it in two, keyed by jax_impl."""
    if not timm_compat._NEEDED:
        pytest.skip("timm 0.4.12 keeps its own _init_vit_weights")

    called = []
    monkeypatch.setattr(
        timm_compat._vit,
        "init_weights_vit_jax",
        lambda m, n, head_bias=0.0: called.append(("jax", n, head_bias)),
    )
    monkeypatch.setattr(
        timm_compat._vit,
        "init_weights_vit_timm",
        lambda m, n: called.append(("timm", n)),
    )

    timm_compat._init_vit_weights(object(), "head", head_bias=-6.9, jax_impl=True)
    timm_compat._init_vit_weights(object(), "blocks.0", jax_impl=False)

    assert called == [("jax", "head", -6.9), ("timm", "blocks.0")]


def test_build_model_with_cfg_renames_default_cfg(monkeypatch):
    """0.4.12 passed default_cfg=; 1.x calls the same thing pretrained_cfg=."""
    if not timm_compat._NEEDED:
        pytest.skip("timm 0.4.12 accepts default_cfg directly")

    seen = {}
    monkeypatch.setattr(
        timm_compat._builder,
        "build_model_with_cfg",
        lambda cls, variant, pretrained, **kw: seen.update(kw) or "model",
    )

    out = timm_compat.build_model_with_cfg(
        object, "swin_base", False, default_cfg={"num_classes": 1000}, img_size=384
    )

    assert out == "model"
    assert seen["pretrained_cfg"] == {"num_classes": 1000}
    assert "default_cfg" not in seen
    assert seen["img_size"] == 384
