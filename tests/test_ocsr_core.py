"""Hermetic tests for the OCSR core helpers.

No network and no model loads: everything here runs from RDKit alone. Mirrors
``tests/test_docking_tools.py``, which is likewise hermetic by default.

Several of these pin behaviour that is silent when broken. Those are marked in the test
docstring with the consequence, because a reviewer tidying the code later needs to know
what the assertion is protecting.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from chemgraph.tools import ocsr_core as core


# ---------------------------------------------------------------------------
# Image loading and validation
# ---------------------------------------------------------------------------


@pytest.fixture
def png(tmp_path):
    """A real 1-molecule PNG, rendered with RDKit so no fixture file is needed."""
    from rdkit import Chem
    from rdkit.Chem import Draw

    path = tmp_path / "mol.png"
    Draw.MolToImage(Chem.MolFromSmiles("CCO"), size=(120, 120)).save(path)
    return path


def test_load_image_bytes_round_trip(png):
    import base64

    data, mime = core.load_image_bytes(str(png))
    assert mime == "image/png"
    assert data.startswith(b"\x89PNG")
    b64, mime64 = core.load_image_b64(str(png))
    assert (mime64, base64.b64decode(b64)) == (mime, data)
    with pytest.raises(FileNotFoundError):
        core.load_image_bytes(str(png) + ".missing")


def test_extract_image_path_only_picks_a_real_image(tmp_path, png):
    """Pulling a path out of free text must verify it, not trust the extension.

    ``png`` is a tmp_path fixture, so on Windows it carries a drive letter. A pattern
    that cannot match ``C:`` returns the remainder joined against the current drive,
    which points at a different filesystem location.
    """
    decoy = tmp_path / "notes.png"
    decoy.write_text("not an image")
    assert core.extract_image_path(f"read {png} please") == str(png)
    assert core.extract_image_path(f"read {decoy} please") is None
    assert core.extract_image_path("no image here") is None


def test_mime_is_sniffed_not_taken_from_the_extension(tmp_path):
    """A text file renamed .png must be refused.

    Silent when broken: the bytes would be base64-ed to a remote endpoint.
    """
    fake = tmp_path / "notes.png"
    fake.write_text("this is not an image")
    with pytest.raises(ValueError, match="not a recognized image"):
        core.load_image_bytes(str(fake))


def test_oversized_file_is_refused_before_being_read(tmp_path):
    big = tmp_path / "big.png"
    big.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 5000)
    with pytest.raises(ValueError, match="over the"):
        core.load_image_bytes(str(big), max_bytes=1000)


def test_directories_are_refused(tmp_path):
    """A path that is not a regular file must be refused before it is opened."""
    with pytest.raises(ValueError, match="not a regular file"):
        core.load_image_bytes(str(tmp_path))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo is POSIX-only")
def test_fifos_are_refused(tmp_path):
    """FIFOs and device files must not be opened.

    Silent when broken: reading a FIFO blocks forever with no caller-side timeout,
    and /dev/zero grows the buffer without bound.
    """
    fifo = tmp_path / "pipe.png"
    os.mkfifo(fifo)
    assert stat.S_ISFIFO(os.lstat(fifo).st_mode)
    with pytest.raises(ValueError, match="not a regular file"):
        core.load_image_bytes(str(fifo))


# ---------------------------------------------------------------------------
# SMILES handling
# ---------------------------------------------------------------------------


def test_canonicalize_collapses_kekule_and_aromatic():
    """DECIMER emits Kekule, the other specialists emit aromatic.

    Silent when broken: vote() would group the same molecule as two answers, and
    unanimity would drop from 289/422 benchmark items to 12/422. The confidence
    feature stops working while still returning plausible numbers.
    """
    assert core.canonicalize("C1=CC=CC=C1") == core.canonicalize("c1ccccc1")
    assert core.canonicalize("C(=O)(C(Br)(F)F)O") == core.canonicalize("O=C(O)C(F)(F)Br")
    for junk in ["not a molecule", "", None]:
        assert core.canonicalize(junk) is None


def test_validate_smiles_core_on_a_good_molecule():
    r = core.validate_smiles_core("CCO")
    assert r["valid"] is True
    assert r["formula"] == "C2H6O"
    assert r["n_atoms"] == 9  # implicit hydrogens included
    assert r["n_heavy_atoms"] == 3
    assert r["n_fragments"] == 1
    assert r["errors"] == []


def test_validate_smiles_core_surfaces_the_rdkit_message():
    """An invalid SMILES must carry RDKit's own complaint, not a generic failure.

    That message is the whole value of the tool here: a model can act on
    "Can't kekulize mol" but not on "invalid".
    """
    r = core.validate_smiles_core("c1ccccc1(C)(C)")
    assert r["valid"] is False
    assert any("kekulize" in e.lower() for e in r["errors"])


def test_multi_fragment_is_flagged_but_still_valid():
    """Two molecules in one image parse cleanly and would corrupt the downstream job.

    Silent when broken: RDKit embeds disconnected fragments overlapping (measured
    0.20 A closest approach, 0.00 A between centroids for CCO.CCN), so a geometry
    optimization runs on an interpenetrating pair and reports success.
    """
    r = core.validate_smiles_core("CCO.CCN")
    assert r["valid"] is True
    assert r["n_fragments"] == 2
    assert any("disconnected" in e for e in r["errors"])

    salt = core.validate_smiles_core("[Na+].[Cl-]")
    assert salt["n_fragments"] == 2


def test_validate_smiles_core_never_raises():
    for bad in ["", None, "?????", ">>", 42]:
        r = core.validate_smiles_core(bad)  # type: ignore[arg-type]
        assert r["valid"] is False
        assert r["errors"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CCO", "CCO"),
        ("```\nCCO\n```", "CCO"),
        ("```smiles\nc1ccccc1\n```", "c1ccccc1"),
        ("SMILES: CC(=O)O", "CC(=O)O"),
        ("The SMILES string is CC(=O)O", "CC(=O)O"),
        ("N#Cc1no[nH]c1=O", "N#Cc1no[nH]c1=O"),
    ],
)
def test_extract_smiles_finds_the_answer(raw, expected):
    assert core.extract_smiles(raw) == expected


@pytest.mark.parametrize(
    "refusal",
    [
        "I cannot process images.",
        "I'm unable to view images.",
        "Sorry, I can't see the image.",
        "As a text-based model, I cannot analyze pictures.",
        "No image was provided.",
    ],
)
def test_extract_smiles_rejects_refusals(refusal):
    """A refusal must not become a prediction.

    Silent when broken: RDKit parses "I cannot process images." as iodine, because
    I is an element and the parser stops at the space. The tool would report a
    confident one-atom molecule for an image the model never saw.
    """
    assert core.extract_smiles(refusal) is None


# ---------------------------------------------------------------------------
# Consensus
# ---------------------------------------------------------------------------


def _results(*pairs):
    return [{"model": m, "smiles": s, "ok": True} for m, s in pairs]


# ---------------------------------------------------------------------------
# Calibration and confidence
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Single-model priors
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The result contract
# ---------------------------------------------------------------------------


def test_build_result_has_every_contract_key():
    """Every documented key is present, so an agent never tests for a missing one."""
    r = core.build_result()
    assert set(r) == {
        "ok", "smiles", "valid", "formula", "n_fragments",
        "model_used", "cold_start", "latency_s", "error", "warning",
    }


# ---------------------------------------------------------------------------
# Extending the registry without touching Python
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutate, why", [
    (lambda r: r["specialists"]["decimer"].pop("latency_s"), "no latency_s"),
    (lambda r: r["specialists"]["decimer"].pop("import_name"), "no import_name"),
    (lambda r: r["specialists"]["decimer"].update({"import_name": 7}),
     "import_name is a number"),
    (lambda r: r["specialists"]["decimer"].update({"accuracy": "high"}),
     "accuracy is a string"),
    (lambda r: r["specialists"]["decimer"].update({"install": []}),
     "install is a list"),
    (lambda r: r.update({"defaults": []}), "defaults is a list"),
    (lambda r: r.update({"specialists": {}}), "no specialists at all"),
])
def test_a_registry_that_would_crash_at_import_is_rejected(tmp_path, mutate, why):
    """The registry is read during module import, so a bad field breaks `import`.

    Each of these used to surface as a KeyError or TypeError from inside a dict
    comprehension in ocsr_models, taking down `import chemgraph.tools.ocsr_tools` with
    a traceback that named neither the registry file nor the field at fault.
    """
    from importlib import resources

    from chemgraph.tools.ocsr_models import _validate_registry

    packaged = resources.files("chemgraph.tools").joinpath("ocsr_registry.json")
    registry = json.loads(packaged.read_text())
    mutate(registry)
    with pytest.raises(ValueError, match="unusable"):
        _validate_registry(registry, "test")


def test_canonicalize_is_stereo_blind_by_default():
    """Most reference labels carry no stereochemistry, so scoring with it marks a
    model wrong for correctly reading a wedge bond, and splits one molecule into two
    answers in vote()."""
    chiral, flat = "C[C@H](N)C(=O)O", "CC(N)C(=O)O"
    assert core.canonicalize(chiral) == core.canonicalize(flat)
    assert core.canonicalize(chiral, stereo=True) != core.canonicalize(flat)


@pytest.mark.parametrize("raw, want", [
    # Markdown emphasis is the dangerous wrapper: "*" is RDKit's wildcard atom, so
    # "**CCO**" parses as a valid five-atom molecule and was returned as a prediction.
    ("**CCO**", "CCO"),
    ("The SMILES is **CC(=O)Oc1ccccc1C(=O)O**", "CC(=O)Oc1ccccc1C(=O)O"),
    ("*CCO*", "CCO"),
    ("The answer is `CCO`.", "CCO"),
    # Parentheses and brackets only cost a match, since "(CCO)" fails to parse.
    ("(CCO)", "CCO"),
    ("This is aspirin (CC(=O)Oc1ccccc1C(=O)O).", "CC(=O)Oc1ccccc1C(=O)O"),
    ("[CCO]", "CCO"),
    # A JSON reply is the most common structured form, and the "elements" list
    # beside it is full of strings that parse: "Cl" is a molecule.
    ('{"elements": ["C", "H", "Cl"], "smiles": "CCCl"}', "CCCl"),
    ('```json\n{\n  "elements": ["C", "O"],\n  "smiles": "CCO"\n}\n```', "CCO"),
    # Brackets and parentheses are SMILES syntax and must survive untouched.
    ("[NH4+]", "[NH4+]"),
    ("[Na+].[Cl-]", "[Na+].[Cl-]"),
    ("C(=O)O", "C(=O)O"),
    ("[C@H](N)C", "[C@H](N)C"),
    ("*CCO", "*CCO"),
    ("**[Na+].[Cl-]**", "[Na+].[Cl-]"),
])
def test_wrappers_are_peeled_without_damaging_real_smiles(raw, want):
    """A wrapped SMILES must come back unwrapped, and a real one untouched.

    Silent when broken: emphasis markers are wildcard atoms, so the extracted string
    stays a valid molecule with extra atoms in it, and every downstream check passes
    while the answer is wrong.
    """
    assert core.extract_smiles(raw) == want


def test_a_marked_up_refusal_is_not_a_molecule():
    """A refusal wrapped in emphasis must not become a wildcard-atom molecule.

    Measured on benchmark replies: an auth-error page yielded the SMILES "**", two
    wildcard atoms. The LLM backend screens those by keyword before extraction, so
    this is the guard for every other caller and for wording the screen misses.
    """
    denied = (
        "⚠️ **IMPORTANT AUTHENTICATION NOTICE FROM ARGO** ⚠️\n\n"
        "\U0001f6ab **ACCESS DENIED** \U0001f6ab\n\nThe username is not authorized."
    )
    assert core.extract_smiles(denied) is None


def test_a_new_specialist_needs_no_source_change(tmp_path, monkeypatch):
    """The promise the registry file exists to keep.

    Adding a model must be a data edit: one entry in ocsr_registry.json plus a loader
    in ocsr_backends. If this test needs a change elsewhere to pass again, something
    has grown a second hardcoded list of models and the promise is broken.
    """
    import importlib
    import json as _json
    from importlib import resources

    packaged = resources.files("chemgraph.tools").joinpath("ocsr_registry.json")
    reg = _json.loads(packaged.read_text())
    reg["specialists"]["newmodel"] = {
        "import_name": "newmodel_pkg",
        "accuracy": 0.5,
        "latency_s": 1.5,
        "note": "A hypothetical fifth specialist.",
        "install": {"weights": "~/w/new.pth", "weights_gb": 0.5},
    }
    custom = tmp_path / "registry.json"
    custom.write_text(_json.dumps(reg))
    monkeypatch.setenv("CHEMGRAPH_OCSR_REGISTRY", str(custom))

    from chemgraph.tools import ocsr_backends, ocsr_models, ocsr_tools

    mods = (ocsr_models, ocsr_backends, ocsr_tools)
    for mod in mods:
        importlib.reload(mod)
    try:
        assert "newmodel" in ocsr_models.SPECIALIST_MODELS
        assert "newmodel" in ocsr_models.MODEL_CHOICES
        assert "newmodel" in ocsr_models.describe_models()
        # It reaches dispatch instead of being rejected as an unknown model.
        r = ocsr_tools.image_to_smiles_core("/nonexistent.png", model="newmodel")
        assert "unknown model" not in (r["error"] or "")
        # And its checkpoint path is resolved from the entry, not from a table here.
        assert ocsr_backends.checkpoint_path("newmodel").endswith("w/new.pth")
    finally:
        monkeypatch.delenv("CHEMGRAPH_OCSR_REGISTRY")
        for mod in mods:
            importlib.reload(mod)


def test_decimer_needs_no_checkpoint_path():
    """It fetches and caches its own weights, so the tool must not demand a path."""
    from chemgraph.tools import ocsr_backends

    assert ocsr_backends.checkpoint_path("decimer") is None
    for name in ("molnextr", "molscribe", "ocsrglyph"):
        assert ocsr_backends.checkpoint_path(name), f"{name} needs a checkpoint path"


def test_a_missing_checkpoint_is_reported_before_the_model_loads(monkeypatch):
    """The message has to name the file, or the user cannot tell what to fetch."""
    from chemgraph.tools import ocsr_backends

    monkeypatch.setattr(ocsr_backends, "checkpoint_path",
                        lambda name: "/nonexistent/weights.pth")
    monkeypatch.setattr(
        ocsr_backends, "_LOADERS",
        {"molnextr": lambda w: pytest.fail("must not load without a checkpoint")})

    model, cold, error = ocsr_backends._get_model("molnextr")

    assert model is None and cold
    assert "/nonexistent/weights.pth" in error


def test_weights_dir_env_relocates_every_checkpoint(monkeypatch, tmp_path):
    """On a cluster the weights sit on shared scratch, not under $HOME.

    Without this, moving them means copying the whole registry file just to edit
    three paths.
    """
    from chemgraph.tools import ocsr_backends

    monkeypatch.setenv("CHEMGRAPH_OCSR_WEIGHTS_DIR", str(tmp_path))

    for name in ("molnextr", "molscribe", "ocsrglyph"):
        path = ocsr_backends.checkpoint_path(name)
        assert path.startswith(str(tmp_path)), f"{name} ignored the override"
        assert name in path, "each model keeps its own subdirectory"

    # DECIMER caches its own weights, so the override has nothing to relocate.
    assert ocsr_backends.checkpoint_path("decimer") is None
