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
    data, mime = core.load_image_bytes(str(png))
    assert mime == "image/png"
    assert data.startswith(b"\x89PNG")
    with pytest.raises(FileNotFoundError):
        core.load_image_bytes(str(png) + ".missing")


def test_a_bare_name_resolves_against_the_log_dir(tmp_path, monkeypatch, png):
    """Matches every other file-reading tool: a sibling wrote it there by bare name."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "written.png").write_bytes(png.read_bytes())
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(log_dir))
    monkeypatch.chdir(tmp_path)

    data, mime = core.load_image_bytes("written.png")

    assert mime == "image/png" and data.startswith(b"\x89PNG")


def test_a_cwd_relative_path_still_wins_over_the_log_dir(tmp_path, monkeypatch, png):
    """The log dir is a fallback; a file that exists in cwd is the one meant."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "here.png").write_text("not an image")
    (tmp_path / "here.png").write_bytes(png.read_bytes())
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(log_dir))
    monkeypatch.chdir(tmp_path)

    assert core.load_image_bytes("here.png")[1] == "image/png"


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


# ---------------------------------------------------------------------------
# Committee voting
# ---------------------------------------------------------------------------


def _results(*pairs):
    return [{"model": m, "smiles": s, "ok": True} for m, s in pairs]


def test_the_local_prefix_names_the_same_model():
    """Silent when broken: a "local:" name never matches the table's committee, so
    check_committee nulls the confidence on every ensemble call."""
    v = core.vote(_results(("local:decimer", "CCO"), ("molnextr", "CCO")))
    assert v["committee"] == ["decimer", "molnextr"]


def test_vote_groups_by_canonical_form_not_string():
    """Same molecule written two ways is one vote."""
    v = core.vote(_results(("decimer", "C(=O)(C(Br)(F)F)O"),
                           ("molnextr", "O=C(O)C(F)(F)Br"),
                           ("molscribe", "O=C(O)C(F)(F)Br"),
                           ("ocsrglyph", "CC(C)(Br)C(=O)O")))
    assert v["pattern"] == "3/1"
    assert v["winner"] == core.canonicalize("O=C(O)C(F)(F)Br")


def test_an_abstainer_stays_in_the_committee_but_out_of_the_voters():
    """A model that ran and produced junk is counted, and named as abstaining.

    Silent when broken: it vanishes from the committee, which makes a partial
    install look like a full one to check_committee.
    """
    v = core.vote(_results(("decimer", "CCO"), ("molnextr", "CCO"),
                           ("molscribe", "CCO"), ("ocsrglyph", "@@@junk")))
    assert v["committee"] == ["decimer", "molnextr", "molscribe", "ocsrglyph"]
    assert v["voters"] == ["decimer", "molnextr", "molscribe"]
    assert "ocsrglyph" in v["abstained"]


@pytest.mark.parametrize("smiles, expected", [
    (("CCO", "CCO", "CCO", "CCO"), "4"),
    (("CCO", "CCO", "CCO", "@@@"), "3/1"),
    (("CCO", "CCO", "@@@", "???"), "2/1/1"),
    (("CCO", "CCO", "CCC", "CCC"), "2/2"),
    (("CCO", "CCC", "@@@", "???"), "1/1/1/1"),
    (("@@@", "???", "!!!", "###"), "1/1/1/1"),
])
def test_pattern_always_sums_to_the_committee_size(smiles, expected):
    """Every four-model outcome lands in one of five buckets.

    Silent when broken: a shorter pattern reads as a smaller committee and looks up
    the wrong row.
    """
    models = ["decimer", "molnextr", "molscribe", "ocsrglyph"]
    v = core.vote(_results(*zip(models, smiles)))
    assert v["pattern"] == expected
    assert sum(int(x) for x in v["pattern"].split("/")) == 4


def test_vote_breaks_ties_by_model_priority():
    v = core.vote(
        _results(("decimer", "CCO"), ("molnextr", "CCN"),
                 ("molscribe", "CCC"), ("ocsrglyph", "CCF")),
        priority=["decimer", "molnextr", "molscribe", "ocsrglyph"],
    )
    assert v["pattern"] == "1/1/1/1"
    assert v["winner"] == "CCO"


def test_an_abstaining_model_cannot_win_the_tie_break():
    """Only a model that actually voted may decide the answer.

    Silent when broken: the winner comes from outside `voters`, and the all-different
    bucket stops carrying the strongest model's solo accuracy it is assumed to.
    """
    v = core.vote(
        [{"model": "decimer", "ok": False, "smiles": "CCN"},
         {"model": "molnextr", "ok": True, "smiles": "CCO"},
         {"model": "molscribe", "ok": True, "smiles": "CCN"},
         {"model": "ocsrglyph", "ok": True, "smiles": "CCF"}],
        ["decimer", "molnextr", "molscribe", "ocsrglyph"],
    )
    assert v["winner"] == "CCO"
    assert "decimer" in v["abstained"]
    assert set(v["votes"][v["winner"]]) <= set(v["voters"])


def test_vote_with_nobody_voting():
    """Total failure is the all-singletons bucket, not a missing row.

    Dropping the item would remove it from the calibration table and flatter every
    other bucket.
    """
    v = core.vote(_results(("decimer", "@@@"), ("molnextr", "???")))
    assert v["pattern"] == "1/1"
    assert v["winner"] is None
    assert v["voters"] == []
    assert len(v["abstained"]) == 2


def test_a_repeated_model_name_still_contributes_its_own_singleton():
    """Two results for one model must not collapse into one dict key.

    Silent when broken: the pattern loses a part and stops summing to the committee
    size, which `--models decimer,decimer` and the "local:" alias both produce.
    """
    v = core.vote([{"model": "decimer", "ok": True, "smiles": "@@@"},
                   {"model": "decimer", "ok": True, "smiles": "???"},
                   {"model": "molnextr", "ok": True, "smiles": "CCO"}])
    assert v["pattern"] == "1/1/1"


# ---------------------------------------------------------------------------
# Calibration tables
# ---------------------------------------------------------------------------


def test_packaged_table_is_internally_consistent():
    """The shipped table must satisfy the validator it is loaded through."""
    t = core.load_calibration()
    assert t["committee"] == ["decimer", "molnextr", "molscribe", "ocsrglyph"]
    assert t["n_items"] == 722
    for name, cell in t["patterns"].items():
        assert sum(int(x) for x in name.split("/")) == len(t["committee"])
        assert 0 <= cell["k"] <= cell["n"]


def test_the_shipped_tie_break_matches_its_own_measured_accuracies():
    """The recorded priority must be strongest model first.

    Silent when broken: the all-different bucket's accuracy was measured under this
    order, so a different order quotes a number measured for another model's answer.
    """
    t = core.load_calibration()
    order = core.tie_break_order(t)
    accuracies = [t["model_performance"][m]["accuracy"] for m in order]
    assert accuracies == sorted(accuracies, reverse=True), order


def test_a_table_with_no_tie_break_falls_back_to_committee_order(tmp_path):
    custom = tmp_path / "cal.json"
    custom.write_text(json.dumps({"committee": ["a", "b"],
                                  "patterns": {"2": {"k": 1, "n": 1}}}))
    assert core.tie_break_order(core.load_calibration(str(custom))) == ["a", "b"]


@pytest.mark.parametrize("tie_break, why", [
    ("decimer,molnextr,molscribe,ocsrglyph", "no 'model-priority:' prefix to parse"),
    ("model-priority: decimer,molnextr", "names fewer models than the committee"),
    ("model-priority: a,b,c,d", "names models the committee does not contain"),
])
def test_an_unusable_tie_break_is_rejected_at_load(tmp_path, tie_break, why):
    """A tie_break that cannot be trusted must fail loudly.

    Silent when broken: the fallback is the committee's arbitrary JSON order, so the
    tool votes one way and quotes a number measured the other way.
    """
    custom = tmp_path / "cal.json"
    custom.write_text(json.dumps({
        "committee": ["decimer", "molnextr", "molscribe", "ocsrglyph"],
        "tie_break": tie_break,
        "patterns": {"4": {"k": 1, "n": 1}},
    }))
    with pytest.raises(ValueError, match="unusable"):
        core.load_calibration(str(custom))


@pytest.mark.parametrize("table, why", [
    ([], "top level is not an object"),
    ({"patterns": {"1": {"k": 1, "n": 2}}}, "no committee at all"),
    ({"committee": "a,b", "patterns": {"2": {"k": 1, "n": 2}}},
     "committee is a string"),
    ({"committee": [], "patterns": {"1": {"k": 1, "n": 1}}},
     "an empty committee would disable the mismatch guard entirely"),
    ({"committee": ["a", "a"], "patterns": {"1/1": {"k": 1, "n": 1}}},
     "a repeated model mismatches forever"),
    ({"committee": ["a"], "patterns": {}}, "no patterns"),
    ({"committee": ["a", "b"], "patterns": {"3": {"k": 1, "n": 2}}},
     "pattern sums past the committee, so it was fit under another abstention rule"),
    ({"committee": ["a"], "patterns": {"2/-1": {"k": 1, "n": 1}}},
     "int() accepts a minus sign, so this sums to 1 and passes the sum check"),
    ({"committee": ["a"], "patterns": {"1": {"k": 5, "n": 2}}}, "k exceeds n"),
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1, "p": "high"}}},
     "p is a string, which used to raise TypeError inside a lookup"),
    ({"committee": ["a"], "patterns": {"1": {"k": 0, "n": 100, "p": 0.99}}},
     "p contradicts its own k and n"),
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1,
                                             "ci": [float("nan"), 1.0]}}},
     "json accepts NaN, which compares false against everything"),
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1, "ci": [0.99, 0.1]}}},
     "a reversed interval"),
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1}},
      "model_performance": {"decimer": {"accuracy": 0.9, "n": 0}}},
     "an accuracy backed by zero observations reads as a real measurement"),
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1}},
      "model_performance": {"decimer": {"accuracy": "90%"}}},
     "accuracy is a string"),
])
def test_a_table_that_cannot_mean_what_it_says_is_rejected_at_load(
        tmp_path, monkeypatch, table, why):
    """A table decides which answers get a confidence, so it fails at load.

    Every case here once loaded cleanly and then either raised a TypeError from deep
    inside a lookup, naming neither the table nor the field, or returned a confident
    number that contradicted the table's own k and n.
    """
    custom = tmp_path / "cal.json"
    custom.write_text(json.dumps(table))
    with pytest.raises(ValueError, match="unusable"):
        core.load_calibration(str(custom))
    # Same rejection whichever route the table arrives by.
    monkeypatch.setenv("CHEMGRAPH_OCSR_CALIBRATION", str(custom))
    with pytest.raises(ValueError, match="unusable"):
        core.load_calibration()


def test_a_calibration_path_that_is_not_a_regular_file_is_refused(tmp_path):
    """A FIFO blocks open() forever, and the caller has already paid for inference."""
    fifo = tmp_path / "cal.json"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="not a regular file"):
        core.load_calibration(str(fifo))


def test_unparseable_json_is_a_valueerror_not_a_recursionerror(tmp_path):
    """Callers guard on ValueError, so a bad table costs the confidence, not the run.

    Silent when broken: json.load raises RecursionError on a deeply nested file, which
    is neither ValueError nor TypeError and discarded a completed ensemble run.
    """
    custom = tmp_path / "cal.json"
    custom.write_text("[" * 400)
    with pytest.raises(ValueError):
        core.load_calibration(str(custom))


# ---------------------------------------------------------------------------
# Confidence lookup
# ---------------------------------------------------------------------------


def test_agreement_is_worth_more_than_any_single_model():
    """The measurement the whole committee exists for.

    Four agreeing models beat the strongest model alone, and four disagreeing ones
    are worse than a coin flip. An agent that cannot see this difference has no
    reason to pay for four inferences.
    """
    t = core.load_calibration()
    assert core.confidence("4", t)["p"] > t["model_performance"]["decimer"]["accuracy"]
    assert core.confidence("1/1/1/1", t)["p"] < 0.5


@pytest.mark.parametrize("pattern, label", [
    ("4", "unanimous"),
    ("3/1", "strong"),
    ("2/1/1", "weak"),
    ("1/1/1/1", "conflicting"),
])
def test_each_shipped_pattern_gets_its_band(pattern, label):
    assert core.confidence(pattern, core.load_calibration())["label"] == label


def test_a_thin_bucket_reports_a_label_but_no_number():
    """Below the sample floor the interval spans tens of points.

    Silent when broken: quoting a decimal from twelve items reads as a measurement.
    """
    got = core.confidence("2/2", core.load_calibration())
    assert got["p"] is None
    assert got["reason"] == "below_n_floor"
    assert got["label"].startswith("low_n_")
    assert got["n"] == 12


def test_a_thin_bucket_label_comes_from_the_jeffreys_estimate(tmp_path):
    """7/7 is low_n_weak: thin evidence pointing one way.

    The raw estimate would call it unanimous, claiming certainty from seven items;
    the interval's lower bound would call it conflicting, overstating the doubt.
    """
    custom = tmp_path / "cal.json"
    custom.write_text(json.dumps({"committee": ["a", "b"],
                                  "patterns": {"2": {"k": 7, "n": 7}}}))
    got = core.confidence("2", core.load_calibration(str(custom)))
    assert got["label"] == "low_n_weak"


def test_an_unknown_pattern_gets_no_number_at_all():
    """A pattern the table never measured must not borrow another bucket's number."""
    got = core.confidence("5/5", core.load_calibration())
    assert (got["p"], got["reason"]) == (None, "unknown_pattern")


def test_no_prediction_is_distinct_from_an_unknown_pattern():
    got = core.confidence(None, core.load_calibration())
    assert (got["p"], got["reason"]) == (None, "no_prediction")


@pytest.mark.parametrize("floor, why", [
    ("20", "a string floor silently disables the check"),
    (True, "bool is an int in Python, so isinstance alone lets it through"),
    (-5, "a negative floor can never hold"),
])
def test_an_unusable_sample_floor_is_refused_at_load(tmp_path, floor, why):
    """The floor decides whether a number is quotable, so it fails at load.

    Silent when broken: the table records a floor, confidence() cannot read it, and
    a point estimate over four images is quoted as measured.
    """
    custom = tmp_path / "cal.json"
    custom.write_text(json.dumps({"committee": ["a"],
                                  "min_n_for_point_estimate": floor,
                                  "patterns": {"1": {"k": 4, "n": 4}}}))
    with pytest.raises(ValueError, match="min_n_for_point_estimate"):
        core.load_calibration(str(custom))


@pytest.mark.parametrize("literal, why", [
    ("NaN", "NaN < anything is False, so the floor would never fire"),
    ("Infinity", "inf fires on every bucket, however large"),
])
def test_a_non_finite_sample_floor_is_refused(tmp_path, literal, why):
    """json accepts both, and each disables the floor in a different direction.

    Silent when broken: NaN quotes a number from four images, inf reports a bucket
    of a thousand as too thin to quote.
    """
    custom = tmp_path / "cal.json"
    custom.write_text('{"committee": ["a"], "min_n_for_point_estimate": '
                      + literal + ', "patterns": {"1": {"k": 990, "n": 1000}}}')
    with pytest.raises(ValueError, match="min_n_for_point_estimate"):
        core.load_calibration(str(custom))





def test_a_table_declaring_a_floor_is_held_to_it(tmp_path):
    """A point estimate over fewer images than the table's own floor is withheld.

    Silent when broken: a hand-written table, or one fitted with a lower --min-n
    than it records, quotes a hard number from four images with a 45 pp interval.
    """
    custom = tmp_path / "cal.json"
    custom.write_text(json.dumps({
        "committee": ["a", "b"], "min_n_for_point_estimate": 20,
        "patterns": {"2": {"k": 4, "n": 4, "p": 0.9}},
    }))

    got = core.confidence("2", core.load_calibration(str(custom)))

    assert got["p"] is None
    assert got["reason"] == "below_n_floor"
    assert got["label"].startswith("low_n_")


def test_a_floor_written_as_a_json_decimal_still_holds(tmp_path):
    """JSON 20.0 deserializes to float, which an int-only check would skip."""
    custom = tmp_path / "cal.json"
    custom.write_text(json.dumps({"committee": ["a"],
                                  "min_n_for_point_estimate": 20.0,
                                  "patterns": {"1": {"k": 4, "n": 4, "p": 0.9}}}))

    assert core.confidence("1", core.load_calibration(str(custom)))["p"] is None


# ---------------------------------------------------------------------------
# Single-model priors
# ---------------------------------------------------------------------------


def test_single_model_must_not_be_routed_through_vote():
    """The trap prior_confidence exists to avoid.

    Silent when broken: a one-model vote yields pattern "1", which a four-model
    table does not contain, so the strongest model reports unknown_pattern instead
    of its measured accuracy.
    """
    from chemgraph.tools.ocsr_models import DEFAULT_SPECIALIST

    t = core.load_calibration()
    assert core.vote(_results(("decimer", "CCO")))["pattern"] == "1"
    assert core.confidence("1", t)["reason"] == "unknown_pattern"
    assert core.prior_confidence(DEFAULT_SPECIALIST, t)["p"] is not None


def test_every_registered_specialist_has_a_prior():
    """A model the registry offers must have a measured accuracy to report."""
    from chemgraph.tools.ocsr_models import SPECIALIST_MODELS

    t = core.load_calibration()
    for name in SPECIALIST_MODELS:
        assert core.prior_confidence(name, t)["p"] is not None, name


def test_a_prior_reads_the_table_and_not_a_constant(tmp_path):
    """Refitting on other images must move the priors.

    Silent when broken: a figure compiled into the source outlives the table it was
    measured on and reports another dataset's accuracy.
    """
    custom = tmp_path / "cal.json"
    custom.write_text(json.dumps({
        "committee": ["decimer"], "patterns": {"1": {"k": 1, "n": 1}},
        "model_performance": {"decimer": {"accuracy": 0.5, "n": 10}},
    }))
    t = core.load_calibration(str(custom))
    assert core.prior_confidence("decimer", t)["p"] == 0.5
    assert core.prior_confidence("local:decimer", t)["p"] == 0.5


def test_an_unmeasured_model_reports_why_it_has_no_prior():
    got = core.prior_confidence("nosuchmodel", core.load_calibration())
    assert (got["p"], got["reason"]) == (None, "no_prior_for_model")


def test_an_unreadable_table_costs_the_prior_and_not_the_run(tmp_path, monkeypatch):
    """A broken table must not raise out of a call that already read the image."""
    bad = tmp_path / "cal.json"
    bad.write_text("{not json")
    monkeypatch.setenv("CHEMGRAPH_OCSR_CALIBRATION", str(bad))
    got = core.prior_confidence("decimer")
    assert got["p"] is None
    assert got["reason"].startswith("calibration_unreadable")
    assert core.model_performance("decimer") == {}


def test_model_performance_reports_the_counts_behind_an_accuracy():
    got = core.model_performance("decimer", core.load_calibration())
    assert got["k"] == 649 and got["n"] == 722
    assert got["ci"][0] < got["accuracy"] < got["ci"][1]


# ---------------------------------------------------------------------------
# Committee mismatch
# ---------------------------------------------------------------------------


def test_a_matching_committee_reports_no_problem():
    t = core.load_calibration()
    v = core.vote(_results(*[(m, "CCO") for m in t["committee"]]))
    assert core.check_committee(v, t) is None


def test_a_partial_install_is_told_how_to_complete_itself():
    """The common case: fewer models installed than the table was fit on.

    Silent when broken: every ensemble call reports no confidence, and the message
    that says so does not say what to do about it.
    """
    t = core.load_calibration()
    v = core.vote(_results(("decimer", "CCO"), ("molnextr", "CCO")))
    why = core.check_committee(v, t)
    assert "committee_mismatch" in why
    assert "pip install 'chemgraph[ocsr]'" in why
    assert "molscribe" in why and "ocsrglyph" in why
    # The command has to exist: an earlier version named a module this repo does
    # not ship, so following the instruction gave ModuleNotFoundError.
    assert "ocsr_setup" not in why and "ocsr_download" not in why


def test_running_more_models_than_the_table_describes_is_also_a_mismatch():
    """A larger set is told to subset rather than to install."""
    t = core.load_calibration()
    v = core.vote(_results(*[(m, "CCO") for m in t["committee"] + ["extra"]]))
    why = core.check_committee(v, t)
    assert "models_wanted" in why


def test_abstention_does_not_change_which_table_applies():
    """The check compares the models asked, not the ones that voted.

    Silent when broken: a model that failed to read one image would look like a
    partial install and null the confidence for that image alone.
    """
    t = core.load_calibration()
    v = core.vote([{"model": m, "smiles": "CCO", "ok": m != "ocsrglyph"}
                   for m in t["committee"]])
    assert "ocsrglyph" in v["abstained"]
    assert core.check_committee(v, t) is None


def test_an_integer_too_large_for_float_is_refused_at_load(tmp_path):
    """A 400-digit k passes every isinstance check and then breaks the arithmetic.

    Silent when broken: float() raises OverflowError, which is neither ValueError
    nor TypeError, so it escapes the guard at every call site and surfaces from
    inside a lookup instead of from the load.
    """
    huge = 10 ** 400
    custom = tmp_path / "cal.json"
    custom.write_text(json.dumps({"committee": ["a"],
                                  "patterns": {"1": {"k": huge, "n": huge}}}))
    with pytest.raises(ValueError, match="not a usable number"):
        core.load_calibration(str(custom))


def test_a_committee_differing_both_ways_is_told_both_remedies():
    """Neither installing nor subsetting alone fixes this, so the message says both.

    Silent when broken: the user follows one instruction and still gets no number.
    """
    t = {"committee": ["a", "b"], "patterns": {"2": {"k": 1, "n": 1}}}
    v = core.vote(_results(("b", "CCO"), ("c", "CCO")))

    why = core.check_committee(v, t)

    assert "install a" in why
    assert "ocsr_calibrate" in why

