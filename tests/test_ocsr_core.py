"""Hermetic tests for the OCSR core helpers.

No network, no conda envs, no model loads: everything here runs from RDKit and the
packaged calibration table. Mirrors ``tests/test_docking_tools.py``, which is likewise
hermetic by default.

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


def test_vote_unanimous_and_the_local_prefix():
    """Four agreeing models are pattern "4", and "local:x" is the same model as "x"."""
    v = core.vote(_results(("decimer", "CCO"), ("molnextr", "CCO"),
                           ("molscribe", "CCO"), ("ocsrglyph", "CCO")))
    assert (v["pattern"], v["winner"]) == ("4", "CCO")
    # Silent when broken: a "local:" name would not match the table's committee, so
    # check_committee would null the confidence on every ensemble call.
    v = core.vote(_results(("local:decimer", "CCO"), ("molnextr", "CCO")))
    assert v["committee"] == ["decimer", "molnextr"]


def test_vote_groups_by_canonical_form_not_string():
    """Same molecule written two ways is one vote, not two."""
    v = core.vote(_results(("decimer", "C(=O)(C(Br)(F)F)O"),
                           ("molnextr", "O=C(O)C(F)(F)Br"),
                           ("molscribe", "O=C(O)C(F)(F)Br"),
                           ("ocsrglyph", "CC(C)(Br)C(=O)O")))
    assert v["pattern"] == "3/1"
    assert v["winner"] == core.canonicalize("O=C(O)C(F)(F)Br")


def test_vote_treats_unparseable_output_as_a_dissenting_vote():
    """A model that ran but produced junk contributes a singleton, not an absence.

    The pattern stays "3/1" rather than collapsing to "3", because a model unable to
    read the image is evidence about the image, not a smaller committee.
    """
    v = core.vote(_results(("decimer", "CCO"), ("molnextr", "CCO"),
                           ("molscribe", "CCO"), ("ocsrglyph", "@@@junk")))
    assert v["pattern"] == "3/1"
    assert v["committee"] == ["decimer", "molnextr", "molscribe", "ocsrglyph"]
    assert v["voters"] == ["decimer", "molnextr", "molscribe"]
    assert "ocsrglyph" in v["abstained"]


def test_pattern_always_sums_to_the_committee_size():
    """The invariant that makes a table's buckets mean one thing.

    Every combination of agreement and abstention over four models must land in one
    of five buckets, never in a shorter pattern that a smaller committee would give.
    """
    cases = [
        (("CCO", "CCO", "CCO", "CCO"), "4"),
        (("CCO", "CCO", "CCO", "@@@"), "3/1"),
        (("CCO", "CCO", "@@@", "???"), "2/1/1"),
        (("CCO", "CCO", "CCC", "CCC"), "2/2"),
        (("CCO", "CCC", "@@@", "???"), "1/1/1/1"),
        (("@@@", "???", "!!!", "###"), "1/1/1/1"),
    ]
    for smiles, expected in cases:
        v = core.vote(_results(*zip(["decimer", "molnextr", "molscribe", "ocsrglyph"],
                                    smiles)))
        assert v["pattern"] == expected, smiles
        assert sum(int(x) for x in v["pattern"].split("/")) == 4


def test_vote_breaks_ties_by_model_priority():
    v = core.vote(
        _results(("decimer", "CCO"), ("molnextr", "CCN"),
                 ("molscribe", "CCC"), ("ocsrglyph", "CCF")),
        priority=["decimer", "molnextr", "molscribe", "ocsrglyph"],
    )
    assert v["pattern"] == "1/1/1/1"
    assert v["winner"] == "CCO"  # decimer wins the four-way tie


def test_vote_with_nobody_voting():
    """Total failure is the all-singletons bucket, not a missing row.

    Dropping the item would remove it from the calibration table and flatter every
    other bucket. It happened once in 722 benchmark items: a fused polycyclic where
    all four models emitted unparseable strings, two of them inventing metal atoms.
    """
    v = core.vote(_results(("decimer", "@@@"), ("molnextr", "???")))
    assert v["pattern"] == "1/1"
    assert v["winner"] is None
    assert v["voters"] == []
    assert len(v["abstained"]) == 2


# ---------------------------------------------------------------------------
# Calibration and confidence
# ---------------------------------------------------------------------------


def test_packaged_table_is_internally_consistent():
    """Guards against a hand-edited table: the arithmetic must still close."""
    t = core.load_calibration()
    assert sum(c["n"] for c in t["patterns"].values()) == t["n_items"]
    for name, c in t["patterns"].items():
        assert sum(int(x) for x in name.split("/")) == len(t["committee"])
        assert 0 <= c["k"] <= c["n"]
        assert (c["p"] is None) == (c["n"] < t["min_n_for_point_estimate"])
        if c["p"] is not None:
            assert c["p"] == round((c["k"] + 0.5) / (c["n"] + 1), 4)
        assert c["ci"][0] <= c["ci"][1]


def test_check_committee_catches_a_mismatch():
    t = core.load_calibration()
    three = core.vote(_results(("decimer", "CCO"), ("molnextr", "CCO"), ("molscribe", "CCO")))
    assert "committee_mismatch" in core.check_committee(three, t)


# ---------------------------------------------------------------------------
# Single-model priors
# ---------------------------------------------------------------------------


def test_every_registered_specialist_has_a_prior():
    """Registering a specialist must give it a confidence, not just a backend.

    The prior reads from SPECIALIST_MODELS, so there is no second table to forget.
    Values are the measured solo accuracies over the same 722 items.
    """
    from chemgraph.tools.ocsr_models import SPECIALIST_MODELS

    n_items = core.load_calibration()["n_items"]
    measured = core.load_calibration()["model_performance"]
    assert set(SPECIALIST_MODELS) <= set(measured), (
        "every registered specialist needs a model_performance entry in the "
        "calibration table, or backend=<name> reports no confidence"
    )
    for model in SPECIALIST_MODELS:
        c = core.prior_confidence(model)
        assert c["p"] == measured[model]["accuracy"]
        assert c["n"] == n_items  # from the table, not a constant in the source
        assert c["reason"] is None
        assert not c["label"].startswith("low_n_")


def test_single_model_must_not_be_routed_through_vote():
    """Documents the trap that prior_confidence exists to avoid.

    A one-model vote yields pattern "1", which in the four-model table is the
    all-different bucket: 8 items, 12.5% correct. Routing DECIMER's 89.9% through
    it would attach the table's least reliable label to the most accurate model.
    """
    t = core.load_calibration()
    v = core.vote(_results(("decimer", "CCO")))
    assert v["pattern"] == "1"
    assert core.confidence("1", t)["reason"] == "unknown_pattern"
    from chemgraph.tools.ocsr_models import DEFAULT_SPECIALIST

    assert core.prior_confidence(DEFAULT_SPECIALIST)["p"] is not None  # correct path


# ---------------------------------------------------------------------------
# The result contract
# ---------------------------------------------------------------------------


def test_build_result_has_every_contract_key():
    r = core.build_result()
    assert len(r) == 18
    for key in ["ok", "smiles", "valid", "formula", "n_fragments", "confidence",
                "confidence_label", "confidence_unavailable_reason", "agreement",
                "basis", "backend_used", "model_used", "cold_start", "latency_s",
                "error", "warning", "votes", "abstained"]:
        assert key in r


def test_mock_ocsr_satisfies_the_contract():
    """The hermetic stand-in must produce the shape a real single-model call does.

    Otherwise tests written against it pass while the real path is broken. It used to
    return a number with basis="prior", which is a shape no backend produces now that
    a single model reports no per-image confidence.
    """
    r = core.mock_ocsr("/nonexistent.png")
    assert set(r) == set(core.build_result())
    assert r["ok"] is True
    assert core.canonicalize(r["smiles"]) is not None
    assert r["confidence"] is None
    assert r["basis"] is None
    assert r["confidence_unavailable_reason"] == (
        "single_model_has_no_per_image_confidence")


# ---------------------------------------------------------------------------
# Extending the registry without touching Python
# ---------------------------------------------------------------------------


def test_an_abstaining_model_cannot_win_the_tie_break():
    """Only a model that actually voted may decide the answer.

    The tie-break used to scan every result without re-checking ok, so a model whose
    output failed to parse still got consulted, and won whenever its unusable string
    happened to canonicalize into a tied group. The returned SMILES then came from
    outside `voters`, and the all-different bucket stopped carrying the strongest
    model's solo accuracy that the calibration table assumes it does.
    """
    v = core.vote(
        [{"model": "decimer", "ok": False, "smiles": "CCN"},
         {"model": "molnextr", "ok": True, "smiles": "CCO"},
         {"model": "molscribe", "ok": True, "smiles": "CCN"},
         {"model": "ocsrglyph", "ok": True, "smiles": "CCF"}],
        ["decimer", "molnextr", "molscribe", "ocsrglyph"],
    )
    assert v["winner"] == "CCO"  # top-priority *voter*, not the abstainer
    assert "decimer" in v["abstained"]
    winning_models = v["votes"][v["winner"]]
    assert set(winning_models) <= set(v["voters"])


@pytest.mark.parametrize("table, why", [
    # A cell whose numbers contradict each other or cannot be read.
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1, "p": "high"}}},
     "p is a string"),
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1, "p": 2.5}}},
     "p outside [0, 1]"),
    ({"committee": ["a"], "patterns": {"1": {"k": 0, "n": 100, "p": 0.99}}},
     "p contradicts its own k and n"),
    ({"committee": ["a"], "patterns": {"1": {"k": 5, "n": 2}}}, "k exceeds n"),
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1, "ci": "nope"}}},
     "ci is a string"),
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1, "ci": [0.99, 0.1]}}},
     "ci is reversed"),
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1,
                                             "ci": [float("nan"), 1.0]}}},
     "json.loads accepts NaN, which compares false against everything"),
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1, "ci": [-0.5, 1.0]}}},
     "a probability bound outside [0, 1]"),
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1, "label": 7}}},
     "label is not a string"),
    # A pattern name that cannot describe this committee.
    ({"committee": ["a", "b"], "patterns": {"3": {"k": 1, "n": 2}}},
     "pattern sums to more than the committee"),
    ({"committee": ["a"], "patterns": {"0/1": {"k": 1, "n": 1}}},
     "a zero part means a group with no models in it"),
    ({"committee": ["a"], "patterns": {"2/-1": {"k": 1, "n": 1}}},
     "int() accepts a minus sign, so this summed to 1 and passed the sum check"),
    # A committee that can never match a real run.
    ({"committee": [], "patterns": {"1": {"k": 1, "n": 1}}},
     "empty committee would disable the mismatch guard entirely"),
    ({"committee": ["a", "a"], "patterns": {"1/1": {"k": 1, "n": 1}}},
     "a repeated model mismatches forever"),
    ({"committee": ["a"], "patterns": {}}, "no patterns"),
    ({"committee": "a,b", "patterns": {"2": {"k": 1, "n": 2}}},
     "committee is a string, not a list"),
    ([], "top level is not an object"),
    # The model_performance section, which prior_confidence and mock_ocsr read.
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1}},
      "model_performance": "not-an-object"}, "section is a string"),
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1}},
      "model_performance": {"decimer": {"accuracy": "90%"}}},
     "accuracy is a string"),
    ({"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1}},
      "model_performance": {"decimer": {"accuracy": 0.9, "n": 0}}},
     "an accuracy backed by zero observations"),
    ({"patterns": {"1": {"k": 1, "n": 2}}}, "no committee at all"),
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


@pytest.mark.parametrize("mutate, why", [
    (lambda r: r["vision_llms"]["shim"].update({"x": {}}), "shim entry has no wire_name"),
    (lambda r: r["vision_llms"]["shim"].update({"x": "y"}), "shim entry is a string"),
    (lambda r: r["vision_llms"]["alcf"].update({"x": "y"}), "alcf entry is a string"),
    (lambda r: r["specialists"]["decimer"].pop("latency_s"), "no latency_s"),
    (lambda r: r.update({"defaults": []}), "defaults is a list"),
    (lambda r: r["specialists"]["decimer"].update({"worker": 7}), "worker is a number"),
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


def test_the_tie_break_follows_the_priority_it_is_given():
    """Which model wins a tie decides what the all-different bucket measures.

    Ordered so the answer differs from "first in the results list": a test where the
    priority head is also the list head passes with the rule deleted entirely, which
    is how this went uncovered. The calibration table records the priority it was
    fitted under precisely because the number depends on it.
    """
    results = _results(("ocsrglyph", "CCC"), ("decimer", "CCO"))
    assert core.vote(results, ["decimer", "ocsrglyph"])["winner"] == "CCO"
    assert core.vote(results, ["ocsrglyph", "decimer"])["winner"] == "CCC"

    # And the table's own recorded rule is what the ensemble must use.
    table = core.load_calibration()
    assert core.tie_break_order(table) == table["committee"]
    assert core.tie_break_order({"committee": ["a", "b"]}) == ["a", "b"]


def test_canonicalize_is_stereo_blind_by_default():
    """Most reference labels carry no stereochemistry, so scoring with it marks a
    model wrong for correctly reading a wedge bond, and splits one molecule into two
    answers in vote()."""
    chiral, flat = "C[C@H](N)C(=O)O", "CC(N)C(=O)O"
    assert core.canonicalize(chiral) == core.canonicalize(flat)
    assert core.canonicalize(chiral, stereo=True) != core.canonicalize(flat)


def test_the_shipped_tie_break_matches_its_own_measured_accuracies():
    """The packaged table's priority must be the ranking its own numbers imply.

    The order decides who wins an even split, and the all-different bucket measures
    how often that first model was right, so a priority that disagrees with the
    accuracies in the same file would be measuring something nobody expects. This
    order was written by hand before the fitter derived it; the assertion is what
    keeps the two in step.
    """
    table = core.load_calibration()
    performance = table["model_performance"]
    by_accuracy = sorted(performance, key=lambda m: -performance[m]["accuracy"])
    assert core.tie_break_order(table) == by_accuracy


@pytest.mark.parametrize("tie_break, why", [
    ("model priority: b,a", "a typo in the marker falls back to the committee order"),
    (["b", "a"], "a JSON array is a plausible hand edit and does not parse"),
    ("model-priority:", "an empty order"),
    ("model-priority: zzz", "a name that is not in the committee"),
    ("model-priority: a", "only part of the committee"),
    ("model-priority: a,b,c", "a name too many"),
])
def test_an_unusable_tie_break_is_rejected_at_load(tie_break, why):
    """Falling back silently would vote one way and quote a number measured another.

    Every parse failure used to return the committee's arbitrary JSON order, so a
    single mistyped character swapped which model's answer was returned while the
    confidence stayed the same, with nothing in the result to reveal it.
    """
    table = {"committee": ["a", "b"], "tie_break": tie_break,
             "patterns": {"1/1": {"k": 1, "n": 2}}}
    with pytest.raises(ValueError, match="tie_break"):
        core._validate_calibration(table, "test")

    # Absent is fine: an older table falls back to its committee order.
    core._validate_calibration(
        {"committee": ["a", "b"], "patterns": {"1/1": {"k": 1, "n": 2}}}, "test")


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
