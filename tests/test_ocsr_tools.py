"""Tests for the OCSR tool layer.

Hermetic: no model is loaded and no network is touched. The backends are stubbed, so
these cover what this layer owns, namely model resolution, the failure paths, and the
shape of the result contract. Whether a model reads an image correctly is the models'
business and is checked by `examples/ocsr/run_ocsr.py` against known structures.
"""

import json
import os

import pytest

pytest.importorskip("rdkit")

from chemgraph.tools import ocsr_backends as backends  # noqa: E402
from chemgraph.tools import ocsr_core as core  # noqa: E402
from chemgraph.tools import ocsr_models as models  # noqa: E402
from chemgraph.tools import ocsr_tools as tools  # noqa: E402

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture
def image(tmp_path):
    """A real PNG, since the tool sniffs magic bytes before dispatching."""
    Draw = pytest.importorskip("rdkit.Chem.Draw")
    from rdkit import Chem

    path = tmp_path / "aspirin.png"
    Draw.MolToFile(Chem.MolFromSmiles(ASPIRIN), str(path), size=(300, 300))
    return str(path)


def _stub(monkeypatch, **narrow):
    """Make every specialist return one fixed narrow dict."""
    base = {"ok": False, "smiles": None, "raw": "", "model_used": "decimer",
            "cold_start": False, "latency_s": 0.1, "error": ""}
    base.update(narrow)
    monkeypatch.setattr(backends, "smiles_from_specialist",
                        lambda name, path: {**base, "model_used": name})
    monkeypatch.setattr(backends, "is_installed", lambda name: True)


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #


def test_model_choices_are_the_four_specialists_plus_llm():
    assert set(models.MODEL_CHOICES) == {
        "decimer", "molnextr", "molscribe", "ocsrglyph", "llm"
    }


def test_default_is_decimer():
    """Highest exact match of the four and the only plain pip install."""
    assert models.DEFAULT_SPECIALIST == "decimer"


def test_unset_model_uses_the_default_when_it_is_installed(monkeypatch, image):
    _stub(monkeypatch, ok=True, smiles=ASPIRIN)
    assert tools.image_to_smiles_core(image)["model_used"] == "decimer"


def test_unset_model_falls_back_to_the_llm_when_nothing_is_installed(monkeypatch, image):
    """The fallback exists so a machine with no specialists still answers."""
    monkeypatch.setattr(backends, "is_installed", lambda name: False)
    monkeypatch.setattr(backends, "available_specialists", lambda: [])
    seen = {}

    def fake_llm(image_bytes, mime, llm, structured=False):
        seen["called"] = True
        return {"ok": True, "smiles": ASPIRIN, "raw": "", "model_used": "gpt-4o",
                "cold_start": False, "latency_s": 1.0, "error": ""}

    monkeypatch.setattr(backends, "smiles_from_llm", fake_llm)
    result = tools.image_to_smiles_core(image)

    assert seen.get("called"), "no specialist installed should route to the LLM"
    assert result["ok"]


def test_unset_model_uses_an_installed_specialist_over_the_llm(monkeypatch, image):
    """A machine with molscribe but no decimer should use molscribe, not the LLM."""
    monkeypatch.setattr(backends, "is_installed", lambda name: name == "molscribe")
    monkeypatch.setattr(backends, "available_specialists", lambda: ["molscribe"])
    monkeypatch.setattr(
        backends, "smiles_from_specialist",
        lambda name, path: {"ok": True, "smiles": ASPIRIN, "raw": "",
                            "model_used": name, "cold_start": False,
                            "latency_s": 1.0, "error": ""})
    monkeypatch.setattr(
        backends, "smiles_from_llm",
        lambda *a, **k: pytest.fail("should not reach the LLM"))

    assert tools.image_to_smiles_core(image)["model_used"] == "molscribe"


def test_unknown_model_names_the_valid_choices(image):
    result = tools.image_to_smiles_core(image, model="molscribbe")

    assert not result["ok"]
    assert "molscribbe" in result["error"]
    for name in models.MODEL_CHOICES:
        assert name in result["error"], f"{name} should be offered in the error"


def test_model_name_is_case_insensitive(monkeypatch, image):
    _stub(monkeypatch, ok=True, smiles=ASPIRIN)
    assert tools.image_to_smiles_core(image, model="DECIMER")["model_used"] == "decimer"


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


def test_missing_image_says_so_without_calling_a_model(monkeypatch):
    monkeypatch.setattr(
        backends, "smiles_from_specialist",
        lambda *a, **k: pytest.fail("a missing file must fail before dispatch"))

    result = tools.image_to_smiles_core("/nonexistent/molecule.png")

    assert not result["ok"]
    assert "molecule.png" in result["error"]


def test_a_non_image_is_rejected_before_dispatch(monkeypatch, tmp_path):
    """Extension is attacker-controlled; the sniff is what decides."""
    fake = tmp_path / "not_really.png"
    fake.write_text("#!/bin/sh\necho hello\n")
    monkeypatch.setattr(
        backends, "smiles_from_specialist",
        lambda *a, **k: pytest.fail("a non-image must fail before dispatch"))

    assert not tools.image_to_smiles_core(str(fake))["ok"]


def test_backend_failure_is_reported_not_raised(monkeypatch, image):
    _stub(monkeypatch, ok=False, error="checkpoint missing at ~/ocsr-weights/x.pth")

    result = tools.image_to_smiles_core(image, model="molnextr")

    assert not result["ok"]
    assert "checkpoint missing" in result["error"]
    assert result["model_used"] == "molnextr"


def test_llm_model_without_a_bound_llm_explains_itself(image):
    result = tools.image_to_smiles_core(image, model="llm", llm=None)

    assert not result["ok"]
    assert "no LLM was bound" in result["error"]


# --------------------------------------------------------------------------- #
# The result contract
# --------------------------------------------------------------------------- #


def test_success_carries_validation(monkeypatch, image):
    _stub(monkeypatch, ok=True, smiles=ASPIRIN, latency_s=0.7)

    result = tools.image_to_smiles_core(image)

    assert result["ok"] and result["valid"]
    assert result["formula"] == "C9H8O4"
    assert result["n_fragments"] == 1
    assert result["latency_s"] == 0.7
    assert result["error"] == ""


def test_multiple_fragments_warn_without_failing(monkeypatch, image):
    """A salt is a real answer; the caller has to be told before optimizing it."""
    _stub(monkeypatch, ok=True, smiles="[Na+].CC(=O)[O-]")

    result = tools.image_to_smiles_core(image)

    assert result["ok"]
    assert result["n_fragments"] == 2
    assert "2 disconnected fragments" in result["warning"]


def test_result_keys_match_the_contract(monkeypatch, image):
    _stub(monkeypatch, ok=True, smiles=ASPIRIN)

    result = tools.image_to_smiles_core(image)

    assert set(result) == set(core.build_result())


def test_a_single_model_result_says_why_it_carries_no_confidence(monkeypatch, image):
    """The reason distinguishes "nothing to measure" from "the table failed".

    Silent when broken: an agent sees confidence=None on a perfectly good read and
    cannot tell whether to retry with a committee or to fix its calibration table.
    """
    _stub(monkeypatch, ok=True, smiles=ASPIRIN)

    result = tools.image_to_smiles_core(image)

    assert result["confidence"] is None
    assert result["confidence_unavailable_reason"] == (
        "single_model_has_no_per_image_confidence")
    assert result["backend_used"] == "specialist"
    assert result["agreement"] is None

def test_make_ocsr_tools_binds_the_llm(monkeypatch, image):
    """The whole point of the factory: model='llm' reaches the agent's own model."""
    sentinel = object()
    seen = {}

    def fake_llm(image_bytes, mime, llm, structured=False):
        seen["llm"] = llm
        return {"ok": True, "smiles": ASPIRIN, "raw": "", "model_used": "bound",
                "cold_start": False, "latency_s": 1.0, "error": ""}

    monkeypatch.setattr(backends, "smiles_from_llm", fake_llm)
    tool = tools.make_ocsr_tools(sentinel)[0]
    tool.invoke({"image_path": image, "model": "llm"})

    assert seen["llm"] is sentinel


def test_module_level_tool_has_no_llm(image):
    """It is bindable statically, so its fallback must fail loudly, not silently."""
    result = tools.image_to_smiles.invoke({"image_path": image, "model": "llm"})

    assert not result["ok"]
    assert "no LLM was bound" in result["error"]


def test_list_ocsr_models_reports_the_default_and_install_state():
    listing = tools.list_ocsr_models.invoke({})

    assert "[default]" in listing
    for name in models.MODEL_CHOICES:
        assert name in listing


def test_an_explicit_model_is_never_silently_swapped(monkeypatch, image):
    """Asking for molscribe and getting decimer would be a wrong answer, quietly.

    Only an unset `model` falls back. A named one either runs or explains why it
    cannot, so a user comparing two models can trust which produced the SMILES.
    """
    monkeypatch.setattr(backends, "is_installed", lambda name: name == "decimer")
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])
    monkeypatch.setattr(
        backends, "smiles_from_specialist",
        lambda name, path: {"ok": False, "smiles": None, "raw": "",
                            "model_used": name, "cold_start": True,
                            "latency_s": 0.0,
                            "error": f"{name} is not installed in this environment"})
    monkeypatch.setattr(
        backends, "smiles_from_llm",
        lambda *a, **k: pytest.fail("an explicit model must not fall back to the LLM"))

    result = tools.image_to_smiles_core(image, model="molscribe")

    assert not result["ok"]
    assert result["model_used"] == "molscribe", "the named model must be reported"
    assert "molscribe" in result["error"]


def test_install_hint_names_the_extra():
    """One command installs all four, so every missing model points at the same one."""
    assert "chemgraph[ocsr]" in backends._install_hint()


def test_importing_the_backends_does_not_import_torchvision():
    """ocsr_agent is imported by llm_agent for every workflow, OCSR or not.

    A module-scope torch import would make an install with no torch fail on an
    unrelated workflow, so the preload is deferred to the model-loading path.
    """
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(backends.__file__).read_text())
    module_scope = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.Try))
        for sub in ast.walk(node)
        if isinstance(sub, ast.Import)
        for alias in sub.names
    }
    assert "torchvision" not in module_scope, (
        "importing torchvision at module scope breaks installs without torch"
    )


def test_torchvision_is_preloaded_before_a_specialist_loads():
    """Loading torchvision into a process that already holds TensorFlow segfaults.

    DECIMER pulls in TensorFlow, so reading one image with it and the next with any
    torch model kills the process unless torchvision was imported first. The call
    looks pointless and is easy to delete during a tidy-up; this fails if that
    happens.
    """
    import inspect

    assert "_preload_torchvision()" in inspect.getsource(backends._get_model), (
        "_get_model must preload torchvision before loading any specialist"
    )


def test_the_ocsr_modules_import_and_run_without_torch(tmp_path):
    """An install with no torch must still reach the LLM path and the error paths.

    llm_agent imports ocsr_agent for every workflow, so a torch import anywhere on
    this module's import path breaks unrelated workflows on such an install. Runs in
    a subprocess because the guard has to be in place before the first import.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import sys
        BLOCK = ("torch", "torchvision", "tensorflow")

        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCK:
                    raise ImportError("blocked " + name)
                return None

        sys.meta_path.insert(0, Blocker())

        from chemgraph.tools import ocsr_backends, ocsr_core, ocsr_models, ocsr_tools

        result = ocsr_tools.image_to_smiles_core("/nonexistent/molecule.png")
        assert not result["ok"], result
        leaked = [m for m in BLOCK if m in sys.modules]
        assert not leaked, leaked
        print("ok")
    """)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr[-2000:]


def test_structured_selects_the_json_prompt(monkeypatch, image):
    """The flag has to reach the system message, which is the only thing it changes."""
    from chemgraph.prompt import ocsr_prompt

    seen = {}

    def fake_llm_call(image_bytes, mime, llm, structured=False):
        seen["structured"] = structured
        return {"ok": True, "smiles": ASPIRIN, "raw": "", "model_used": "m",
                "cold_start": False, "latency_s": 1.0, "error": ""}

    monkeypatch.setattr(backends, "smiles_from_llm", fake_llm_call)
    tools.image_to_smiles_core(image, model="llm", structured=True, llm=object())
    assert seen["structured"] is True

    tools.image_to_smiles_core(image, model="llm", llm=object())
    assert seen["structured"] is False, "the default must stay the vendored prompt"

    assert "json" in ocsr_prompt.OCSR_STRUCTURED_SYSTEM_PROMPT.lower() or (
        '"smiles"' in ocsr_prompt.OCSR_STRUCTURED_SYSTEM_PROMPT)


def test_a_structured_null_reply_is_an_error_not_a_molecule(monkeypatch, image):
    """The JSON prompt lets a model say "not a molecule"; that must not parse."""
    class NullLLM:
        model_name = "fake"

        def invoke(self, messages):
            class R:
                content = '{"smiles": null}'
            return R()

    result = tools.image_to_smiles_core(image, model="llm", structured=True,
                                        llm=NullLLM())

    assert not result["ok"]
    assert result["smiles"] is None


def test_structured_is_ignored_by_the_specialists(monkeypatch, image):
    """Specialists take no prompt, so the flag must not change their dispatch."""
    _stub(monkeypatch, ok=True, smiles=ASPIRIN)

    plain = tools.image_to_smiles_core(image, model="decimer")
    flagged = tools.image_to_smiles_core(image, model="decimer", structured=True)

    assert plain == flagged


def test_smiles_is_canonicalized_so_models_agree_on_one_string(monkeypatch, image):
    """DECIMER writes Kekule where the others write aromatic; same molecule."""
    _stub(monkeypatch, ok=True, smiles="CC(=O)OC1=CC=CC=C1C(=O)O")

    assert tools.image_to_smiles_core(image)["smiles"] == ASPIRIN


def test_canonicalization_keeps_stereochemistry(monkeypatch, image):
    """validate_smiles_core's canonical_smiles drops stereo for benchmark scoring.

    Returning that would silently discard what a model correctly read off the wedge
    bonds, so the returned SMILES is canonicalized separately with stereo kept.
    """
    penicillin = "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O"
    _stub(monkeypatch, ok=True, smiles=penicillin)

    assert "@" in tools.image_to_smiles_core(image)["smiles"]


def test_an_unparseable_smiles_is_still_reported(monkeypatch, image):
    """Canonicalization cannot silently blank an answer RDKit rejects."""
    _stub(monkeypatch, ok=True, smiles="C1CC")

    result = tools.image_to_smiles_core(image)

    assert result["smiles"] == "C1CC" and not result["valid"]


# --------------------------------------------------------------------------- #
# Committee runs
# --------------------------------------------------------------------------- #


def _committee(monkeypatch, answers: dict):
    """Make each specialist return its own SMILES, or None to abstain."""
    monkeypatch.setattr(backends, "available_specialists", lambda: list(answers))

    def one(name, path):
        smiles = answers[name]
        return {"ok": smiles is not None, "smiles": smiles, "raw": "",
                "model_used": name, "cold_start": False, "latency_s": 0.1,
                "error": "" if smiles else "unreadable"}

    monkeypatch.setattr(backends, "smiles_from_specialist", one)


ALL_FOUR = ["decimer", "molnextr", "molscribe", "ocsrglyph"]


def test_a_unanimous_committee_earns_the_top_confidence(monkeypatch, image):
    """What the whole feature is for: agreement the caller can act on."""
    _committee(monkeypatch, dict.fromkeys(ALL_FOUR, ASPIRIN))

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["ok"] and result["agreement"] == "4"
    assert result["confidence"] > 0.99
    assert result["confidence_label"] == "unanimous"
    assert result["backend_used"] == "ensemble"
    assert result["model_used"] == "+".join(ALL_FOUR)


def test_a_split_committee_reports_the_lower_number(monkeypatch, image):
    _committee(monkeypatch, {"decimer": ASPIRIN, "molnextr": ASPIRIN,
                             "molscribe": ASPIRIN, "ocsrglyph": "CCO"})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["agreement"] == "3/1"
    assert result["smiles"] == core.canonicalize(ASPIRIN)
    assert 0.9 < result["confidence"] < 0.99
    assert result["abstained"] == {}
    assert set(result["votes"]) == {core.canonicalize(ASPIRIN), core.canonicalize("CCO")}


def test_a_committee_that_all_failed_returns_no_molecule(monkeypatch, image):
    _committee(monkeypatch, dict.fromkeys(ALL_FOUR, None))

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert not result["ok"]
    assert result["smiles"] is None
    assert result["confidence_unavailable_reason"] == "no_prediction"
    assert len(result["abstained"]) == 4


def test_a_partial_install_gets_no_number_and_is_told_why(monkeypatch, image):
    """Silent when broken: the caller paid for an ensemble and gets a bare None."""
    _committee(monkeypatch, {"decimer": ASPIRIN, "molnextr": ASPIRIN})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["ok"] and result["smiles"] == core.canonicalize(ASPIRIN)
    assert result["confidence"] is None
    assert "committee_mismatch" in result["confidence_unavailable_reason"]
    assert "committee_mismatch" in result["warning"]


def test_an_unreadable_table_keeps_the_prediction(monkeypatch, image, tmp_path):
    """A typo in the path must not throw away four inferences."""
    _committee(monkeypatch, dict.fromkeys(ALL_FOUR, ASPIRIN))

    result = tools.image_to_smiles_core(image, ensemble=True,
                                        calibration=str(tmp_path / "nope.json"))

    assert result["ok"] and result["smiles"] == core.canonicalize(ASPIRIN)
    assert result["agreement"] == "4"
    assert result["confidence"] is None
    assert result["confidence_unavailable_reason"].startswith("calibration_unreadable")
    assert "could not be read" in result["warning"]


def test_models_wanted_votes_the_subset_a_table_describes(monkeypatch, image,
                                                          tmp_path):
    """Someone with four installed but a two-model table can still get a number."""
    _committee(monkeypatch, dict.fromkeys(ALL_FOUR, ASPIRIN))
    table = tmp_path / "cal.json"
    table.write_text(json.dumps({
        "committee": ["decimer", "molnextr"],
        "tie_break": "model-priority: decimer,molnextr",
        "patterns": {"2": {"k": 19, "n": 20, "p": 0.9286}},
    }))

    result = tools.image_to_smiles_core(
        image, ensemble=True, calibration=str(table),
        models_wanted=["decimer", "molnextr"])

    assert result["agreement"] == "2"
    assert result["confidence"] == 0.9286
    assert result["model_used"] == "decimer+molnextr"


@pytest.mark.parametrize("wanted, expect", [
    ("decimer", "must be a list"),
    ([], "is empty"),
    (["nosuch"], "not OCSR specialists"),
    (["decimer", "molscribe"], "not installed"),
])
def test_an_unusable_models_wanted_says_what_to_pass_instead(monkeypatch, image,
                                                             wanted, expect):
    """Silent when broken: a bare string iterates per character and votes nothing."""
    _committee(monkeypatch, {"decimer": ASPIRIN, "molnextr": ASPIRIN})

    result = tools.image_to_smiles_core(image, ensemble=True, models_wanted=wanted)

    assert not result["ok"]
    assert expect in result["error"]


def test_the_tool_exposes_the_committee_to_an_agent():
    """The flag has to be callable and the description has to recommend it.

    Silent when broken: the argument is invisible to the agent, or the description
    tells it no confidence number exists, and the feature is never reached.
    """
    built = tools.make_ocsr_tools(llm=None)[0]
    schema = built.args_schema.model_json_schema()

    assert "ensemble" in schema["properties"]
    assert "models_wanted" in schema["properties"]
    assert "No confidence number is reported" not in built.description
    assert "ensemble" in built.description and "confidence" in built.description


def test_the_tie_break_comes_from_the_table_and_not_the_registry(monkeypatch, image,
                                                                 tmp_path):
    """An all-different vote must be decided by the order the table was fit under.

    Silent when broken: the registry's order wins the tie, so the answer comes from
    one model while the number quoted for it was measured for another's. A committee
    check cannot see this, because it compares sorted names.
    """
    _committee(monkeypatch, {"decimer": "CCO", "molnextr": "CCN",
                             "molscribe": "CCC", "ocsrglyph": "CCF"})
    table = tmp_path / "cal.json"
    # Reverse of the registry order, so the two disagree about who wins.
    table.write_text(json.dumps({
        "committee": ALL_FOUR,
        "tie_break": "model-priority: ocsrglyph,molscribe,molnextr,decimer",
        "patterns": {"1/1/1/1": {"k": 5, "n": 20, "p": 0.2619}},
    }))

    result = tools.image_to_smiles_core(image, ensemble=True,
                                        calibration=str(table))

    assert result["agreement"] == "1/1/1/1"
    assert result["smiles"] == core.canonicalize("CCF")


def test_an_unreadable_table_still_lists_the_models(monkeypatch, tmp_path):
    """The listing is how a user finds out what they have; it must not depend on it."""
    monkeypatch.setenv("CHEMGRAPH_OCSR_CALIBRATION", str(tmp_path / "nope.json"))

    listing = tools.list_ocsr_models.func()

    assert "decimer" in listing and "ocsrglyph" in listing


PENICILLIN = "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O"


def test_a_unanimous_committee_keeps_the_stereochemistry_it_read(monkeypatch, image):
    """Four models reading one enantiomer must not answer with the racemate.

    Silent when broken: vote() groups stereo-blind, so its winner is the stripped
    key. Returning that directly hands back less chemistry than a single-model call
    does, with 0.9989 confidence attached and nothing in the result to show it.
    """
    _committee(monkeypatch, dict.fromkeys(ALL_FOUR, PENICILLIN))

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["agreement"] == "4"
    assert result["smiles"] == core.canonicalize(PENICILLIN, stereo=True)
    assert "@" in result["smiles"]
    # The same molecule a single model would have returned for the same reading.
    assert result["smiles"] == tools.image_to_smiles_core(image)["smiles"]


def test_the_strongest_model_in_the_group_supplies_the_stereochemistry(monkeypatch,
                                                                       image,
                                                                       tmp_path):
    """Members of one group can agree on the skeleton and differ on the wedges.

    Silent when broken: which enantiomer comes back depends on dict ordering.
    """
    flat = core.canonicalize(PENICILLIN)  # same skeleton, no stereocentres marked
    _committee(monkeypatch, {"decimer": flat, "molnextr": PENICILLIN,
                             "molscribe": PENICILLIN, "ocsrglyph": flat})
    table = tmp_path / "cal.json"
    table.write_text(json.dumps({
        "committee": ALL_FOUR,
        "tie_break": "model-priority: molnextr,molscribe,decimer,ocsrglyph",
        "patterns": {"4": {"k": 99, "n": 100, "p": 0.9851}},
    }))

    result = tools.image_to_smiles_core(image, ensemble=True,
                                        calibration=str(table))

    assert result["agreement"] == "4"
    # molnextr leads this table's priority and read the wedges.
    assert result["smiles"] == core.canonicalize(PENICILLIN, stereo=True)


def test_a_fragment_warning_does_not_hide_the_calibration_warning(monkeypatch,
                                                                  image, tmp_path):
    """Both conditions are independent, so both caveats have to survive.

    Silent when broken: a salt read against a missing table looks like a normal
    fragment warning, and the missing confidence shows only as a bare None.
    """
    _committee(monkeypatch, dict.fromkeys(ALL_FOUR, "[Na+].CC(=O)[O-]"))

    result = tools.image_to_smiles_core(image, ensemble=True,
                                        calibration=str(tmp_path / "nope.json"))

    assert result["n_fragments"] == 2
    assert "disconnected" in result["warning"]
    assert "could not be read" in result["warning"]


def test_a_single_model_label_bands_its_measured_accuracy(monkeypatch, image,
                                                          tmp_path):
    """The one confidence question a single read can answer.

    Silent when broken: the field is always 'unavailable', so it carries nothing
    and an agent has no way to tell a strong model from a weak one at the call site.
    """
    _stub(monkeypatch, ok=True, smiles=ASPIRIN)
    table = tmp_path / "cal.json"
    table.write_text(json.dumps({
        "committee": ["decimer"], "patterns": {"1": {"k": 1, "n": 1}},
        "model_performance": {"decimer": {"accuracy": 0.996, "k": 498, "n": 500}},
    }))
    monkeypatch.setenv("CHEMGRAPH_OCSR_CALIBRATION", str(table))

    result = tools.image_to_smiles_core(image, model="decimer")

    assert result["confidence"] is None  # still no per-image number
    assert result["confidence_label"] == "unanimous"  # the model's own accuracy


def test_both_backends_word_the_fragment_warning_the_same(monkeypatch, image):
    """An agent that learns to act on one wording must see it from the other.

    Silent when broken: the two copies drift and a committee's salt warning stops
    matching whatever the agent was told to look for.
    """
    salt = "[Na+].CC(=O)[O-]"
    _stub(monkeypatch, ok=True, smiles=salt)
    single = tools.image_to_smiles_core(image)

    _committee(monkeypatch, dict.fromkeys(ALL_FOUR, salt))
    committee = tools.image_to_smiles_core(image, ensemble=True)

    assert "disconnected" in single["warning"]
    assert single["warning"] == committee["warning"]


def test_both_tools_of_the_same_name_offer_the_committee(monkeypatch, image):
    """A static binding must not silently lack the feature the factory has.

    Silent when broken: two tools called image_to_smiles take different arguments,
    and which one an agent got decides whether ensemble=True is even accepted.
    """
    _committee(monkeypatch, dict.fromkeys(ALL_FOUR, ASPIRIN))

    static = tools.image_to_smiles.func(image, ensemble=True)
    built = tools.make_ocsr_tools(llm=None)[0].func(image, ensemble=True)

    assert static["agreement"] == built["agreement"] == "4"
    assert static["smiles"] == built["smiles"]

    # Every committee argument, not only the flag that turns it on.
    committee_args = {"ensemble", "models_wanted"}
    static_schema = tools.image_to_smiles.args_schema.model_json_schema()
    built_schema = tools.make_ocsr_tools(llm=None)[0].args_schema.model_json_schema()
    assert committee_args <= set(static_schema["properties"])
    assert committee_args <= set(built_schema["properties"])


def test_a_model_that_never_ran_is_not_a_dissenting_vote(monkeypatch, image,
                                                         tmp_path):
    """A missing checkpoint is not evidence about the image.

    Silent when broken: three specialists install without their checkpoints, the
    fourth reads the image correctly, and the pattern is 1/1/1/1, so the tool
    reports 0.3772 for an answer its own table measures at 0.8989.
    """
    monkeypatch.setattr(backends, "available_specialists", lambda: ALL_FOUR)

    def one(name, path):
        if name == "decimer":
            return {"ok": True, "smiles": ASPIRIN, "raw": "", "model_used": name,
                    "cold_start": False, "latency_s": 0.1, "error": "", "ran": True}
        return {"ok": False, "smiles": None, "raw": "", "model_used": name,
                "cold_start": False, "latency_s": 0.0, "ran": False,
                "error": "checkpoint not found"}

    monkeypatch.setattr(backends, "smiles_from_specialist", one)

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["ok"] and result["smiles"] == core.canonicalize(ASPIRIN)
    assert result["agreement"] == "1"  # one voter, not four
    assert result["confidence"] is None
    assert "committee_mismatch" in result["confidence_unavailable_reason"]


def test_no_specialist_able_to_run_says_which_and_why(monkeypatch, image):
    monkeypatch.setattr(backends, "available_specialists", lambda: ALL_FOUR)
    monkeypatch.setattr(backends, "smiles_from_specialist", lambda n, p: {
        "ok": False, "smiles": None, "raw": "", "model_used": n, "cold_start": False,
        "latency_s": 0.0, "ran": False, "error": "checkpoint not found"})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert not result["ok"]
    assert "no specialist could run" in result["error"]
    assert "checkpoint not found" in result["error"]
    # Distinct from nothing being pip-installed: the remedy is a download, not
    # an install, and a caller branching on the reason has to tell them apart.
    assert result["confidence_unavailable_reason"] == "no_specialist_could_run"


def test_a_prefixed_priority_still_picks_the_strongest_model(monkeypatch, image,
                                                             tmp_path):
    """vote() stores bare names, so the tie-break order has to be compared bare.

    Silent when broken: the membership test matches nothing and the stereo answer
    falls back to dict insertion order, which is the arbitrary choice the
    tie-break exists to replace.
    """
    # An even split, so the priority decides rather than the count.
    flat = core.canonicalize(PENICILLIN)
    _committee(monkeypatch, {"decimer": flat, "molnextr": PENICILLIN,
                             "molscribe": PENICILLIN, "ocsrglyph": flat})
    table = tmp_path / "cal.json"
    # A table has to prefix both fields to load: the validator makes tie_break name
    # exactly the committee. It then reports a mismatch, since vote() bares the
    # names, but the stereo pick still runs and must not fall back to dict order.
    prefixed = ["local:" + m for m in ALL_FOUR]
    table.write_text(json.dumps({
        "committee": prefixed,
        "tie_break": "model-priority: " + ",".join(
            ["local:molnextr", "local:decimer", "local:molscribe",
             "local:ocsrglyph"]),
        "patterns": {"4": {"k": 99, "n": 100, "p": 0.9851}},
    }))

    result = tools.image_to_smiles_core(image, ensemble=True,
                                        calibration=str(table))

    assert result["smiles"] == core.canonicalize(PENICILLIN, stereo=True)


def test_a_single_model_label_uses_the_table_the_caller_named(monkeypatch, image,
                                                              tmp_path):
    """An explicit calibration path outranks the env var and the packaged table.

    Silent when broken: a caller who refit gets their own numbers from a committee
    and someone else's from every single-model read.
    """
    _stub(monkeypatch, ok=True, smiles=ASPIRIN)
    mine = tmp_path / "mine.json"
    mine.write_text(json.dumps({
        "committee": ["decimer"], "patterns": {"1": {"k": 1, "n": 1}},
        "model_performance": {"decimer": {"accuracy": 0.55, "k": 110, "n": 200}},
    }))

    result = tools.image_to_smiles_core(image, model="decimer",
                                        calibration=str(mine))

    assert result["confidence_label"] == "conflicting"  # 0.55, from the given table


def test_a_shrunken_committee_says_what_stopped_the_others(monkeypatch, image):
    """The mismatch text advises an install that is already done.

    Silent when broken: three models are installed and cannot load, the warning
    tells the user to install them, and the checkpoint path that would actually
    fix it is collected and then dropped.
    """
    monkeypatch.setattr(backends, "available_specialists", lambda: ALL_FOUR)

    def one(name, path):
        if name == "decimer":
            return {"ok": True, "smiles": ASPIRIN, "raw": "", "model_used": name,
                    "cold_start": False, "latency_s": 0.5, "error": "", "ran": True}
        return {"ok": False, "smiles": None, "raw": "", "model_used": name,
                "cold_start": True, "latency_s": 3.0, "ran": False,
                "error": f"{name} checkpoint is missing at /weights/{name}"}

    monkeypatch.setattr(backends, "smiles_from_specialist", one)

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert "checkpoint is missing" in result["warning"]
    for name in ["molnextr", "molscribe", "ocsrglyph"]:
        assert name in result["warning"]


def test_a_split_the_table_never_measured_says_so(monkeypatch, image, tmp_path):
    """The third confidence-less path, which used to surface as a bare None.

    Silent when broken: an agent sees no number, no warning, and a label of
    'unknown' that no document explains.
    """
    _committee(monkeypatch, {"decimer": ASPIRIN, "molnextr": "CCO"})
    table = tmp_path / "cal.json"
    table.write_text(json.dumps({
        "committee": ["decimer", "molnextr"],
        "patterns": {"2": {"k": 19, "n": 20, "p": 0.9286}},
    }))

    result = tools.image_to_smiles_core(image, ensemble=True,
                                        calibration=str(table))

    assert result["confidence_unavailable_reason"] == "unknown_pattern"
    assert result["confidence_label"] == "unknown"
    assert "no '1/1' bucket" in result["warning"]


def test_a_thin_bucket_still_hands_back_its_interval(monkeypatch, image, tmp_path):
    """Withholding the number is only defensible if the interval survives.

    Silent when broken: the docs offer the interval as what a thin bucket gives
    instead of a point estimate, and the caller receives neither.
    """
    _committee(monkeypatch, {"decimer": ASPIRIN, "molnextr": ASPIRIN,
                             "molscribe": "CCO", "ocsrglyph": "CCO"})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["agreement"] == "2/2"
    assert result["confidence"] is None
    assert result["confidence_interval"] == [0.3119, 0.8195]


def test_a_shrunken_committee_that_finds_a_fitting_table_still_says_so(monkeypatch,
                                                                       image,
                                                                       tmp_path):
    """A table can describe the survivors, so no mismatch fires to carry the news.

    Silent when broken: three of four models never ran, the answer comes back with
    a real confidence and an empty warning, and the only trace is a latency nobody
    reads.
    """
    monkeypatch.setattr(backends, "available_specialists", lambda: ALL_FOUR)

    def one(name, path):
        if name == "decimer":
            return {"ok": True, "smiles": ASPIRIN, "raw": "", "model_used": name,
                    "cold_start": False, "latency_s": 0.5, "error": "", "ran": True}
        return {"ok": False, "smiles": None, "raw": "", "model_used": name,
                "cold_start": True, "latency_s": 3.0, "ran": False,
                "error": f"{name} checkpoint is missing"}

    monkeypatch.setattr(backends, "smiles_from_specialist", one)
    table = tmp_path / "cal.json"
    table.write_text(json.dumps({
        "committee": ["decimer"],
        "patterns": {"1": {"k": 88, "n": 100, "p": round(88.5 / 101, 4)}},
    }))

    result = tools.image_to_smiles_core(image, ensemble=True,
                                        calibration=str(table))

    assert result["confidence"] is not None  # the table fits the one that ran
    assert "could not run" in result["warning"]
    assert "molscribe" in result["warning"]


def test_the_failure_list_in_a_warning_is_bounded(monkeypatch, image):
    """The errors are joined per model and go back into an agent's context.

    Silent when broken: a long weights directory pushes the warning past 6 kB.
    """
    monkeypatch.setattr(backends, "available_specialists", lambda: ALL_FOUR)
    monkeypatch.setattr(backends, "smiles_from_specialist", lambda n, p: {
        "ok": n == "decimer", "smiles": ASPIRIN if n == "decimer" else None,
        "raw": "", "model_used": n, "cold_start": False, "latency_s": 0.1,
        "ran": n == "decimer", "error": "" if n == "decimer" else "x" * 3000})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert len(result["warning"]) < 1000


def test_the_model_listing_reports_the_table_and_not_the_registry(monkeypatch,
                                                                  tmp_path):
    """A user who refit on their own images must see their own accuracies.

    Silent when broken: the listing quotes the benchmark figures compiled into the
    registry, which say nothing about the images this install actually reads.
    """
    table = tmp_path / "cal.json"
    table.write_text(json.dumps({
        "committee": ["decimer"], "patterns": {"1": {"k": 1, "n": 1}},
        "model_performance": {"decimer": {"accuracy": 0.12, "k": 6, "n": 50}},
    }))
    monkeypatch.setenv("CHEMGRAPH_OCSR_CALIBRATION", str(table))

    listing = tools.list_ocsr_models.func()

    assert "0.120 exact match" in listing  # this table states no p, so accuracy
    assert "0.899" not in listing  # the registry figure


def test_an_unreadable_table_still_returns_the_read(monkeypatch, image, tmp_path):
    """The label degrades; the SMILES the model produced still comes back."""
    _stub(monkeypatch, ok=True, smiles=ASPIRIN)
    monkeypatch.setenv("CHEMGRAPH_OCSR_CALIBRATION", str(tmp_path / "nope.json"))

    result = tools.image_to_smiles_core(image, model="decimer")

    assert result["ok"]
    assert result["confidence_label"] == "unavailable"


def test_the_example_script_marks_a_missing_checkpoint(monkeypatch, capsys):
    """describe_models defaults ready to installed, so omitting it is vacuous.

    Silent when broken: the example tells a first-time user setup is finished, and
    the next call fails on a checkpoint that was never downloaded.
    """
    import runpy
    import pathlib

    monkeypatch.setattr(backends, "available_specialists", lambda: ALL_FOUR)
    monkeypatch.setattr(backends, "usable_specialists", lambda: ["decimer"])
    monkeypatch.setattr(backends, "smiles_from_specialist", lambda name, path: {
        "ok": True, "smiles": ASPIRIN, "raw": "", "model_used": name,
        "cold_start": False, "latency_s": 0.1, "error": ""})
    script = (pathlib.Path(__file__).parent.parent / "examples" / "ocsr"
              / "run_ocsr.py")

    runpy.run_path(str(script))["run_direct"](None)

    listing = capsys.readouterr().out

    assert "molnextr" in listing and "installed, checkpoint missing" in listing
    assert "molnextr: 0.835 exact match (your table), 4.35s/image (ready)" \
        not in listing


def test_the_example_script_prints_what_the_tool_reports(monkeypatch, tmp_path):
    """Both read accuracies through the same call, so a refit moves both.

    Silent when broken: the example prints the registry's benchmark figures while
    the tool prints the user's own, and nothing says why they differ.
    """
    table = tmp_path / "cal.json"
    table.write_text(json.dumps({
        "committee": ["decimer"], "patterns": {"1": {"k": 1, "n": 1}},
        "model_performance": {"decimer": {"accuracy": 0.4242, "k": 42, "n": 99}},
    }))
    monkeypatch.setenv("CHEMGRAPH_OCSR_CALIBRATION", str(table))

    assert tools.measured_accuracies()["decimer"]["accuracy"] == 0.4242
    assert "0.424 exact match" in tools.list_ocsr_models.func()


def test_the_listing_quotes_the_number_the_tool_quotes(monkeypatch, tmp_path):
    """One install cannot report two accuracies for one model.

    Silent when broken: the listing prints the table's raw rate while a single-model
    read reports the Jeffreys estimate, and nothing says why they differ.
    """
    table = tmp_path / "cal.json"
    table.write_text(json.dumps({
        "committee": ["decimer"], "min_n_for_point_estimate": 20,
        "patterns": {"1": {"k": 90, "n": 100}},
        "model_performance": {"decimer": {"accuracy": 0.9, "p": 0.8960,
                                          "k": 90, "n": 100}},
    }))
    monkeypatch.setenv("CHEMGRAPH_OCSR_CALIBRATION", str(table))

    shown = tools.measured_accuracies()["decimer"]["accuracy"]

    assert shown == core.prior_confidence("decimer")["p"] == 0.8960
    assert "0.896 exact match" in tools.list_ocsr_models.func()


def test_the_listing_withholds_what_the_tool_withholds(monkeypatch, tmp_path):
    """Below the floor the tool quotes no number, so neither can the listing.

    Silent when broken: a model measured on seven images is listed at 1.000 while
    asking the tool for the same model returns no figure at all.
    """
    table = tmp_path / "cal.json"
    table.write_text(json.dumps({
        "committee": ["decimer"], "min_n_for_point_estimate": 20,
        "patterns": {"1": {"k": 7, "n": 7}},
        "model_performance": {"decimer": {"accuracy": 1.0, "p": None,
                                          "k": 7, "n": 7}},
    }))
    monkeypatch.setenv("CHEMGRAPH_OCSR_CALIBRATION", str(table))

    listing = tools.list_ocsr_models.func()

    assert core.prior_confidence("decimer")["p"] is None
    # Positively, so falling back to the registry's packaged figure fails here too:
    # a negative assertion alone is satisfied by printing 0.899 instead of 1.000.
    assert "measured on too few images to quote (your table)" in listing
    assert "0.899" not in listing


def test_a_model_the_table_omits_falls_back_to_the_registry(monkeypatch, tmp_path):
    """A table that never measured a model is not a table that withheld its figure.

    Silent when broken: model_performance returns {} for an uncovered model, and
    `{} is not None` reports every one of them as measured on too few images, by a
    table that does not mention them.
    """
    table = tmp_path / "cal.json"
    table.write_text(json.dumps({
        "committee": ["decimer"], "min_n_for_point_estimate": 20,
        "patterns": {"1": {"k": 90, "n": 100}},
        "model_performance": {"decimer": {"accuracy": 0.9, "p": 0.896,
                                          "k": 90, "n": 100}},
    }))
    monkeypatch.setenv("CHEMGRAPH_OCSR_CALIBRATION", str(table))

    listing = tools.list_ocsr_models.func()

    assert "0.896 exact match (your table)" in listing  # covered
    assert "0.835 exact match" in listing               # uncovered: the registry
    assert listing.count("too few images") == 0


def test_the_listing_separates_installed_from_ready(monkeypatch, tmp_path):
    """Three of four need a checkpoint the extra cannot fetch.

    Silent when broken: the listing says "installed", the user believes setup is
    finished, and the next call fails on a missing 1.1 GB file.
    """
    monkeypatch.setenv("CHEMGRAPH_OCSR_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(backends, "available_specialists", lambda: ALL_FOUR)

    listing = tools.list_ocsr_models.func()

    assert "decimer [default]" in listing and "(ready)" in listing
    assert listing.count("installed, checkpoint missing") == 3


def test_a_missing_checkpoint_error_quotes_a_command_that_downloads_it(monkeypatch,
                                                                       tmp_path):
    """The one install step the extra cannot do, so most first runs stop here.

    Silent when broken: the error points at a README the user then has to find,
    parse, and adapt to their own weights directory.
    """
    monkeypatch.setenv("CHEMGRAPH_OCSR_WEIGHTS_DIR", str(tmp_path))

    _, error = backends._resolve_weights("molnextr")

    assert "hf download" in error
    # os.path.join, so the separator is the platform's own.
    assert f"--local-dir {os.path.join(str(tmp_path), 'molnextr')}" in error


def test_decimer_needs_no_checkpoint_and_is_never_called_unready(monkeypatch,
                                                                 tmp_path):
    """It caches its own weights, so a weights directory says nothing about it."""
    monkeypatch.setenv("CHEMGRAPH_OCSR_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])

    assert backends.usable_specialists() == ["decimer"]


L_ALANINE = "C[C@@H](N)C(=O)O"
D_ALANINE = "C[C@H](N)C(=O)O"


def test_the_strongest_model_supplies_the_stereochemistry(monkeypatch, image):
    """Not the majority: the models rank the same way on stereochemistry.

    Silent when broken: three weaker readings outvote the one measured most likely
    to be right. On the benchmark's stereo-bearing items DECIMER is exact 76.6% of
    the time against molnextr 57.4%, ocsrglyph 48.9% and molscribe 40.4%, and where
    the two rules disagreed the priority pick matched gold 55 times to nil.
    """
    _committee(monkeypatch, {"decimer": D_ALANINE, "molnextr": L_ALANINE,
                             "molscribe": L_ALANINE, "ocsrglyph": L_ALANINE})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["agreement"] == "4"
    assert result["smiles"] == core.canonicalize(D_ALANINE, stereo=True)
    # The three that read otherwise are not silently discarded.
    assert "different stereochemistries" in result["warning"]

def test_a_stereo_split_says_the_number_does_not_cover_it(monkeypatch, image):
    """The table was fitted stereo-blind, so it scores the skeleton only.

    Silent when broken: an even split on the wedge bonds reports 0.9989 unanimous
    with nothing to show half the committee read the other enantiomer.
    """
    _committee(monkeypatch, {"decimer": D_ALANINE, "molnextr": D_ALANINE,
                             "molscribe": L_ALANINE, "ocsrglyph": L_ALANINE})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["confidence"] > 0.99  # the skeleton really is unanimous
    assert "different stereochemistries" in result["warning"]
    assert "stereo-blind" in result["warning"]


def test_agreeing_on_the_stereochemistry_raises_no_caveat(monkeypatch, image):
    """The warning has to stay off the path where every model read the same thing."""
    _committee(monkeypatch, dict.fromkeys(ALL_FOUR, L_ALANINE))

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["smiles"] == core.canonicalize(L_ALANINE, stereo=True)
    assert result["warning"] == ""


def test_reading_no_stereocentre_is_not_a_conflict(monkeypatch, image):
    """A model that marked nothing read less, it did not read differently.

    Silent when broken: on the benchmark this warns on all 115 groups that hold
    more than one stereo form instead of the 112 that lose something, and an agent
    taught to treat the caveat as meaningful learns to ignore it.
    """
    flat = core.canonicalize(L_ALANINE)
    _committee(monkeypatch, {"decimer": L_ALANINE, "molnextr": L_ALANINE,
                             "molscribe": L_ALANINE, "ocsrglyph": flat})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["smiles"] == core.canonicalize(L_ALANINE, stereo=True)
    assert result["warning"] == ""


def test_dropping_a_stereocentre_the_minority_read_is_a_conflict(monkeypatch, image):
    """The answer loses information one model supplied, so it has to say so."""
    flat = core.canonicalize(L_ALANINE)
    _committee(monkeypatch, {"decimer": flat, "molnextr": flat,
                             "molscribe": flat, "ocsrglyph": L_ALANINE})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["smiles"] == flat
    assert "different stereochemistries" in result["warning"]


DIENE_BOTH = "C/C=C/C=C/C"
DIENE_ONE = "C/C=C/C=CC"


def test_an_answer_that_says_more_is_not_a_conflict(monkeypatch, image):
    """Reading one of two double bonds where the answer reads both loses nothing.

    Silent when broken: 29 of the 115 real multi-form groups warn about a
    difference in which the answer already carries everything the others read.
    """
    _committee(monkeypatch, {"decimer": DIENE_BOTH, "molnextr": DIENE_ONE,
                             "molscribe": DIENE_ONE, "ocsrglyph": DIENE_ONE})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["smiles"] == core.canonicalize(DIENE_BOTH, stereo=True)
    assert result["warning"] == ""


def test_an_answer_that_says_less_is_a_conflict(monkeypatch, image):
    """The mirror case: the answer drops a centre the others assigned."""
    _committee(monkeypatch, {"decimer": DIENE_ONE, "molnextr": DIENE_BOTH,
                             "molscribe": DIENE_BOTH, "ocsrglyph": DIENE_BOTH})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["smiles"] == core.canonicalize(DIENE_ONE, stereo=True)
    assert "different stereochemistries" in result["warning"]


def test_the_caveat_names_who_backs_the_answer(monkeypatch, image):
    """The largest group is regularly not the answer's.

    Silent when broken: a leading count reads as the answer's support when it
    belongs to the readings the answer overruled, which is half of the real cases.
    """
    flat = core.canonicalize(L_ALANINE)
    _committee(monkeypatch, {"decimer": flat, "molnextr": L_ALANINE,
                             "molscribe": L_ALANINE, "ocsrglyph": L_ALANINE})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["smiles"] == flat
    assert "decimer's reading" in result["warning"]
    assert "3 other models read otherwise" in result["warning"]
    # The largest group is the one that lost, so its size must not lead.
    assert "3 different" not in result["warning"]


def test_a_centre_the_answer_lacks_entirely_is_still_a_conflict(monkeypatch, image):
    """The comparison is keyed on the atom, so a one-sided centre cannot be skipped.

    Silent when broken: zipping the two elementwise truncates to the shorter list,
    so an answer carrying no stereo at all covers everything by default. On the
    benchmark that hid 26 of the 112 groups where the answer discards a reading,
    including a sulfur centre three models never saw.
    """
    plain = "O=[N+]([O-])c1cccnc1S(=O)(=O)O"
    marked = "O=[N+]([O-])c1cccnc1[S@SP3](=O)(=O)O"
    _committee(monkeypatch, {"decimer": plain, "molnextr": marked,
                             "molscribe": plain, "ocsrglyph": plain})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["smiles"] == core.canonicalize(plain, stereo=True)
    assert "different stereochemistries" in result["warning"]


@pytest.mark.parametrize("answer, other, covered, why", [
    ("O=[N+]([O-])c1cccnc1S(=O)(=O)O",
     "O=[N+]([O-])c1cccnc1[S@SP3](=O)(=O)O",
     False, "the answer drops a square-planar sulfur centre entirely"),
    ("O=[S@SP1](=O)(NCCCBr)c1ccccc1",
     "O=[S@SP2](=O)(NCCCBr)c1ccccc1",
     False, "two square-planar assignments RDKit both describes as NoValue"),
    ("C[C@H](Cl)[C@@H](C)N", "C[C@H](N)[C@@H](C)Cl",
     False, "same atom indices, opposite substituents at each centre"),
    ("O=C(O)[C@@H](O)C(O)[C@H](O)C(=O)O",
     "O=C(O)[C@@H](O)C(O)[C@@H](O)C(=O)O",
     False, "the meso form perceives one fewer centre than its diastereomer"),
    ("C/C=C/C=C/C", "C/C=C/C=CC", True, "the answer assigns a bond the other left open"),
    ("C[C@@H](N)C(=O)O", "C[C@@H](N)C(=O)O", True, "identical"),
])
def test_stereo_coverage_holds_where_element_comparison_does_not(monkeypatch, image,
                                                                 answer, other,
                                                                 covered, why):
    """Each of these defeats a comparison of perceived stereo elements.

    Silent when broken: RDKit returns those in an unpromised order, in a count
    that changes when one assignment settles another centre, and with a descriptor
    of NoValue for every class but tetrahedral. Comparing the stereoisomers each
    form still admits needs no alignment and reads every class.
    """
    # decimer leads the shipped tie-break, so its reading becomes the answer.
    _committee(monkeypatch, {"decimer": answer, "molnextr": other,
                             "molscribe": other, "ocsrglyph": other})

    result = tools.image_to_smiles_core(image, ensemble=True)

    warned = "different stereochemistries" in (result["warning"] or "")
    assert warned is not covered, why


@pytest.mark.parametrize("model", [5, 5.0, True, ["decimer"], {"a": 1}])
def test_a_model_that_is_not_a_name_is_refused_not_raised(model):
    """models_wanted is type-checked and this was not.

    Silent when broken: .strip() on an int raises AttributeError past the contract
    that says every failure comes back as ok=False.
    """
    result = tools.image_to_smiles_core("/nonexistent.png", model=model)

    assert not result["ok"]
    assert "must be a name" in result["error"]


def test_an_image_path_that_is_not_a_path_is_refused_not_raised():
    result = tools.image_to_smiles_core(42)

    assert not result["ok"]
    assert "must be a path" in result["error"]


# Raffinose, a trisaccharide with 14 unassigned centres flat. Chosen over sucrose
# because sucrose sits at exactly 512 isomers, the cap itself, so the enumeration
# still completes and the truncation branch below is never taken.
RAFFINOSE_FLAT = "OCC1OC(OCC2OC(OC3(CO)OC(CO)C(O)C3O)C(O)C(O)C2O)C(O)C(O)C1O"
RAFFINOSE = ("OC[C@H]1O[C@H](OC[C@H]2O[C@H](O[C@]3(CO)O[C@H](CO)[C@@H](O)[C@@H]3O)"
             "[C@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@@H](O)[C@@H]1O")
# The same skeleton with only the middle ring's two centres read, in each of two
# incompatible ways. Both leave 4096 forms open, so both take the fallback.
RAFFINOSE_ONE_RING = ("OCC1OC(OC[C@H]2O[C@H](OC3(CO)OC(CO)C(O)C3O)C(O)C(O)C2O)"
                      "C(O)C(O)C1O")
RAFFINOSE_OTHER_RING = ("OCC1OC(OC[C@@H]2O[C@@H](OC3(CO)OC(CO)C(O)C3O)C(O)C(O)C2O)"
                        "C(O)C(O)C1O")


def test_a_sugar_read_flat_by_the_rest_is_not_a_conflict(monkeypatch, image):
    """Past the enumeration cap a truncated set is not a subset of anything.

    Silent when broken: any sugar or polyol with 9 or more unassigned centres warns
    that the committee read different stereochemistries when three of four read
    none at all, which is the case the coverage rule exists to allow.
    """
    _committee(monkeypatch, {"decimer": RAFFINOSE, "molnextr": RAFFINOSE_FLAT,
                             "molscribe": RAFFINOSE_FLAT,
                             "ocsrglyph": RAFFINOSE_FLAT})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert result["agreement"] == "4"
    assert result["warning"] == ""


def test_a_sugar_read_with_a_flipped_centre_still_conflicts(monkeypatch, image):
    """The fallback must not wave everything through: it still catches a real one.

    Both readings leave 4096 forms open, so both take the fallback, and they
    disagree on the two centres each does assign.
    """
    _committee(monkeypatch, {"decimer": RAFFINOSE_ONE_RING,
                             "molnextr": RAFFINOSE_OTHER_RING,
                             "molscribe": RAFFINOSE_OTHER_RING,
                             "ocsrglyph": RAFFINOSE_OTHER_RING})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert "different stereochemistries" in result["warning"]


def test_an_unread_square_planar_centre_is_still_less_detail(monkeypatch, image):
    """Reading one centre where the others read none stays less detail.

    Silent when broken: comparing _chiralPermutation to catch a metal-centre
    conflict was tried and reverted, because the index depends on the neighbour
    order the SMILES was written in and 12 of 18 same-geometry pairs came back as
    conflicts. This is the case that regression broke.
    """
    pad = "C(O)" * 11
    cis = f"Cl[Pt@SP1](Cl)([NH3])[NH2]{pad}O"
    flat = f"Cl[Pt](Cl)([NH3])[NH2]{pad}O"
    _committee(monkeypatch, {"decimer": cis, "molnextr": flat,
                             "molscribe": flat, "ocsrglyph": flat})

    assert tools.image_to_smiles_core(image, ensemble=True)["warning"] == ""


def test_the_truncation_fallback_is_the_path_these_two_take(monkeypatch, image):
    """Guards the pair above: at or under the cap they never exercise it.

    Silent when broken: the molecule drifts to one the enumeration can finish, both
    tests keep passing, and the branch they exist for stops being reached at all.
    """
    from rdkit import Chem
    from rdkit.Chem.EnumerateStereoisomers import (GetStereoisomerCount,
                                                   StereoEnumerationOptions)

    opts = StereoEnumerationOptions(onlyUnassigned=True, unique=True,
                                    tryEmbedding=False, maxIsomers=tools._MAX_ISOMERS)
    for smiles in (RAFFINOSE_FLAT, RAFFINOSE_ONE_RING, RAFFINOSE_OTHER_RING):
        mol = Chem.MolFromSmiles(core.canonicalize(smiles, stereo=True))
        assert GetStereoisomerCount(mol, options=opts) > tools._MAX_ISOMERS

    # And the cap is above what a real prediction reaches: the largest count over
    # the benchmark's 2777 predictions is 512, so no ordinary molecule pays for the
    # fallback's weaker comparison.
    assert tools._MAX_ISOMERS >= 512


def test_the_no_specialist_error_is_bounded(monkeypatch, image):
    """It joins one backend error per model and goes into an agent's context.

    Silent when broken: a long weights directory measured 32,932 characters.
    """
    monkeypatch.setattr(backends, "available_specialists", lambda: ALL_FOUR)
    monkeypatch.setattr(backends, "smiles_from_specialist", lambda n, p: {
        "ok": False, "smiles": None, "raw": "", "model_used": n, "cold_start": False,
        "latency_s": 0.1, "ran": False, "error": "x" * 9000})

    result = tools.image_to_smiles_core(image, ensemble=True)

    assert not result["ok"]
    assert len(result["error"]) < 1000


def test_a_str_subclass_model_cannot_run_its_own_code(image):
    """isinstance passes for a subclass, which is free to override strip.

    Silent when broken: whatever the subclass raises escapes the never-raises
    contract from inside the guard added to keep a non-string out.
    """
    class HostileStrip(str):
        def strip(self, *a):
            raise RuntimeError("boom")

    class HostileStr(str):
        def __str__(self):
            raise RuntimeError("boom")

    for hostile in (HostileStrip("decimer"), HostileStr("decimer")):
        assert isinstance(tools.image_to_smiles_core(image, model=hostile), dict)


def test_the_models_wanted_error_does_not_call_a_caller_supplied_repr(monkeypatch,
                                                                     image):
    """The guard reports what it rejected, and repr is caller code too.

    Silent when broken: the object's own exception replaces the error message the
    guard exists to produce.
    """
    class Hostile:
        def __repr__(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(backends, "available_specialists", lambda: ALL_FOUR)

    result = tools.image_to_smiles_core(image, ensemble=True,
                                        models_wanted=[Hostile()])

    assert not result["ok"]
    assert "a Hostile" in result["error"]


@pytest.mark.parametrize("wanted", [[["decimer"]], [{"a": 1}], [set()], [None], [5]])
def test_an_unhashable_models_wanted_element_is_refused_not_raised(monkeypatch,
                                                                   image, wanted):
    """The guard tests membership in a dict, which hashes before it can report.

    Silent when broken: TypeError from inside the check written to prevent it.
    """
    monkeypatch.setattr(backends, "available_specialists", lambda: ALL_FOUR)

    result = tools.image_to_smiles_core(image, ensemble=True, models_wanted=wanted)

    assert not result["ok"]
    assert "not OCSR specialists" in result["error"]


@pytest.mark.parametrize("call, field", [
    (lambda: tools.image_to_smiles_core("/x" + "y" * 50000 + ".png"), "error"),
    (lambda: tools.image_to_smiles_core("/x.png", model="z" * 50000), "error"),
])
def test_no_result_field_echoes_an_unbounded_input(call, field):
    """Every message can quote a path, a model name or a backend error.

    Silent when broken: 100,000 characters of one bad argument go back into an
    agent's context.
    """
    assert len(call()[field]) <= core._MAX_MESSAGE_CHARS


@pytest.mark.parametrize("call, field", [
    (lambda: tools.image_to_smiles_core("/x" + "y" * 50000 + ".png"), "error"),
    (lambda: tools.image_to_smiles_core("/x.png", model="z" * 50000), "error"),
])
def test_the_echo_is_bounded_before_the_field_is(call, field):
    """The field cap has to hold four legitimate caveats, so it is too wide alone.

    Silent when broken: one 50,000-character argument fills the whole budget and
    pushes off the end the part of the message a user has to act on.
    """
    assert len(call()[field]) < 900


def test_the_cap_applies_to_the_warning_and_the_reason_too(monkeypatch, image):
    _committee(monkeypatch, dict.fromkeys(ALL_FOUR, ASPIRIN))

    result = tools.image_to_smiles_core(
        image, ensemble=True, calibration="/q" + "w" * 50000 + ".json")

    assert len(result["warning"]) <= core._MAX_MESSAGE_CHARS
    assert len(result["confidence_unavailable_reason"]) <= core._MAX_MESSAGE_CHARS
    # The echo is what keeps this well under the cap: the path appears in the
    # warning, and the actionable half of the sentence follows it.
    assert result["warning"].endswith("carries no confidence")


def test_a_legitimate_four_caveat_warning_is_not_truncated(monkeypatch, image):
    """A salt, read by a shrunken committee whose two absentees named a download.

    Silent when broken: the cap cuts the refit command off the mismatch note, which
    is the one part of that message a user can act on.
    """
    monkeypatch.setattr(backends, "available_specialists", lambda: ALL_FOUR)

    def one(name, path):
        if name in ("molscribe", "ocsrglyph"):
            return {"ok": False, "smiles": None, "raw": "", "model_used": name,
                    "cold_start": False, "latency_s": 0.1, "ran": False,
                    "error": "checkpoint not found at ~/ocsr-weights/x/model.pth; "
                             "download it with: hf download Org/Repo model.pth"}
        return {"ok": True, "smiles": "C[C@H](N)C(=O)O.Cl.O" if name == "decimer"
                else "CC(N)C(=O)O.Cl.O", "raw": "", "model_used": name,
                "cold_start": False, "latency_s": 0.1, "error": "", "ran": True}

    monkeypatch.setattr(backends, "smiles_from_specialist", one)

    warning = tools.image_to_smiles_core(image, ensemble=True)["warning"]

    assert not warning.endswith("...")
    assert "disconnected fragments" in warning
    assert "could not run" in warning
    assert "--models decimer,molnextr" in warning
