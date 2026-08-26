"""Tests for the OCSR tool layer.

Hermetic: no model is loaded and no network is touched. The backends are stubbed, so
these cover what this layer owns, namely model resolution, the failure paths, and the
shape of the result contract. Whether a model reads an image correctly is the models'
business and is checked by `examples/ocsr/run_ocsr.py` against known structures.
"""

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
