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

    def fake_llm(image_bytes, mime, llm):
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


def test_committee_keys_are_gone():
    """The ensemble surface belongs to the follow-up PR, not this contract."""
    contract = set(core.build_result())
    assert not contract & {"votes", "abstained", "agreement", "confidence"}


# --------------------------------------------------------------------------- #
# Tool wiring
# --------------------------------------------------------------------------- #


def test_make_ocsr_tools_binds_the_llm(monkeypatch, image):
    """The whole point of the factory: model='llm' reaches the agent's own model."""
    sentinel = object()
    seen = {}

    def fake_llm(image_bytes, mime, llm):
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


def test_install_hint_does_not_send_users_to_a_command_that_does_nothing():
    """`pip install chemgraph[ocsr]` reaches DECIMER only.

    The other three are unpublished, so telling a user to run the extra when
    MolNexTR is missing sends them to a command that changes nothing.
    """
    assert "chemgraph[ocsr]" in backends._install_hint("decimer")
    for name in ("molnextr", "molscribe", "ocsrglyph"):
        hint = backends._install_hint(name)
        assert "chemgraph[ocsr]" not in hint, f"{name} is not in the extra"
        assert "README" in hint, f"{name} should point at the install steps"
