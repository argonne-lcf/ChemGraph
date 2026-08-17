"""Hermetic tests for the OCSR tool and backend layers.

No network, no conda envs, no model loads: the backends are monkeypatched. What is
under test here is the dispatch and the contract assembly, not the models.

A live test that actually runs a specialist is marked ``llm`` so it only runs under
``pytest --run-llm`` (see tests/conftest.py).
"""

from __future__ import annotations

import os
import pathlib

import pytest

from chemgraph.tools import ocsr_backends as backends
from chemgraph.tools import ocsr_core as core
from chemgraph.tools import ocsr_models as models
from chemgraph.tools import ocsr_tools as tools


@pytest.fixture
def png(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import Draw

    path = tmp_path / "mol.png"
    Draw.MolToImage(Chem.MolFromSmiles("CCO"), size=(120, 120)).save(path)
    return str(path)


def _narrow(smiles="CCO", ok=True, model="decimer", **kw):
    base = {"ok": ok, "smiles": smiles, "raw": "", "model_used": model,
            "cold_start": False, "latency_s": 0.1, "error": ""}
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Input validation, before any model runs
# ---------------------------------------------------------------------------


def test_core_never_raises(monkeypatch, png):
    """Even an exploding backend comes back as ok=False."""
    def boom(*a, **k):
        raise RuntimeError("worker exploded")

    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])
    monkeypatch.setattr(backends, "smiles_from_specialist", boom)
    r = tools.image_to_smiles_core(png, backend="decimer")
    assert r["ok"] is False
    assert "worker exploded" in r["error"]


# ---------------------------------------------------------------------------
# Single-model dispatch
# ---------------------------------------------------------------------------


def test_auto_fails_loudly_when_nothing_is_installed(monkeypatch, png):
    """It must NOT silently fall through to an LLM.

    Falling through would make the same image give a DECIMER answer on one machine
    and a Maverick answer on another, with different accuracy, and nothing in the
    return would make the user look.
    """
    monkeypatch.setattr(backends, "available_specialists", lambda: [])
    r = tools.image_to_smiles_core(png)
    assert r["ok"] is False
    assert r["backend_used"] == "none"
    assert "alcf" in r["error"]  # names the alternative


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------


def _ensemble(monkeypatch, answers: dict):
    monkeypatch.setattr(backends, "available_specialists", lambda: list(answers))
    monkeypatch.setattr(
        backends, "smiles_from_specialist",
        lambda name, b, **k: _narrow(smiles=answers[name], model=name,
                                     ok=answers[name] is not None),
    )


@pytest.mark.parametrize("answers, pattern, has_number", [
    ({"decimer": "CCO", "molnextr": "CCO", "molscribe": "CCO", "ocsrglyph": "CCO"},
     "4", True),
    ({"decimer": "CCO", "molnextr": "CCO", "molscribe": "CCO", "ocsrglyph": "CCN"},
     "3/1", True),
    ({"decimer": "CCO", "molnextr": "CCO", "molscribe": "CCN", "ocsrglyph": "CCN"},
     "2/2", False),   # below the sample floor: a label, no decimal
    ({"decimer": "C1=CC=CC=C1", "molnextr": "c1ccccc1",
      "molscribe": "c1ccccc1", "ocsrglyph": "c1ccccc1"}, "4", True),
    ({"decimer": "CCO", "molnextr": "CCO", "molscribe": "CCO", "ocsrglyph": "@@@junk"},
     "3/1", True),    # an abstention is a dissenting vote, not a smaller committee
])
def test_the_ensemble_pattern_and_confidence_track_the_votes(
        monkeypatch, png, answers, pattern, has_number):
    """One shape covers unanimity, a majority, a tie, Kekule/aromatic, and abstention.

    The Kekule case is silent when broken: grouping by raw string instead of
    canonical form drops unanimity from 289/422 benchmark items to 12/422, and the
    confidence feature stops working while still returning plausible numbers.
    """
    _ensemble(monkeypatch, answers)
    r = tools.image_to_smiles_core(png, backend="ensemble")
    assert r["ok"] is True
    assert r["agreement"] == pattern
    assert (r["confidence"] is not None) is has_number
    # basis names where the number came from, so it is set only when there is one.
    assert r["basis"] == ("agreement" if has_number else None)
    assert sum(int(x) for x in r["agreement"].split("/")) == len(answers)


def test_ensemble_on_a_partial_install_refuses_to_guess(monkeypatch, png):
    """A three-model committee must not be scored with the four-model table."""
    _ensemble(monkeypatch, {"decimer": "CCO", "molnextr": "CCO", "molscribe": "CCO"})
    r = tools.image_to_smiles_core(png, backend="ensemble")
    assert r["ok"] is True                      # we still have an answer
    assert r["smiles"] == "CCO"
    assert r["confidence"] is None              # but no number for it
    assert "committee_mismatch" in r["confidence_unavailable_reason"]
    assert "committee_mismatch" in r["warning"]


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------


def test_llm_backend_reports_no_confidence(monkeypatch, png):
    monkeypatch.setattr(backends, "smiles_from_llm",
                        lambda b, m, backend, model=None, **k:
                        _narrow(model="meta-llama/Llama-4-Maverick-17B-128E-Instruct"))
    r = tools.image_to_smiles_core(png, backend="alcf")
    assert r["ok"] is True
    assert r["confidence"] is None
    assert r["basis"] is None
    assert r["backend_used"] == "alcf"


@pytest.mark.parametrize("backend, var", [
    ("alcf", "ALCF_ACCESS_TOKEN"),
    ("shim", "ARGO_SHIM_API_KEY"),
])
def test_a_missing_credential_names_the_variable(monkeypatch, png, backend, var):
    """The error has to name the variable, since that is the whole fix."""
    monkeypatch.delenv(var, raising=False)
    r = tools.image_to_smiles_core(png, backend=backend)
    assert r["ok"] is False
    assert var in r["error"]


# ---------------------------------------------------------------------------
# Model-name resolution
# ---------------------------------------------------------------------------


def test_model_names_are_translated_and_endpoints_are_not_interchangeable():
    """The shim wants claudeopus48, not claude-opus-4.8. Nobody should memorize that.

    And an ALCF name is not a shim name: sending one the other's spelling fails
    mid-run with "Invalid model", so it is refused before any call is made.
    """
    assert models.resolve_model("shim", "argo:claude-opus-4.8") == "claudeopus48"
    assert models.resolve_model("shim", "claudeopus48") == "claudeopus48"
    with pytest.raises(ValueError):
        models.resolve_model("alcf", "argo:claude-opus-4.8")


def test_non_vision_models_are_refused_before_a_call_is_made():
    """A text-only model handed an image describes a picture it never saw.

    That is worse than an error, so it is refused up front rather than discovered
    from a confident wrong answer.
    """
    with pytest.raises(ValueError, match="not a known"):
        models.resolve_model("shim", "argo:gpt-4")
    with pytest.raises(ValueError, match="not a known"):
        models.resolve_model("alcf", "gpt-4o")


# ---------------------------------------------------------------------------
# The @tool wrappers
# ---------------------------------------------------------------------------


def test_tools_are_registered_with_the_expected_arguments():
    assert tools.image_to_smiles.name == "image_to_smiles"
    assert set(tools.image_to_smiles.args) == {"image_path", "backend", "model"}
    assert set(tools.validate_smiles.args) == {"smiles"}


def test_tool_docstring_documents_every_contract_key():
    """The docstring is the only part of this tool an agent reads, and no test
    catches a wrong one. This at least pins that the keys are all mentioned."""
    doc = tools.image_to_smiles.description
    for key in ["confidence", "confidence_label", "n_fragments", "cold_start",
                "backend_used", "warning", "votes"]:
        assert key in doc


def test_validate_smiles_tool_runs_through_the_tool_interface():
    out = tools.validate_smiles.invoke({"smiles": "CCO.CCN"})
    assert out["valid"] is True
    assert out["n_fragments"] == 2


def test_image_to_smiles_tool_delegates_to_core(monkeypatch, png):
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])
    monkeypatch.setattr(backends, "smiles_from_specialist",
                        lambda n, b, **k: _narrow())
    out = tools.image_to_smiles.invoke({"image_path": png})
    assert out["ok"] is True
    assert out["smiles"] == "CCO"


# ---------------------------------------------------------------------------
# Live: needs a real specialist installed. Skipped unless --run-llm.
# ---------------------------------------------------------------------------


@pytest.mark.llm
def test_live_default_specialist_reads_a_real_image(png):
    default = models.DEFAULT_SPECIALIST
    if default not in backends.available_specialists():
        pytest.skip(f"{default} is not installed on this host")
    r = tools.image_to_smiles_core(png, backend=default)
    assert r["ok"] is True, r["error"]
    assert core.canonicalize(r["smiles"]) is not None
    assert r["confidence"] is None  # one model gives no per-image number
    opted_in = tools.image_to_smiles_core(png, backend=default,
                                          report_solo_accuracy=True)
    assert opted_in["confidence"] == core.prior_confidence(default)["p"]


# ---------------------------------------------------------------------------
# Unparseable and prose output from a specialist
# ---------------------------------------------------------------------------


def _worker_returns(monkeypatch, smiles):
    """Make the worker client hand back a given string, bypassing the real models."""
    class FakeClient:
        def predict(self, model, image_bytes, timeout_s=None):
            return {"ok": bool(smiles), "smiles": smiles, "error": "",
                    "infer_s": 0.1, "cold_start": False}

    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])
    monkeypatch.setattr(backends, "_specialist_client", lambda: FakeClient())


@pytest.mark.parametrize("output", [
    "@@@not a molecule",                 # unparseable
    "I cannot process images.",          # a refusal RDKit reads as iodine
    "",                                  # empty
    None,                                # nothing at all
])
def test_specialist_output_that_is_not_a_molecule_is_a_failure(monkeypatch, png, output):
    """ok=True from a worker means "a string came back", not "it is a molecule".

    Silent when broken: these arrive as confident answers. RDKit parses "I cannot
    process images." as iodine, because I is an element and the parser stops at the
    space.
    """
    _worker_returns(monkeypatch, output)
    r = tools.image_to_smiles_core(png, backend="decimer")
    assert r["ok"] is False
    assert r["smiles"] is None


# ---------------------------------------------------------------------------
# Untrusted string length
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Path errors an agent produces routinely
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["not_a_directory", "name_too_long"])
def test_os_errors_return_rather_than_raise(tmp_path, kind, png):
    """The docstring promises a dict. These are reachable from a path an agent
    invented, and an exception becomes a hard graph failure instead of something the
    agent could correct."""
    if kind == "not_a_directory":
        f = tmp_path / "plain.txt"
        f.write_text("x")
        path = str(f / "inner.png")
    else:
        path = str(tmp_path / ("a" * 300 + ".png"))

    r = tools.image_to_smiles_core(path)  # must not raise
    assert r["ok"] is False
    assert r["error"]


@pytest.mark.skipif(os.name != "posix", reason="chmod 0o000 does not deny reads on Windows")
def test_an_unreadable_file_returns_rather_than_raises(tmp_path, png):
    """PermissionError is the OSError an agent hits most often, and it must not
    escape as an exception."""
    f = tmp_path / "locked.png"
    f.write_bytes(pathlib.Path(png).read_bytes())
    os.chmod(f, 0o000)

    r = tools.image_to_smiles_core(str(f))  # must not raise
    assert r["ok"] is False
    assert r["error"]


def _custom_table(tmp_path, accuracy=0.111):
    """A valid table whose accuracies differ from the packaged one."""
    import json as _json
    from importlib import resources

    packaged = resources.files("chemgraph.tools").joinpath(
        "ocsr_calibration_4model.json")
    table = _json.loads(packaged.read_text())
    for entry in table["model_performance"].values():
        entry["accuracy"] = accuracy
    path = tmp_path / "cal.json"
    path.write_text(_json.dumps(table))
    return str(path)


def test_calibration_argument_reaches_the_single_model_path(monkeypatch, png, tmp_path):
    """A refit table must apply to every backend, not only the ensemble.

    The specialist branch called prior_confidence with no table, so it always read
    the packaged default. A user who refit on their own images got their numbers from
    backend='ensemble' and someone else's from backend='auto', silently.
    """
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])
    monkeypatch.setattr(backends, "smiles_from_specialist",
                        lambda n, b, **k: _narrow(model="decimer"))
    custom = _custom_table(tmp_path)
    for backend in ("decimer", "auto"):
        r = tools.image_to_smiles_core(png, backend=backend, calibration=custom,
                                       report_solo_accuracy=True)
        assert r["confidence"] == 0.111, backend


def test_an_unreadable_table_keeps_the_prediction(monkeypatch, png):
    """A path typo must not throw away an answer that already cost inference.

    The ensemble loaded the table after running every model, so a bad path discarded
    a correct prediction and up to 80 s of cold-start work. Both paths now report the
    failure and keep the SMILES.
    """
    _ensemble(monkeypatch, {m: "CCO" for m in
                            ["decimer", "molnextr", "molscribe", "ocsrglyph"]})
    r = tools.image_to_smiles_core(png, backend="ensemble", calibration="/nope.json")
    assert r["ok"] is True
    assert r["smiles"] == "CCO"
    assert r["confidence"] is None
    assert "calibration_unreadable" in r["confidence_unavailable_reason"]


def test_auto_uses_an_installed_specialist_when_the_default_is_absent(monkeypatch, png):
    """Reporting "no local models are installed" while one is installed is false.

    The tool docstring tells an agent to answer that message by switching to a vision
    LLM, so a compliant agent abandoned a working local model for a less accurate
    remote one.
    """
    monkeypatch.setattr(backends, "available_specialists", lambda: ["molnextr"])
    monkeypatch.setattr(backends, "smiles_from_specialist",
                        lambda n, b, **k: _narrow(model=n))
    r = tools.image_to_smiles_core(png, backend="auto", report_solo_accuracy=True)
    assert r["ok"] is True
    assert r["backend_used"] == "molnextr"
    assert r["confidence"] == core.prior_confidence("molnextr")["p"]

    monkeypatch.setattr(backends, "available_specialists", lambda: [])
    r = tools.image_to_smiles_core(png, backend="auto")
    assert r["ok"] is False
    assert r["confidence_unavailable_reason"] == "no_specialists_installed"


@pytest.mark.parametrize("backend", ["bogus", ["auto"], {}, 7, None, 3.5])
def test_a_bad_backend_is_reported_and_never_raises(png, backend):
    """A typo must not be masked by a filesystem error, and must not raise.

    The name is checked before the image is opened, so a bad backend with a bad path
    says which one is wrong. The check itself used to raise TypeError on an
    unhashable value, before the catch-all could see it.
    """
    r = tools.image_to_smiles_core("/does/not/exist.png", backend=backend)
    assert r["ok"] is False
    assert "unknown backend" in r["error"]


def test_a_single_model_quotes_no_confidence_unless_asked(monkeypatch, png):
    """One model cannot say how likely it is to be right about THIS image.

    Its benchmark accuracy is a property of the model on someone else's images, so it
    is withheld by default and available behind report_solo_accuracy for a caller who
    knows what it means. Only the ensemble produces a per-image number.
    """
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])
    monkeypatch.setattr(backends, "smiles_from_specialist",
                        lambda n, b, **k: _narrow(model="decimer"))

    r = tools.image_to_smiles_core(png, backend="decimer")
    assert r["ok"] is True and r["smiles"] == "CCO"
    assert r["confidence"] is None
    assert r["confidence_unavailable_reason"] == "single_model_has_no_per_image_confidence"

    opted_in = tools.image_to_smiles_core(png, backend="decimer",
                                          report_solo_accuracy=True)
    assert opted_in["confidence"] == core.prior_confidence("decimer")["p"]
    assert opted_in["basis"] == "prior"


def test_the_ensemble_can_run_a_subset_of_the_installed_models(monkeypatch, png):
    """The committee is what a table describes, so a caller must be able to pick it.

    Someone with all four installed who fitted a table on two of them could not get a
    number: the ensemble ran everything and the committee check rejected the table.
    """
    _ensemble(monkeypatch, {m: "CCO" for m in
                            ["decimer", "molnextr", "molscribe", "ocsrglyph"]})

    r = tools.image_to_smiles_core(png, backend="ensemble",
                                   models_wanted=["decimer", "molnextr"])
    assert r["model_used"] == "decimer+molnextr"
    assert r["agreement"] == "2"

    unknown = tools.image_to_smiles_core(png, backend="ensemble",
                                         models_wanted=["nosuchmodel"])
    assert unknown["ok"] is False
    assert "not OCSR specialists" in unknown["error"]


def test_a_partial_install_says_which_models_are_missing(monkeypatch, png):
    """"committee_mismatch: [...] vs [...]" told a user their answer had no
    confidence without telling them what to do about it."""
    _ensemble(monkeypatch, {"decimer": "CCO", "molnextr": "CCO"})
    reason = tools.image_to_smiles_core(
        png, backend="ensemble")["confidence_unavailable_reason"]
    assert "molscribe" in reason and "ocsrglyph" in reason
    assert "ocsr_setup molscribe" in reason      # how to install what is missing
    assert "ocsr_calibrate --labels" in reason   # or refit for what is present


def test_an_unreadable_table_is_reported_on_the_single_model_path(monkeypatch, png):
    """Falling back to the packaged table would answer with the very numbers the
    caller was trying to replace."""
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])
    monkeypatch.setattr(backends, "smiles_from_specialist",
                        lambda n, b, **k: _narrow(model="decimer"))
    r = tools.image_to_smiles_core(png, backend="decimer", calibration="/nope.json",
                                   report_solo_accuracy=True)
    assert r["ok"] is True and r["smiles"] == "CCO"
    assert r["confidence"] is None
    assert "calibration_unreadable" in r["confidence_unavailable_reason"]


def test_the_ensemble_votes_by_the_order_the_table_records(monkeypatch, png, tmp_path):
    """The table's tie_break decides who wins an even split, so the tool must use it.

    Deleting either wiring point, here or in validate(), left the whole suite green:
    the shipped table's tie_break happens to equal its committee order, so a test
    built on it cannot tell the two apart. This table records the reverse.
    """
    import json as _json

    table = {
        "committee": ["decimer", "molnextr"],
        "tie_break": "model-priority: molnextr,decimer",
        "n_items": 100, "scoring": "stereo_blind", "min_n_for_point_estimate": 20,
        "patterns": {"1/1": {"k": 77, "n": 100, "p": round(77.5 / 101, 4),
                             "ci": [0.68, 0.85]}},
    }
    path = tmp_path / "cal.json"
    path.write_text(_json.dumps(table))

    _ensemble(monkeypatch, {"decimer": "CCC", "molnextr": "CCO"})
    r = tools.image_to_smiles_core(png, backend="ensemble", calibration=str(path))
    assert r["smiles"] == "CCO"          # molnextr wins, as the table says
    assert r["confidence"] == round(77.5 / 101, 4)
    assert r["basis"] == "agreement"


# ---------------------------------------------------------------------------
# Prompt mode
# ---------------------------------------------------------------------------


def _capture_prompt(monkeypatch, reply):
    """Run smiles_from_llm against a stub endpoint; return (system_prompt, result)."""
    seen = {}

    class _Msg:
        content = reply

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, model, messages, **kw):
            seen["system"] = messages[0]["content"]
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        def __init__(self, **kw):
            self.chat = _Chat()

    import openai
    monkeypatch.setattr(openai, "OpenAI", _Client)
    monkeypatch.setenv("ALCF_ACCESS_TOKEN", "x")
    return seen


def test_legacy_benchmark_mode_sends_the_vendored_prompt(monkeypatch, png):
    """The vendored prompt is what the shipped calibration table was fitted under, so
    the default path must keep sending it byte for byte."""
    from chemgraph.prompt import ocsr_prompt

    seen = _capture_prompt(monkeypatch, "CCO")
    data, mime = core.load_image_bytes(png)
    r = backends.smiles_from_llm(data, mime, "alcf")
    assert seen["system"] == ocsr_prompt.OCSR_SYSTEM_PROMPT
    assert r["smiles"] == "CCO"


def test_structured_mode_asks_for_json_and_reads_the_field(monkeypatch, png):
    """Structured mode's whole point is that the answer arrives in a named field
    instead of having to be guessed out of prose."""
    from chemgraph.prompt import ocsr_prompt

    seen = _capture_prompt(monkeypatch, '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}')
    data, mime = core.load_image_bytes(png)
    r = backends.smiles_from_llm(data, mime, "alcf", prompt_mode="structured")
    assert seen["system"] == ocsr_prompt.OCSR_STRUCTURED_SYSTEM_PROMPT
    assert seen["system"] != ocsr_prompt.OCSR_SYSTEM_PROMPT
    assert r["ok"] is True
    assert r["smiles"] == "CC(=O)Oc1ccccc1C(=O)O"


def test_an_unknown_prompt_mode_is_refused(png):
    """A typo must not silently fall back to a different prompt from the one the
    caller asked for, since the confidence numbers depend on which one ran."""
    r = tools.image_to_smiles_core(png, backend="alcf", prompt_mode="strutured")
    assert r["ok"] is False
    assert "prompt_mode" in r["error"]
