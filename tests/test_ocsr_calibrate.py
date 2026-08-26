"""Tests for the OCSR recalibration CLI.

Hermetic: no model is loaded and no image is read. The fitter takes per-image
prediction rows, so everything below is built from literal SMILES.
"""

import pytest

pytest.importorskip("rdkit")

from chemgraph.tools import ocsr_calibrate as calibrate  # noqa: E402
from chemgraph.tools import ocsr_core as core  # noqa: E402


def _rows(*specs):
    """Build (per-image results, reference) pairs from {model: smiles} dicts."""
    return [([{"model": m, "smiles": s, "ok": s is not None, "error": ""}
              for m, s in preds.items()], ref) for preds, ref in specs]


def test_fit_calibration_counts_what_actually_happened():
    # 25 unanimous and right, 5 unanimous and wrong.
    rows = _rows(*([({"a": "CCO", "b": "CCO"}, "CCO")] * 25
                   + [({"a": "CCN", "b": "CCN"}, "CCO")] * 5))

    t = calibrate.fit_calibration(rows, ["a", "b"], min_n=20)

    cell = t["patterns"]["2"]
    assert (cell["k"], cell["n"]) == (25, 30)
    assert 0.80 < cell["p"] < 0.85  # Jeffreys on 25/30
    assert cell["ci"][0] < cell["p"] < cell["ci"][1]
    assert t["committee"] == ["a", "b"]


def test_a_fitted_table_is_usable_by_confidence():
    """Round trip: what the fitter writes, the reader must understand.

    Silent when broken: a table that fits cleanly and then reports unknown_pattern
    on every lookup leaves the user with no way to see which half is wrong.
    """
    t = calibrate.fit_calibration(_rows(*([({"a": "CCO", "b": "CCO"}, "CCO")] * 40)),
                                  ["a", "b"], min_n=20)

    got = core.confidence("2", t)

    assert got["p"] is not None and got["reason"] is None
    core._validate_calibration(t, "fitted")


def test_the_fitter_withholds_a_number_below_the_sample_floor():
    """False precision is what the floor exists to prevent.

    A bucket of seven with a quoted decimal reads as a measurement; its interval is
    50 points wide.
    """
    thin = calibrate.fit_calibration(
        _rows(*[({"a": "CCO", "b": "CCO"}, "CCO")] * 7), ["a", "b"], min_n=20)
    assert thin["patterns"]["2"]["n"] == 7
    assert thin["patterns"]["2"]["p"] is None
    assert core.confidence("2", thin)["label"].startswith("low_n_")


def test_the_tie_break_comes_from_the_data_not_the_argument_order():
    """--models is a set of names; nobody should have to know it is also a ranking.

    Silent when broken: the order decides who wins an even split, and the
    all-different bucket measures how often that first model was right. Taking it
    from the order a caller happened to type bakes an arbitrary choice into their
    table.
    """
    data = _rows(*([({"decimer": "CCC", "molnextr": "CCO"}, "CCO")] * 7
                   + [({"decimer": "CCO", "molnextr": "CCO"}, "CCO")] * 2
                   + [({"decimer": "CCO", "molnextr": "CCC"}, "CCO")] * 1))

    # molnextr is right 9/10 here and decimer 3/10, so molnextr must break ties.
    for order in (["decimer", "molnextr"], ["molnextr", "decimer"]):
        table = calibrate.fit_calibration(data, order)
        assert core.tie_break_order(table) == ["molnextr", "decimer"], order

    explicit = calibrate.fit_calibration(data, ["decimer", "molnextr"],
                                         tie_break=["decimer", "molnextr"])
    assert core.tie_break_order(explicit) == ["decimer", "molnextr"]


def test_report_renders_the_shipped_table():
    """The report reads through confidence(), so it cannot drift from the tool.

    Silent when broken: it used to read cell["label"], which the fitter writes only
    above the sample floor, so it crashed on the table ChemGraph ships.
    """
    text = calibrate.report(core.load_calibration(), 20)
    assert "unanimous" in text
    assert "low_n_conflicting" in text  # the 2/2 bucket, below the floor

    thin = {"committee": ["a"], "n_items": 5, "scoring": "stereo_blind",
            "patterns": {"1": {"k": 3, "n": 5}}}
    core._validate_calibration(thin, "test")
    assert "no interval" in calibrate.report(thin, 20)


def test_validate_votes_by_the_table_being_checked():
    """Comparing observed accuracy against a table means voting the table's way."""
    table = {
        "committee": ["a", "b"], "tie_break": "model-priority: b,a",
        "n_items": 1, "scoring": "stereo_blind",
        "patterns": {"1/1": {"k": 10, "n": 12, "p": round(10.5 / 13, 4)}},
    }
    core._validate_calibration(table, "test")

    # b is right and a is wrong: voting b first scores 1/1, voting a first scores 0/1.
    rows = _rows(({"a": "CCC", "b": "CCO"}, "CCO"))

    assert "1/1 = 100%" in calibrate.validate(rows, ["a", "b"], table)


def test_the_refit_instruction_names_this_module():
    """A table rejected for an edited p is told how to regenerate it.

    Silent when broken: the message points at a module that does not exist and the
    user gets ModuleNotFoundError from following it.
    """
    import importlib

    bad = {"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1, "p": 0.99}}}
    with pytest.raises(ValueError, match="ocsr_calibrate"):
        core._validate_calibration(bad, "test")
    assert importlib.import_module("chemgraph.tools.ocsr_calibrate")


def test_collect_passes_a_path_to_the_specialists(monkeypatch, tmp_path):
    """The backend takes a path, and collect must resolve it before handing it over.

    Silent when broken: an earlier signature took image bytes, so passing what
    load_image_bytes returns leaves every model reading a bytes object as a filename
    and abstaining on every image.
    """
    from chemgraph.tools import ocsr_backends as backends

    image = tmp_path / "mol.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    seen = []

    def one(name, path):
        seen.append(path)
        return {"ok": True, "smiles": "CCO", "raw": "", "model_used": name,
                "cold_start": False, "latency_s": 0.1, "error": ""}

    monkeypatch.setattr(backends, "smiles_from_specialist", one)

    rows = calibrate.collect([(str(image), "CCO")], ["a"], verbose=False)

    assert len(rows) == 1
    assert seen == [str(image)]


def test_collect_skips_an_unreadable_image_without_stopping(tmp_path):
    """One bad path in a label file must not abandon the rest of the run."""
    rows = calibrate.collect([("/nonexistent/mol.png", "CCO")], ["a"], verbose=False)
    assert rows == []


def test_validate_rejects_a_bad_table_before_spending_any_inference(monkeypatch,
                                                                    tmp_path,
                                                                    capsys):
    """A typo in a path must not cost a full run over every image.

    Silent when broken: the table loads after collect(), so hours of inference are
    discarded with a traceback to report an unparseable file.
    """
    from chemgraph.tools import ocsr_backends as backends

    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,smiles\n/nonexistent/a.png,CCO\n")
    bad = tmp_path / "cal.json"
    bad.write_text("{not json")
    monkeypatch.setenv("CHEMGRAPH_OCSR_CALIBRATION", str(bad))
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])

    def never(name, path):
        raise AssertionError("collect() ran before the table was checked")

    monkeypatch.setattr(backends, "smiles_from_specialist", never)

    code = calibrate.main(["--labels", str(labels), "--validate"])

    assert code == 2
    assert "cannot validate" in capsys.readouterr().err


def test_a_fitted_table_carries_the_same_rules_as_the_shipped_one():
    """A refit must be describable in the same terms as the default.

    Silent when broken: the shipped table records a rule the fitter never writes,
    so a user comparing the two cannot tell which differences are their data and
    which are the tool.
    """
    rows = _rows(*[({"a": "CCO", "b": "CCO"}, "CCO")] * 30)
    fitted = calibrate.fit_calibration(rows, ["a", "b"])
    shipped = core.load_calibration()

    for key in ["scoring", "abstention", "estimator", "label_rule",
                "model_performance_rule"]:
        assert fitted[key] == shipped[key], key


def test_an_empty_bucket_gets_the_whole_interval_and_not_a_crash():
    """Wilson divides by n. An unpopulated bucket must not raise."""
    ci, method = calibrate._interval(0, 0)
    assert ci == [0.0, 1.0]
    assert "no observations" in method


def test_a_fitted_table_records_which_interval_method_produced_it():
    """scipy is not a declared dependency, so the two methods must be told apart.

    Silent when broken: the same labels yield intervals differing by up to 5.5 pp
    on the small buckets, depending on whether scipy happened to be importable.
    """
    rows = _rows(*[({"a": "CCO", "b": "CCO"}, "CCO")] * 30)

    table = calibrate.fit_calibration(rows, ["a", "b"])

    assert table["interval_method"] == core.load_calibration()["interval_method"]
