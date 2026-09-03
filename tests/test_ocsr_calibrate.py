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


def test_validate_refuses_a_table_that_describes_another_committee():
    """The CLI must not endorse what the tool will refuse.

    Silent when broken: every bucket reads "no number", and the verdict blames the
    sample size when the committee is the wrong set.
    """
    rows = _rows(*[({"decimer": "CCO", "molnextr": "CCO", "molscribe": "CCO",
                     "imagemol": "CCO"}, "CCO")] * 30)

    out = calibrate.validate(rows, ["decimer", "molnextr", "molscribe", "imagemol"],
                             core.load_calibration())

    assert "cannot validate" in out
    assert "committee_mismatch" in out


def test_validate_refuses_a_wrong_committee_before_any_inference(monkeypatch,
                                                                 tmp_path, capsys):
    """Both inputs are known before collect(), so the refusal must be too.

    Silent when broken: the run reads every image with every model and then
    reports a mismatch it could have found in the first millisecond.
    """
    from chemgraph.tools import ocsr_backends as backends

    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,smiles\n"
                      + "".join(f"/x{i}.png,CCO\n" for i in range(30)))
    monkeypatch.setattr(backends, "available_specialists",
                        lambda: ["decimer", "molnextr"])

    def never(name, path):
        raise AssertionError("inference ran before the committee was checked")

    monkeypatch.setattr(backends, "smiles_from_specialist", never)

    code = calibrate.main(["--labels", str(labels), "--validate"])

    assert code == 2
    assert "cannot validate" in capsys.readouterr().err


def test_validate_checks_the_models_the_rows_actually_carry():
    """A direct caller assembling rows is checked on their contents.

    Silent when broken: rows holding three models against a two-model table pass,
    and every bucket then reads "no number" while the verdict blames the sample.
    """
    rows = _rows(*[({"a": "CCO", "b": "CCO", "c": "CCO"}, "CCO")] * 30)

    out = calibrate.validate(rows, ["a", "b"],
                             {"committee": ["a", "b"],
                              "patterns": {"2": {"k": 19, "n": 20, "p": 0.9286}}})

    assert "cannot validate" in out


def test_report_survives_a_table_the_validator_accepts():
    """Only committee and patterns are required, so the header cannot demand more.

    Silent when broken: a bare KeyError from the summary of a table the tool will
    happily use, which is the failure the body below it was already rewritten for.
    """
    thin = {"committee": ["a"], "patterns": {"1": {"k": 1, "n": 1}}}
    core._validate_calibration(thin, "test")

    text = calibrate.report(thin, 20)

    assert "unrecorded" in text
    assert "committee : a" in text


def test_an_unwritable_out_path_is_refused_before_any_inference(monkeypatch,
                                                                tmp_path, capsys):
    """The fitted table is the run's whole output; losing it to a typo is the worst
    case the argument checks exist to prevent.

    Silent when broken: the run pays for every inference and then tracebacks.
    """
    from chemgraph.tools import ocsr_backends as backends

    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,smiles\n/x.png,CCO\n")
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])

    def never(name, path):
        raise AssertionError("inference ran before --out was checked")

    monkeypatch.setattr(backends, "smiles_from_specialist", never)

    code = calibrate.main(["--labels", str(labels), "--out", "/proc/nope/t.json"])

    assert code == 2
    assert "cannot write" in capsys.readouterr().err


@pytest.mark.parametrize("content, why", [
    (b"\xff\xfe\x00bad", "not valid UTF-8"),
    (b"image_path,smiles\n" + b"a" * 200000 + b",CCO\n", "a field past csv's limit"),
])
def test_a_labels_file_csv_cannot_read_is_reported_not_raised(tmp_path, content, why):
    labels = tmp_path / "labels.csv"
    labels.write_bytes(content)

    try:
        rows = calibrate.read_labels(str(labels))
        assert rows == [] or all(len(r) == 2 for r in rows)
    except ValueError as exc:
        assert "not readable as CSV" in str(exc)


def test_a_labels_file_csv_cannot_read_exits_like_its_neighbours(monkeypatch,
                                                                 tmp_path, capsys):
    """read_labels raising is only half the fix; main() has to catch it.

    Silent when broken: a traceback and exit 1, where every other bad-argument
    path exits 2 with a message.
    """
    from chemgraph.tools import ocsr_backends as backends

    labels = tmp_path / "labels.csv"
    labels.write_text("x" * 200000 + ".png,CCO\n")
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])

    code = calibrate.main(["--labels", str(labels)])

    assert code == 2
    assert "cannot read" in capsys.readouterr().err


def test_report_survives_a_cell_missing_its_counts():
    """The validator's k/n check can be skipped by a cell that omits them."""
    text = calibrate.report({"committee": ["a"], "patterns": {"1": {}}}, 20)

    assert "0/0" in text


def test_the_out_probe_leaves_an_existing_file_alone(monkeypatch, tmp_path):
    """It cannot tell a zero-byte file the user had from one it just made.

    Silent when broken: the probe runs before the other argument checks, so a run
    that then exits 2 having done nothing has deleted the user's file.
    """
    from chemgraph.tools import ocsr_backends as backends

    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,smiles\n/x.png,CCO\n")
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])
    theirs = tmp_path / "theirs.json"
    theirs.touch()

    calibrate.main(["--labels", str(labels), "--out", str(theirs), "--min-n", "-1"])

    assert theirs.exists()


def test_both_path_arguments_expand_a_tilde(monkeypatch, tmp_path):
    """A shell expands an unquoted one; a quoted one and argv arrive intact.

    Silent when broken: --labels cleared its existence check expanded and was then
    opened unexpanded, and --out wrote the table to a directory literally named ~
    beside wherever the process happened to be.
    """
    from chemgraph.tools import ocsr_backends as backends

    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    drawer = rdMolDraw2D.MolDraw2DCairo(150, 150)
    drawer.DrawMolecule(Chem.MolFromSmiles("CCO"))
    drawer.FinishDrawing()
    image = tmp_path / "a.png"
    image.write_bytes(drawer.GetDrawingText())
    home = tmp_path / "home"
    home.mkdir()
    (home / "labels.csv").write_text(f"image_path,smiles\n{image},CCO\n")
    monkeypatch.setattr(backends, "available_specialists", lambda: ["a", "b"])
    monkeypatch.setattr(backends, "smiles_from_specialist", lambda n, p: {
        "ok": True, "smiles": "CCO", "raw": "", "model_used": n,
        "cold_start": False, "latency_s": 0.1, "error": "", "ran": True})
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)

    code = calibrate.main(["--labels", "~/labels.csv", "--out", "~/t.json",
                           "--min-n", "1"])

    assert code == 0
    assert (home / "t.json").is_file()
    assert not (tmp_path / "~").exists()


def test_the_probe_and_the_write_use_one_path(monkeypatch, tmp_path):
    """realpath strips a trailing slash and open does not.

    Silent when broken: the probe clears one path, the write uses another, and a
    run that passed the pre-flight check fails after spending its inference.
    """
    from chemgraph.tools import ocsr_backends as backends

    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    drawer = rdMolDraw2D.MolDraw2DCairo(150, 150)
    drawer.DrawMolecule(Chem.MolFromSmiles("CCO"))
    drawer.FinishDrawing()
    image = tmp_path / "a.png"
    image.write_bytes(drawer.GetDrawingText())
    labels = tmp_path / "labels.csv"
    labels.write_text(f"image_path,smiles\n{image},CCO\n")
    monkeypatch.setattr(backends, "available_specialists", lambda: ["a", "b"])
    monkeypatch.setattr(backends, "smiles_from_specialist", lambda n, p: {
        "ok": True, "smiles": "CCO", "raw": "", "model_used": n,
        "cold_start": False, "latency_s": 0.1, "error": "", "ran": True})

    code = calibrate.main(["--labels", str(labels),
                           "--out", str(tmp_path / "table.json") + "/",
                           "--min-n", "1"])

    assert code == 0
    assert (tmp_path / "table.json").is_file()


def test_the_out_probe_refuses_a_read_only_file_through_a_link(monkeypatch,
                                                              tmp_path, capsys):
    """The same file has to get the same answer whichever name reaches it.

    Silent when broken: the link is waved through, the run spends its inference,
    and the write fails afterwards.
    """
    from chemgraph.tools import ocsr_backends as backends

    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,smiles\n/x.png,CCO\n")
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])
    target = tmp_path / "target.json"
    target.write_text("{}")
    target.chmod(0o400)
    link = tmp_path / "link.json"
    link.symlink_to(target)

    code = calibrate.main(["--labels", str(labels), "--out", str(link),
                           "--min-n", "20"])

    assert code == 2
    assert "cannot write" in capsys.readouterr().err
    target.chmod(0o600)


def test_the_out_probe_follows_the_link_to_its_target(monkeypatch, tmp_path,
                                                     capsys):
    """A symlink resolves in its target's directory, not the one holding the link.

    Silent when broken: the probe exists to refuse an unwritable --out before a
    full run of inference, and it waves this one through to fail after it.
    """
    from chemgraph.tools import ocsr_backends as backends

    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,smiles\n/x.png,CCO\n")
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])
    readonly = tmp_path / "readonly"
    readonly.mkdir(mode=0o500)
    link = tmp_path / "link.json"
    link.symlink_to(readonly / "out.json")

    code = calibrate.main(["--labels", str(labels), "--out", str(link),
                           "--min-n", "20"])

    assert code == 2
    assert "cannot write" in capsys.readouterr().err
    readonly.chmod(0o700)


def test_the_out_probe_accepts_a_dangling_symlink(monkeypatch, tmp_path,
                                                  capsys):
    """exists() follows the link, open(x) does not, so they disagree on this one.

    Silent when broken: the run is refused before any inference with "File exists"
    naming a path that does not exist.
    """
    from chemgraph.tools import ocsr_backends as backends

    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,smiles\n/x.png,CCO\n")
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])
    link = tmp_path / "link.json"
    link.symlink_to(tmp_path / "never_created.json")

    code = calibrate.main(["--labels", str(labels), "--out", str(link),
                           "--min-n", "-1"])

    # Reaching the --min-n check means the probe let it through. Asserting only
    # that the link survives passes either way, since the probe never unlinks.
    assert code == 2
    err = capsys.readouterr().err
    assert "--min-n" in err
    assert "File exists" not in err
    assert link.is_symlink()
    # And the probe created nothing through it: appending to a symlink follows it,
    # so a run that goes on to exit 2 would leave the target behind.
    assert not (tmp_path / "never_created.json").exists()


def test_the_out_probe_creates_nothing_of_its_own(monkeypatch, tmp_path):
    from chemgraph.tools import ocsr_backends as backends

    labels = tmp_path / "labels.csv"
    labels.write_text("image_path,smiles\n/x.png,CCO\n")
    monkeypatch.setattr(backends, "available_specialists", lambda: ["decimer"])
    fresh = tmp_path / "fresh.json"

    calibrate.main(["--labels", str(labels), "--out", str(fresh), "--min-n", "-1"])

    assert not fresh.exists()


def test_the_fitter_scores_stereo_blind_as_it_records():
    """SCORING travels with every table, so the fitter has to match the string.

    Silent when broken: nothing in the suite reads a stereo-bearing reference, and
    scoring stereo-aware moves every headline number: on the shipped benchmark
    decimer falls from 649 correct to 609 and molscribe from 595 to 458.
    """
    # A reference with a centre assigned, read flat by both models.
    rows = _rows(*([({"a": "CC(N)C(=O)O", "b": "CC(N)C(=O)O"},
                     "C[C@H](N)C(=O)O")] * 25))

    t = calibrate.fit_calibration(rows, ["a", "b"], min_n=20)

    assert t["scoring"] == calibrate.SCORING == "stereo_blind"
    assert t["patterns"]["2"]["k"] == 25
    assert t["model_performance"]["a"]["k"] == 25


def test_the_fitter_writes_a_quotable_solo_estimate_beside_the_raw_rate():
    """Two fields because they answer two questions, and the tool quotes one.

    Silent when broken: prior_confidence falls back to the raw rate, so a model
    perfect on its images reports 1.0 unanimous while the bucket fitted from those
    same images reports the Jeffreys estimate one band lower.
    """
    rows = _rows(*([({"a": "CCO", "b": "CCO"}, "CCO")] * 90
                   + [({"a": "CCN", "b": "CCN"}, "CCO")] * 10))

    t = calibrate.fit_calibration(rows, ["a", "b"], min_n=20)
    entry = t["model_performance"]["a"]

    assert (entry["k"], entry["n"]) == (90, 100)
    assert entry["accuracy"] == 0.9
    assert entry["p"] == round(90.5 / 101, 4) == 0.896
    assert core.prior_confidence("a", t)["p"] == entry["p"]


def test_a_thin_solo_estimate_is_withheld_the_way_a_thin_bucket_is():
    """The floor applies to both, since both are fitted from the same images.

    Silent when broken: the bucket withholds its number and the solo accuracy
    beside it quotes one, from the same six images.
    """
    rows = _rows(*([({"a": "CCO", "b": "CCO"}, "CCO")] * 6))

    t = calibrate.fit_calibration(rows, ["a", "b"], min_n=20)

    assert t["patterns"]["2"]["p"] is None
    assert t["model_performance"]["a"]["p"] is None
    assert t["model_performance"]["a"]["accuracy"] == 1.0
    assert core.prior_confidence("a", t)["reason"] == "below_n_floor"
