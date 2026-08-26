"""Fit a confidence table for your own committee, on your own labelled images.

The table that ships with ChemGraph describes four specific models on RDKit-rendered
diagrams. It is the right default, but it is not a law of nature: a different set of
models, or a different kind of image (scans, photographs, journal crops), has a
different relationship between agreement and correctness. This module measures that
relationship for whatever you actually run.

    python -m chemgraph.tools.ocsr_calibrate \\
        --labels my_data/labels.csv \\
        --models decimer,molnextr,molscribe \\
        --out my_calibration.json

Then use it, voting the committee the table describes:

    image_to_smiles_core(img, ensemble=True,
                         models_wanted=["decimer", "molnextr", "molscribe"],
                         calibration="my_calibration.json")
    # or, for a whole session:
    export CHEMGRAPH_OCSR_CALIBRATION=my_calibration.json

``models_wanted`` matters whenever the table covers fewer models than are installed:
the ensemble runs everything it finds otherwise, and a table fit on three models
says nothing about what four of them agreeing means.

The method is the one in the shipped table and is deliberately unclever: run every
model on every image, group predictions that are the same molecule, note the vote
pattern, and count how often the majority was right. No independence assumption, so
correlated errors between models are already priced in by construction.

You need labelled images, which is the catch: the reference SMILES has to come from
somewhere. If you have none, `--validate` against the current table is the cheaper
question, since detecting "this table is wrong for my images" takes far fewer labels
than fitting a new one. It checks whatever `load_calibration` would use:
`CHEMGRAPH_OCSR_CALIBRATION` when that is set, and the packaged table otherwise.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date

from chemgraph.tools import ocsr_backends as backends
from chemgraph.tools import ocsr_core as core

# Below this many observations a bucket gets a label and an interval but no point
# estimate. At n=20 the 95% interval is 41 points wide at the midpoint and 26 even
# for a lopsided 18/20; quoting a decimal for a bucket of 7 would be false precision,
# and the shipped table follows the same rule.
DEFAULT_MIN_N = 20

# How correctness is judged, recorded in every table this fits. Stereo-blind because
# most reference labels carry no stereochemistry, so scoring with it would mark a
# model wrong for correctly reading a wedge bond. A table fit under another rule is
# not comparable, which is why the value travels with the numbers.
SCORING = "stereo_blind"


def _jeffreys(k: int, n: int) -> float:
    """Posterior mean under Beta(0.5, 0.5). Keeps a perfect bucket below 1.0."""
    return (k + 0.5) / (n + 1.0)


def _interval(k: int, n: int, alpha: float = 0.05) -> tuple[list[float], str]:
    """A 95% interval and the method that produced it.

    Jeffreys where scipy is importable, Wilson otherwise. The two disagree by up to
    6.2 pp over the buckets below the sample floor, where the interval is the only
    thing quoted, and scipy is not a declared ChemGraph dependency: it arrives
    through ase. The method therefore travels with the table, so two tables fitted
    on the same labels are comparable only when it matches.
    """
    if n == 0:
        return [0.0, 1.0], "none: no observations"
    try:
        from scipy.stats import beta

        lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k + 0.5, n - k + 0.5))
        hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5))
        return [round(lo, 4), round(hi, 4)], "Jeffreys equal-tailed (scipy)"
    except ImportError:
        import math

        z = 1.959963984540054
        p, d = k / n, 1 + z * z / n
        centre = (p + z * z / (2 * n)) / d
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return ([round(max(0.0, centre - half), 4),
                 round(min(1.0, centre + half), 4)], "Wilson score (no scipy)")


def read_labels(path: str) -> list[tuple[str, str]]:
    """Read ``image_path,smiles`` pairs from a CSV.

    Accepts a header or not, and resolves relative image paths against the CSV's own
    directory so a labels file can travel with its images.
    """
    base = os.path.dirname(os.path.abspath(path))
    rows: list[tuple[str, str]] = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 2:
                continue
            img, smiles = row[0].strip(), row[1].strip()
            if not img or not smiles:
                continue
            if img.lower() in ("image_path", "image", "path", "file", "filename"):
                continue  # header
            if not os.path.isabs(img):
                img = os.path.join(base, img)
            rows.append((img, smiles))
    return rows


def collect(labels: list[tuple[str, str]], models: list[str],
            verbose: bool = True) -> list[tuple[list[dict], str]]:
    """Run every model over every image. Returns (results, reference) per image.

    The slow part: one model load per model, then one inference per image per model.
    Failures are kept as ``ok=False`` entries rather than dropped, because a model
    that fails on hard images is part of what the agreement pattern is measuring.
    """
    out = []
    for i, (image_path, reference) in enumerate(labels, 1):
        try:
            # Resolve and sniff once per image rather than once per model, and reject
            # a missing or non-image file before paying for any inference.
            resolved = core.resolve_image_path(image_path)
            core.load_image_bytes(resolved)
        except Exception as e:
            if verbose:
                print(f"  [{i}/{len(labels)}] skipped {image_path}: {e}", file=sys.stderr)
            continue

        results = []
        for name in models:
            r = backends.smiles_from_specialist(name, resolved)
            results.append({"model": name, "smiles": r["smiles"], "ok": r["ok"],
                            "error": r["error"], "infer_s": r["latency_s"]})
        out.append((results, reference))
        if verbose and (i % 10 == 0 or i == len(labels)):
            print(f"  {i}/{len(labels)} images", file=sys.stderr)
    return out


def fit_calibration(rows: list[tuple[list[dict], str]], models: list[str],
                    min_n: int = DEFAULT_MIN_N, dataset: str = "",
                    tie_break: list[str] | None = None) -> dict:
    """Turn (per-image results, reference) pairs into a calibration table.

    Correctness is judged the way :func:`ocsr_core.canonicalize` compares, which is
    stereo-blind by default: most reference labels carry no stereochemistry, so
    scoring with it would mark a model wrong for correctly reading a wedge bond.

    ``tie_break`` is the model priority that decides an evenly split committee, and
    it is recorded in the table because the numbers depend on it: the all-different
    bucket measures how often the *first* model was right. Defaults to ranking the
    committee by the solo accuracy measured here, so a caller who never thinks about
    order still gets the strongest model breaking ties. The order of ``models`` is
    deliberately not used for this: it is a set of names, and treating it as a
    ranking would bake a caller's arbitrary typing order into their table.
    """
    buckets: dict[str, list[int]] = {}
    # Per-model tallies alongside the committee ones: [correct, scored, abstained].
    # These become the table's model_performance section, which is what a single-model
    # backend reports as its confidence. Fitting them here rather than leaving them to
    # a separate run is what makes a refit update every number the tool reports.
    solo: dict[str, list[int]] = {m: [0, 0, 0] for m in models}

    # Two passes: score every model first, so an unspecified tie-break can rank the
    # committee by what the data says rather than by argument order.
    if tie_break is None:
        scored: dict[str, list[int]] = {m: [0, 0] for m in models}
        for results, reference in rows:
            ref = core.canonicalize(reference)
            if ref is None:
                continue
            for r in results:
                name = str(r.get("model", "")).removeprefix("local:")
                if name not in scored:
                    continue
                scored[name][1] += 1
                predicted = (core.canonicalize(r.get("smiles"))
                             if r.get("ok", True) else None)
                if predicted == ref:
                    scored[name][0] += 1
        order = sorted(models, key=lambda m: (-(scored[m][0] / scored[m][1])
                                              if scored[m][1] else 0.0, m))
    else:
        order = list(tie_break)
        if sorted(order) != sorted(models):
            raise ValueError(
                f"tie_break must name exactly the committee, most accurate first. "
                f"Committee: {sorted(models)}. Got: {sorted(order)}."
            )

    for results, reference in rows:
        ref = core.canonicalize(reference)
        if ref is None:
            continue  # an unparseable reference cannot judge anything
        for r in results:
            name = str(r.get("model", "")).removeprefix("local:")
            tally = solo.get(name)
            if tally is None:
                continue
            tally[1] += 1
            predicted = core.canonicalize(r.get("smiles")) if r.get("ok", True) else None
            if predicted is None:
                tally[2] += 1
            elif predicted == ref:
                tally[0] += 1
        v = core.vote(results, priority=order)
        if v["pattern"] is None:
            continue  # nobody voted; no pattern to attribute this to
        entry = buckets.setdefault(v["pattern"], [0, 0])
        entry[1] += 1
        if v["winner"] == ref:
            entry[0] += 1

    patterns = {}
    ci_method = ""
    for pattern, (k, n) in sorted(buckets.items(), key=lambda kv: -kv[1][1]):
        ci, ci_method = _interval(k, n)
        cell: dict = {"k": k, "n": n, "ci": ci}
        if n >= min_n:
            cell["p"] = round(_jeffreys(k, n), 4)
            cell["label"] = core._label_for(cell["p"])
        else:
            # No stored label below the floor: the consumer derives one from the
            # Jeffreys estimate, which is the same rule confidence() applies.
            cell["p"] = None
        patterns[pattern] = cell

    performance = {
        name: {
            "accuracy": round(k / n, 4),
            "k": k,
            "n": n,
            "ci": _interval(k, n)[0],
            "abstention_rate": round(abstained / n, 4),
        }
        for name, (k, n, abstained) in solo.items()
        if n
    }

    return {
        "committee": list(models),
        "n_items": sum(n for _, n in buckets.values()),
        "dataset": dataset,
        "scoring": SCORING,
        # The rule vote() applies. Recorded because a table fit under any other one
        # has buckets that do not line up with what the tool produces, and
        # _validate_calibration rejects such a table on the pattern sums alone.
        "abstention": ("counts as a dissenting singleton vote; pattern parts always "
                       "sum to the committee size"),
        "model_performance": performance,
        "model_performance_rule": (
            "Per-model accuracy over the same images, used when a backend runs one "
            "model and there is no agreement pattern to look up. 'accuracy' is the raw "
            "rate; 'ci' is the 95% Jeffreys interval; 'abstention_rate' is how often "
            "the model returned no parseable SMILES."
        ),
        "tie_break": "model-priority: " + ",".join(order),
        "created": date.today().isoformat(),
        "min_n_for_point_estimate": min_n,
        "estimator": "Jeffreys: (k+0.5)/(n+1)",
        # Which library produced the intervals. Two tables fitted on the same labels
        # are only comparable when this matches: the Wilson fallback differs from
        # Jeffreys by up to 6.2 pp on the buckets below the sample floor.
        "interval_method": ci_method,
        "label_rule": (
            "Where p is present, the consumer may quote it. Where p is null (n below min_n_for_point_estimate), no number is quoted: the label comes from the Jeffreys estimate (k+0.5)/(n+1), the same quantity p reports above the floor, so one rule covers the whole table."
        ),
        "patterns": patterns,
    }


def report(table: dict, min_n: int) -> str:
    """A human-readable summary, including what is too thin to quote."""
    lines = [
        f"committee : {', '.join(table['committee'])}",
        f"images    : {table['n_items']}",
        f"scoring   : {table['scoring']}",
        "",
        f"{'pattern':10s} {'k/n':>9s} {'P(correct)':>11s} {'95% CI':>16s}  label",
    ]
    thin = []
    for pattern, cell in table["patterns"].items():
        k, n = cell["k"], cell["n"]
        # Every displayed field comes from confidence(), the same call image_to_smiles
        # makes, so this report cannot drift from what the tool will say. Reading
        # cell["label"] and cell["ci"] directly crashed with a bare KeyError on the
        # shipped table, whose above-floor cells store no label, and on any table the
        # validator accepts without a ci.
        verdict = core.confidence(pattern, table)
        if verdict["p"] is None:
            thin.append((pattern, n))
        p_str = "  --" if verdict["p"] is None else f"{verdict['p']:.3f}"
        ci = verdict["ci"]
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "(no interval)"
        lines.append(f"{pattern:10s} {f'{k}/{n}':>9s} {p_str:>11s} "
                     f"{ci_str:>16s}  {verdict['label']}")
    if thin:
        lines += ["", f"Too thin to quote a number (n < {min_n}): " +
                  ", ".join(f"{p} (n={n})" for p, n in thin)]
        lines.append("These still get a label from the Jeffreys estimate.")
    return "\n".join(lines)


def validate(rows: list[tuple[list[dict], str]], models: list[str],
             table: dict) -> str:
    """Compare an existing table's predictions against measured outcomes.

    Far cheaper than refitting: 20-40 labelled images is enough to notice that a
    table is badly wrong for your images, where fitting a new one needs a few
    hundred. This is the question most users actually have.
    """
    # Vote by the rule the table was fitted under, so the comparison is like for
    # like. Using any other order would measure a different quantity than the one
    # the table's numbers describe. A table that reached here without going through
    # load_calibration can carry no committee at all, so models is the last resort.
    order = core.tie_break_order(table) or list(models)
    seen: dict[str, list[int]] = {}
    for results, reference in rows:
        ref = core.canonicalize(reference)
        if ref is None:
            continue
        v = core.vote(results, priority=order)
        if v["pattern"] is None:
            continue
        e = seen.setdefault(v["pattern"], [0, 0])
        e[1] += 1
        if v["winner"] == ref:
            e[0] += 1

    lines = [f"{'pattern':10s} {'yours':>12s} {'table says':>11s} {'gap':>8s}"]
    worst = 0.0
    # Tracked separately from `worst`: a bucket that matches the table exactly leaves
    # worst at 0.0, and overloading that to mean "nothing was judged" reported the
    # best possible outcome as a failure to gather data.
    judged = False
    for pattern, (k, n) in sorted(seen.items(), key=lambda kv: -kv[1][1]):
        c = core.confidence(pattern, table)
        observed = k / n
        if c["p"] is None:
            lines.append(f"{pattern:10s} {f'{k}/{n}':>12s} {'(no number)':>11s} "
                         f"{'':>8s}")
            continue
        gap = observed - c["p"]
        if n >= 10:
            judged = True
            worst = max(worst, abs(gap))
        lines.append(f"{pattern:10s} {f'{k}/{n} = {100*observed:.0f}%':>12s} "
                     f"{100*c['p']:10.0f}% {100*gap:+7.0f} pp")

    lines.append("")
    if not judged:
        lines.append("Not enough data in any bucket (n >= 10) to judge. Add more labels.")
    elif worst < 0.10:
        lines.append(f"Largest gap {100*worst:.0f} pp. This table looks usable "
                     f"for your images.")
    else:
        lines.append(f"Largest gap {100*worst:.0f} pp. This table does NOT "
                     f"describe your images well; consider fitting your own.")
    lines.append("Gaps on small buckets are mostly noise: 10 images cannot "
                 "distinguish 90% from 70%.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m chemgraph.tools.ocsr_calibrate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--labels", required=True,
                   help="CSV of image_path,smiles (relative paths resolve to its directory)")
    p.add_argument("--models", default=None,
                   help=("comma-separated committee; default is every installed "
                         "model. Order does not matter here."))
    p.add_argument("--tie-break", default=None, metavar="MODELS",
                   help=("comma-separated models, most accurate first, deciding who "
                         "wins when the committee splits evenly. Defaults to the "
                         "most accurate first, measured on your own images."))
    p.add_argument("--out", default=None, help="where to write the table")
    p.add_argument("--validate", action="store_true",
                   help="check the current table (CHEMGRAPH_OCSR_CALIBRATION, or "
                        "the packaged one) against your images instead of fitting")
    p.add_argument("--min-n", type=int, default=DEFAULT_MIN_N,
                   help=f"observations needed for a point estimate (default {DEFAULT_MIN_N})")
    args = p.parse_args(argv)

    if args.min_n < 0:
        # Checked here for the same reason as --models below: the pre-write
        # validator rejects a negative floor, and reaching it costs a full run.
        print(f"--min-n must be zero or more: {args.min_n}", file=sys.stderr)
        return 2

    installed = backends.available_specialists()
    models = ([m.strip() for m in args.models.split(",")] if args.models else installed)
    missing = [m for m in models if m not in installed]
    if missing:
        print(f"not installed: {', '.join(missing)}. Installed: "
              f"{', '.join(installed) or 'none'}.\n"
              f"Install with: pip install 'chemgraph[ocsr]'",
              file=sys.stderr)
        return 2
    if not models:
        print("no specialist models installed; nothing to calibrate", file=sys.stderr)
        return 2
    if len(set(models)) != len(models):
        # Checked here rather than at the pre-write validator, which would reject it
        # only after running every model over every image.
        print(f"--models repeats a model: {','.join(models)}", file=sys.stderr)
        return 2

    if args.validate and args.tie_break is not None:
        # Checked with the other argument validation, not after collect() has spent
        # minutes of inference that the return would discard.
        print("--tie-break has no effect with --validate, which votes by the order "
              "recorded in the table it is checking", file=sys.stderr)
        return 2

    tie_break = None
    if args.tie_break is not None:
        tie_break = [m.strip() for m in args.tie_break.split(",") if m.strip()]
        if sorted(tie_break) != sorted(models):
            print(f"--tie-break must list exactly the committee, in your preferred "
                  f"order. Committee: {','.join(models)}. Got: {','.join(tie_break)}.",
                  file=sys.stderr)
            return 2

    if not os.path.isfile(os.path.expanduser(args.labels)):
        print(f"no such labels file: {args.labels}. Expected a CSV of "
              f"image_path,smiles.", file=sys.stderr)
        return 2
    labels = read_labels(args.labels)
    if not labels:
        print(f"no usable rows in {args.labels}; expected image_path,smiles",
              file=sys.stderr)
        return 2

    if not args.validate and len(labels) < 100:
        print(f"note: {len(labels)} images is thin for fitting a table. Most buckets "
              f"will be under the n={args.min_n} floor and get a label but no number. "
              f"About 300 is a sensible minimum; --validate is the better question "
              f"below that.\n", file=sys.stderr)

    if args.validate:
        # Load before any inference. Reading it after collect() would discard every
        # model run over every image to report a typo in a path, and the same
        # reasoning already governs the write below.
        try:
            reference_table = core.load_calibration()
        except (OSError, ValueError, TypeError) as exc:
            print(f"cannot validate: {exc}", file=sys.stderr)
            return 2

    print(f"running {len(models)} model(s) over {len(labels)} images; the first call "
          f"per model loads it (9-170 s)", file=sys.stderr)
    rows = collect(labels, models)
    if not rows:
        print("no images could be read", file=sys.stderr)
        return 1

    if args.validate:
        print(validate(rows, models, reference_table))
        return 0

    table = fit_calibration(rows, models, min_n=args.min_n,
                            dataset=os.path.abspath(args.labels),
                            tie_break=tie_break)

    # Check the table against the loader before writing it. The committee comes from
    # --models while the patterns come from whatever the rows actually held, so the
    # two can disagree: a model that never produced a row, a row naming a model not in
    # --models, or ragged rows all yield patterns that do not sum to the committee
    # size. Writing first and failing at load leaves the user holding a file this tool
    # told them to use.
    try:
        core._validate_calibration(table, args.out or "the fitted table")
    except ValueError as exc:
        print(f"\nthe fitted table is not self-consistent, so it was not written:\n"
              f"  {exc}\n"
              f"  Every model in --models must appear in every row, and no row may "
              f"name a model outside it.", file=sys.stderr)
        return 1

    print(report(table, args.min_n))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(table, fh, indent=2)
        installed = backends.available_specialists()
        subset = [m for m in installed if m not in models]
        names = ", ".join(repr(m) for m in models)
        wanted = f"models_wanted=[{names}], " if subset else ""
        print(f"\nwrote {args.out}\nUse it with: "
              f"image_to_smiles_core(img, ensemble=True, {wanted}"
              f"calibration='{args.out}')")
        if subset:
            print(f"  models_wanted is needed here: {', '.join(subset)} "
                  f"{'is' if len(subset) == 1 else 'are'} also installed, and the "
                  f"ensemble would otherwise vote {'it' if len(subset) == 1 else 'them'} too.")
    else:
        print("\n(no --out given, table not saved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
