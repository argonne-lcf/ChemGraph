"""Regenerate the bundled catalog in docs/skills.md, or check it in CI."""

import argparse
from pathlib import Path

from chemgraph.skills.catalog import list_skills, render_catalog

START = "<!-- BEGIN BUNDLED SKILL CATALOG -->"
END = "<!-- END BUNDLED SKILL CATALOG -->"


def update_catalog(path, *, check=False):
    text = path.read_text(encoding="utf-8")
    before, start, rest = text.partition(START)
    _, end, after = rest.partition(END)
    if not start or not end:
        raise ValueError("Missing bundled catalog markers.")
    catalog = render_catalog(list_skills(discover=False)["skills"])
    updated = before + START + "\n\n" + catalog + "\n" + END + after
    if check:
        return text == updated
    path.write_text(updated, encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).resolve().parents[1] / "docs/skills.md"
    if not update_catalog(path, check=args.check):
        print("Bundled catalog is stale; run python scripts/update_skill_catalog.py")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
