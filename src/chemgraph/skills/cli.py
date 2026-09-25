"""Offline skill contribution commands."""

import json

from rich.console import Console
from rich.table import Table
from rich.text import Text

from chemgraph.skills.catalog import list_skills
from chemgraph.skills.lint import lint_skills


def add_skill_args(parser):
    commands = parser.add_subparsers(dest="skill_command", required=True)
    lint = commands.add_parser("lint", help="Validate a skill or collection offline.")
    lint.add_argument("directory")
    lint.add_argument("--core", action="store_true", help="Require Core attribution.")
    lint.add_argument("--json", action="store_true", help="Print JSON diagnostics.")
    listing = commands.add_parser("list", help="Inspect local skill sources offline.")
    listing.add_argument("--workspace", default=".", help="Workspace to inspect (default: current directory).")
    listing.add_argument("--skill-dir", action="append", help="Host collection; repeat to layer sources.")
    listing.add_argument("--no-discover", dest="discover", action="store_false", help="Omit personal/project discovery.")
    listing.add_argument("--all", action="store_true", help="Include shadowed definitions.")
    listing.add_argument("--json", action="store_true", help="Print full metadata and sources as JSON.")


def _text(value):
    from chemgraph.cli.formatting import _safe_review_text

    return Text(_safe_review_text(str(value)), overflow="fold")


def _list(args):
    try:
        result = list_skills(workspace=args.workspace, skill_dirs=args.skill_dir,
                             discover=args.discover, include_shadowed=args.all)
    except (OSError, ValueError, RuntimeError) as exc:
        if args.json:
            print(json.dumps({"skills": [], "warnings": [], "error": str(exc)}))
        else:
            Console(stderr=True).print(_text(exc))
        return 1
    if args.json:
        print(json.dumps(result))
    else:
        table = Table("Name", "Source", "State", "Version", "Authors", "Maintainers", "License")
        for column in table.columns:
            column.overflow = "fold"
        for skill in result["skills"]:
            metadata = skill["metadata"]
            source = f"{skill['source_kind']}: {skill['host_path'] or skill['source']}"
            values = (skill["name"], source, "active" if skill["active"] else "shadowed",
                      metadata.get("version"), metadata.get("authors"),
                      metadata.get("maintainers"), skill.get("license"))
            table.add_row(*(_text(value or "—") for value in values))
        Console().print(table)
        for warning in result["warnings"]:
            Console(stderr=True).print(_text(warning))
    return 0


def run_skills(args):
    if args.skill_command == "list":
        return _list(args)
    errors = lint_skills(args.directory, core=args.core)
    if args.json:
        print(json.dumps({"diagnostics": [error.as_dict() for error in errors]}))
    else:
        console = Console()
        for error in errors:
            console.print(_text(f"{error.path}: {error.code}: {error.message}"))
        if not errors:
            console.print("Skill checks passed.")
    return int(bool(errors))
