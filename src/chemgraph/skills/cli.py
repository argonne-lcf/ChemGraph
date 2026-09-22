"""Offline skill contribution commands."""

import json

from rich.console import Console

from chemgraph.skills.lint import lint_skills


def add_skill_args(parser):
    commands = parser.add_subparsers(dest="skill_command", required=True)
    lint = commands.add_parser("lint", help="Validate a skill or collection offline.")
    lint.add_argument("directory")
    lint.add_argument("--core", action="store_true", help="Require Core attribution.")
    lint.add_argument("--json", action="store_true", help="Print JSON diagnostics.")


def run_skills(args):
    errors = lint_skills(args.directory, core=args.core)
    if args.json:
        print(json.dumps({"diagnostics": [error.as_dict() for error in errors]}))
    else:
        console = Console()
        for error in errors:
            console.print(f"{error.path}: {error.code}: {error.message}", markup=False, highlight=False)
        if not errors:
            console.print("Skill checks passed.")
    return int(bool(errors))
