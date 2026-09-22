"""Hermetic contribution checks for portable and bundled skills."""

import importlib
import json
from pathlib import Path
import shlex
import sys

from markdown_it import MarkdownIt
import pytest
import yaml

from chemgraph.registry import ToolRegistry
from chemgraph.skills.lint import lint_skills

ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "src/chemgraph/skills"


def skill(tmp_path, **frontmatter):
    root = tmp_path / "example"
    root.mkdir(exist_ok=True)
    data = {"name": "example", "description": "A useful workflow", **frontmatter}
    (root / "SKILL.md").write_text(
        "---\n" + yaml.safe_dump(data) + "---\nDo not disable approvals.\n",
        encoding="utf-8",
    )
    return root


@pytest.mark.parametrize("fields", [
    {"name": "different"}, {"name": "Example"}, {"name": "bad--name"},
    {"name": "x" * 65}, {"name": 1}, {"description": None},
    {"description": "x" * 1025}, {"compatibility": "x" * 501},
    {"license": False}, {"allowed-tools": ["execute"]},
    {"metadata": []}, {"metadata": {"version": 1}}, {"metadata": {1: "value"}},
], ids=["mismatch", "uppercase", "hyphens", "long-name", "numeric-name",
        "null-description", "long-description", "long-compatibility", "license-type",
        "tool-list", "metadata-list", "numeric-value", "numeric-key"])
def test_raw_frontmatter_rejected_before_runtime_coercion(tmp_path, fields):
    assert lint_skills(skill(tmp_path, **fields))


@pytest.mark.parametrize("content", [
    "No frontmatter", "---\nname: [\n---\nBody", "---\n[]\n---\nBody",
    "---\nname: example\nname: example\ndescription: Duplicate\n---\nBody",
    "---\nname: example\n---\nBody",
])
def test_malformed_frontmatter(tmp_path, content):
    root = skill(tmp_path)
    (root / "SKILL.md").write_text(content, encoding="utf-8")
    assert lint_skills(root)[0].code == "frontmatter"


def test_core_attribution_is_optional_for_personal_skills(tmp_path):
    root = skill(tmp_path)
    assert lint_skills(root) == []
    assert len(lint_skills(root, core=True)) == 3
    root = skill(tmp_path, license="Apache-2.0", metadata={
        "authors": "Example Author", "maintainers": "example",
        "version": "1.0", "custom-key": "preserved",
    })
    assert lint_skills(root, core=True) == []


def test_relative_links_in_references_and_across_skills(tmp_path):
    root = skill(tmp_path)
    (tmp_path / "shared.md").write_text("Shared", encoding="utf-8")
    (root / "references").mkdir()
    reference = root / "references/with space.md"
    reference.write_text(
        "[shared](../../shared.md#heading) [web](https://example.invalid/no-fetch)\n"
        "[anchor](#heading) `![code](missing.png)`\n"
        "```sh\n[not a link](missing.txt)\n```\n",
        encoding="utf-8",
    )
    with (root / "SKILL.md").open("a", encoding="utf-8") as file:
        file.write("[reference][guide]\n\n[guide]: references/with%20space.md\n")
    assert lint_skills(tmp_path) == []
    reference.write_text("![image](missing.png) [broken](missing.md)", encoding="utf-8")
    errors = lint_skills(root)
    assert len(errors) == 2
    assert all(error.code == "link" and error.path == str(reference) for error in errors)


def test_missing_empty_and_unreadable_sources(tmp_path, monkeypatch):
    assert lint_skills(tmp_path)[0].code == "source"
    assert lint_skills(tmp_path / "absent")[0].code == "source"
    root = skill(tmp_path)
    (root / "SKILL.md").write_bytes(b"\xff")
    assert lint_skills(root)[0].code == "read"
    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", denied)
    assert lint_skills(root)[0].code == "read"


def test_bundled_contribution_gate():
    assert lint_skills(SKILLS, core=True) == []


def test_documented_cli_examples_parse_without_execution():
    from chemgraph.cli.main import create_argument_parser

    parser = create_argument_parser()
    text = (SKILLS / "chemgraph/references/python-and-cli.md").read_text()
    examples = [line for block in MarkdownIt().parse(text) if block.type == "fence"
                for line in block.content.splitlines() if line.startswith("chemgraph ")]
    assert examples
    for command in examples:
        args = parser.parse_args(shlex.split(command)[1:])
        assert args.command == "run"


def test_documented_preparation_tools_are_registered():
    text = (SKILLS / "chemgraph/references/structure-preparation.md").read_text()
    for name in ("smiles_to_coordinate_file", "file_to_atomsdata", "molecule_name_to_smiles"):
        assert f"`{name}`" in text
        assert ToolRegistry().get_spec(name).name == name


@pytest.mark.parametrize("valid", [True, False])
def test_lint_cli_json_and_exit_status(tmp_path, monkeypatch, capsys, valid):
    cli = importlib.import_module("chemgraph.cli.main")
    root = skill(tmp_path, description="Valid" if valid else None)
    monkeypatch.setattr(cli, "initialize_agent", lambda **kw: pytest.fail("must stay offline"))
    monkeypatch.setattr(sys, "argv", ["chemgraph", "skills", "lint", str(root), "--json"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == (0 if valid else 1)
    result = json.loads(capsys.readouterr().out)
    assert bool(result["diagnostics"]) == (not valid)
