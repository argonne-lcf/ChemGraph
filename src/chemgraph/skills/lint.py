"""Offline contribution checks; runtime loading remains deliberately permissive."""

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

from deepagents.middleware.skills import (
    MAX_SKILL_COMPATIBILITY_LENGTH,
    MAX_SKILL_DESCRIPTION_LENGTH,
    MAX_SKILL_FILE_SIZE,
    _validate_skill_name,
)
from markdown_it import MarkdownIt
import yaml


@dataclass(frozen=True)
class Diagnostic:
    path: str
    code: str
    message: str

    def as_dict(self):
        return asdict(self)


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = super().construct_mapping(node, deep=deep)
        if len(result) != len(node.value):
            raise yaml.YAMLError("Duplicate frontmatter keys are not allowed.")
        return result


def _check_frontmatter(text, path, core):
    """Validate raw values before upstream can coerce or truncate them."""
    errors = []

    def error(message):
        errors.append(Diagnostic(str(path), "frontmatter", message))

    # The parser and naming helper match the pinned deepagents==0.7.5 loader.
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        error("Expected YAML frontmatter delimited by ---.")
        return errors
    if len(text) > MAX_SKILL_FILE_SIZE:
        error("SKILL.md exceeds the runtime size limit.")
    try:
        data = yaml.load(match[1], Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        error(f"Invalid YAML: {exc}")
        return errors
    if not isinstance(data, dict):
        error("Frontmatter must be a mapping.")
        return errors
    for key in ("name", "description", "license", "compatibility", "allowed-tools"):
        if key not in data and key not in {"name", "description"}:
            continue
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            error(f"{key} must be a non-empty string.")
    name = data.get("name")
    if isinstance(name, str):
        valid, message = _validate_skill_name(name, path.parent.name)
        if not valid:
            error(message)
    for key, limit in (
        ("description", MAX_SKILL_DESCRIPTION_LENGTH),
        ("compatibility", MAX_SKILL_COMPATIBILITY_LENGTH),
    ):
        if isinstance(data.get(key), str) and len(data[key]) > limit:
            error(f"{key} exceeds {limit} characters.")
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        error("metadata must map strings to strings.")
        metadata = {}
    if core:
        for key, value in (
            ("license", data.get("license")),
            ("metadata.authors", metadata.get("authors")),
            ("metadata.maintainers", metadata.get("maintainers")),
        ):
            if not isinstance(value, str) or not value.strip():
                error(f"Core skills require {key}.")
    return errors


def _check_links(text, path):
    """Check file targets, not web URLs or Markdown heading anchors."""
    errors = []
    for block in MarkdownIt("commonmark").parse(text):
        for token in block.children or ():
            target = token.attrGet("href") or token.attrGet("src")
            if not target:
                continue
            try:
                url = urlsplit(target)
                if url.scheme or url.netloc or not url.path:
                    continue
                file = Path(unquote(url.path))
                if file.is_absolute():
                    continue
                if not (path.parent / file).exists():
                    errors.append(Diagnostic(str(path), "link", f"Missing target: {target}"))
            except (OSError, ValueError) as exc:
                errors.append(Diagnostic(str(path), "link", f"Invalid target {target!r}: {exc}"))
    return errors


def lint_skills(directory, *, core=False):
    """Check one skill or the immediate skills in a host collection.

    Each diagnostic is an error. Skills without ChemGraph metadata remain valid
    unless ``core`` is requested. No commands, models, or network calls are run.
    """
    root = Path(directory).expanduser()
    errors = []
    try:
        if (root / "SKILL.md").is_file():
            skills = [root]
        else:
            skills = sorted(
                child for child in root.iterdir()
                if child.is_dir() and not child.name.startswith((".", "_"))
                and (child / "SKILL.md").is_file()
            )
        if not skills:
            return [Diagnostic(str(root), "source", "No SKILL.md files found.")]
        for skill in skills:
            for path in sorted(skill.rglob("*.md")):
                try:
                    text = path.read_text(encoding="utf-8")
                    if path == skill / "SKILL.md":
                        errors.extend(_check_frontmatter(text, path, core))
                    errors.extend(_check_links(text, path))
                except (OSError, UnicodeError) as exc:
                    errors.append(Diagnostic(str(path), "read", str(exc)))
    except (OSError, ValueError, RuntimeError) as exc:
        errors.append(Diagnostic(str(root), "source", str(exc)))
    return errors
