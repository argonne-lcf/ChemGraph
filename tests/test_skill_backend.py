"""Packaged skill protocol and broken-distribution regressions."""

import asyncio

import pytest

from chemgraph.skills.backend import BundledSkillsBackend


@pytest.mark.parametrize("asynchronous", [False, True])
def test_bundled_file_protocol(asynchronous):
    backend = BundledSkillsBackend()

    def call(name, *args, **kwargs):
        if asynchronous:
            return asyncio.run(getattr(backend, "a" + name)(*args, **kwargs))
        return getattr(backend, name)(*args, **kwargs)

    assert {entry["path"] for entry in call("ls", "/").entries} == {
        "/chemgraph/",
        "/pbs-hpc/",
    }
    assert len(call("glob", "**/SKILL.md").matches) == 2
    assert len(call("glob", "SKILL.md", "/chemgraph/").matches) == 1
    assert "Use ChemGraph" in call("read", "/chemgraph/SKILL.md").file_data["content"]
    assert call("grep", "PBS", "/pbs-hpc/", max_count=1).matches
    downloaded = call(
        "download_files", ["/chemgraph/SKILL.md", "/absent", "/../secret"]
    )
    assert downloaded[0].content.startswith(b"---\nname: chemgraph\n")
    assert downloaded[1].error == "file_not_found"
    assert downloaded[2].error == "invalid_path"
    assert call("read", "/../secret").error
    assert call("ls", "/missing").error
    original = downloaded[0].content
    assert call("write", "/chemgraph/SKILL.md", "changed").error
    assert call("edit", "/chemgraph/SKILL.md", "chemgraph", "changed").error
    assert call("delete", "/chemgraph/").error
    assert call("upload_files", [("/chemgraph/SKILL.md", b"changed")])[0].error
    assert call("download_files", ["/chemgraph/SKILL.md"])[0].content == original


@pytest.mark.parametrize("damage", ["missing", "malformed"])
def test_invalid_bundled_distribution_fails(monkeypatch, tmp_path, damage):
    (tmp_path / "chemgraph").mkdir()
    (tmp_path / "chemgraph/SKILL.md").write_text(
        "---\nname: chemgraph\ndescription: Test skill\n---\nBody\n"
    )
    if damage == "malformed":
        (tmp_path / "pbs-hpc").mkdir()
        (tmp_path / "pbs-hpc/SKILL.md").write_text("no frontmatter")
    monkeypatch.setattr("chemgraph.skills.backend.resources.files", lambda _: tmp_path)
    with pytest.raises(ValueError, match="bundled skill"):
        BundledSkillsBackend()
