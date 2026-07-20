# Contributing

ChemGraph uses a **trunk-based** workflow: `main` is always releasable, and work
lands in **small, frequent PRs** from short-lived branches. There is no
long-lived `dev` branch.

At a glance:

- Branch off the latest `main` (`feature/…`, `fix/…`, `docs/…`, `chore/…`).
- Keep each PR to one logical change (roughly ≤ ~400 lines where practical);
  split large features into incremental PRs.
- Before opening a PR, run `ruff check .` and `pytest tests/ -k "not tblite"`.
- A PR merges once **CI is green** and it has **at least one approval**; merges
  use a merge commit and the branch is deleted afterward.
- Releases are cut from `main`: bump `version` in `pyproject.toml`, then publish
  a `vX.Y.Z` GitHub Release (which triggers the PyPI publish workflow).

The full guide, including development setup and the release process, lives in
[`CONTRIBUTING.md`](https://github.com/argonne-lcf/ChemGraph/blob/main/CONTRIBUTING.md)
at the repository root. See also
[Code Formatting & Linting](code_formatting_and_linting.md).
