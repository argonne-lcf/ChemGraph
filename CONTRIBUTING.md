# Contributing to ChemGraph

Thanks for your interest in improving ChemGraph! This document describes how we
develop, review, and release the project. The goal of this workflow is to keep
`main` healthy and to land changes in **small, frequent, reviewable pieces**.

## TL;DR

- `main` is the single source of truth and is always releasable. There is **no
  long-lived `dev` branch** — work happens on short-lived branches off `main`.
- Open a **small, focused** pull request (one logical change).
- A PR merges when **CI is green** and it has **at least one approval**.
- Be kind, assume good intent, and prefer many small PRs over one big one.

## Branching model

We use a trunk-based model with short-lived topic branches:

1. Branch off the latest `main`.
2. Make a focused change.
3. Open a PR back into `main`.
4. Merge once it passes CI and review, then delete the branch.

We previously accumulated work on a long-lived `dev` branch, which led to large,
hard-to-review PRs. **That model is retired** — please target `main` directly
with short-lived branches.

### Branch naming

Use a type prefix and a short kebab-case description; include an issue number
when there is one:

```
feature/<slug>      e.g. feature/globus-transfer-retry
fix/<slug>          e.g. fix/mcp-relative-path
docs/<slug>         e.g. docs/contributing-guide
chore/<slug>        e.g. chore/bump-langgraph
fix/123-<slug>      when it closes issue #123
```

## Keep pull requests small

Small PRs get reviewed faster, are safer to merge, and are easier to revert.

- Aim for **one logical change per PR** (roughly **≤ ~400 lines** of diff where
  practical — not a hard limit, but a nudge).
- Split large features into a series of incremental PRs. Land scaffolding and
  tests first, then build on top.
- Use **optional extras** and/or **feature flags** so partially complete work
  can merge safely behind tests instead of waiting on one giant branch.
- Keep unrelated changes (refactors, formatting, renames) in separate PRs.

## Using AI assistants

AI coding assistants (Claude Code, Cursor, Copilot, etc.) are welcome, and this
repo ships an [`AGENTS.md`](AGENTS.md) with repo-specific commands and
conventions for them. If you use one, please:

- **You are responsible for the code you submit.** Review and understand every
  AI-generated line as if you wrote it — the same review bar applies.
- **Make sure it actually works:** run `ruff check .` and the test suite, and
  add tests for new behavior. Don't submit unverified generated code.
- **Never paste secrets or non-public data** (API keys, tokens, credentials,
  unpublished research) into a prompt.
- **Watch licensing and attribution** — don't submit code the assistant may have
  reproduced from an incompatible license.
- Keep AI-assisted PRs **small and focused**, like any other PR. Large,
  loosely-reviewed generated diffs are the main risk we want to avoid.
- A brief note in the PR that AI assistance was used is appreciated (not
  required).

## Development setup

```bash
git clone https://github.com/argonne-lcf/ChemGraph.git
cd ChemGraph
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core
# extras as needed, e.g.:
pip install -e ".[academy,parsl,globus_compute]"
```

## Before you open a PR

Run the same gates CI runs:

```bash
# Lint
ruff check .

# Core tests
pytest tests/ -k "not tblite"

# If you touched the Academy / execution-backend features:
pip install -e ".[academy,parsl,globus_compute]"
pytest tests/ -k "not tblite"
```

- Add or update tests for the behavior you change.
- Update docs (`docs/`, `README.md`) when you change user-facing behavior.
- See [docs/code_formatting_and_linting.md](docs/code_formatting_and_linting.md)
  for style details. New code should read like the surrounding code.

## Opening the pull request

- Fill out the PR template checklist.
- Write a clear description: what changed, why, and how you tested it.
- Link related issues (`Closes #123`).
- Draft PRs are welcome for early feedback — mark them "Draft".

## Review and merge

- PRs require **at least one approving review** and **all required checks
  green** (lint + tests) before merging.
- Address review comments by pushing follow-up commits (don't force-push during
  active review unless asked — it makes re-review harder).
- Merges into `main` use a **merge commit** (the repo default). Your branch is
  deleted automatically after merge.
- Prefer letting the maintainer who reviewed do the merge, or merge yourself
  once approved and green.

## Releases

Releases are cut from `main`:

1. Open a small PR that bumps `version` in `pyproject.toml` (e.g. `0.6.0`).
2. After it merges, draft a GitHub Release with tag `vX.Y.Z` targeting `main`.
3. Publishing the release triggers the PyPI publish workflow
   (`.github/workflows/pypi-publish.yml`).

We follow semantic-versioning-ish conventions (pre-1.0): bump the **minor**
version for new features, the **patch** version for fixes.

## Reporting bugs and requesting features

Open a GitHub issue with enough detail to reproduce (versions, OS, a minimal
example, and the full error). Security-sensitive reports should be sent
privately to the maintainers rather than filed as public issues.

## Code of conduct

This project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md) (the
[Contributor Covenant](https://www.contributor-covenant.org) v2.1). By
participating, you are expected to uphold it. Please report unacceptable
behavior as described there.
