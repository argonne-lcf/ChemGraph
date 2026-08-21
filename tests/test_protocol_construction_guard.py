"""Guard: chat clients are constructed only in the protocol builders.

After PR 1, no module under ``chemgraph/models`` may construct ``ChatOpenAI``,
``ChatAnthropic``, or ``ChatGoogleGenerativeAI`` except the corresponding
``protocols/*`` builder. (Agent-layer call sites in ``chemgraph/agent`` are
migrated in PR 2 and are intentionally out of scope for this guard.)
"""

from __future__ import annotations

import pathlib

MODELS_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "chemgraph" / "models"

# Each client class may be constructed only in its designated builder file.
ALLOWED = {
    "ChatOpenAI(": "protocols/openai_compatible.py",
    "ChatAnthropic(": "protocols/anthropic_native.py",
    "ChatGoogleGenerativeAI(": "protocols/google_native.py",
}


def _offenders(needle: str, allowed_rel: str) -> list[str]:
    allowed = (MODELS_DIR / allowed_rel).resolve()
    hits: list[str] = []
    for path in MODELS_DIR.rglob("*.py"):
        if path.resolve() == allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if needle in line:
                hits.append(f"{path.relative_to(MODELS_DIR)}:{lineno}: {line.strip()}")
    return hits


def test_no_direct_client_construction_in_models():
    problems: list[str] = []
    for needle, allowed_rel in ALLOWED.items():
        problems.extend(_offenders(needle, allowed_rel))
    assert not problems, "Unexpected direct client construction:\n" + "\n".join(problems)
