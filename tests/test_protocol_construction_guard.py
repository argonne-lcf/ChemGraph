"""Guard: chat clients are constructed only in the protocol builders.

After PR 2, no module under ``chemgraph/models`` or ``chemgraph/agent`` may
construct ``ChatOpenAI``, ``ChatAnthropic``, or ``ChatGoogleGenerativeAI``
except the corresponding ``protocols/*`` builder. The agent layer now routes all
model loading through ``load_chat_model`` / ``load_chat_model_prepared``.
"""

from __future__ import annotations

import pathlib

CHEMGRAPH_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "chemgraph"

# Directories scanned for stray client construction.
SCANNED_DIRS = (
    CHEMGRAPH_DIR / "models",
    CHEMGRAPH_DIR / "agent",
)

# Each client class may be constructed only in its designated builder file.
# Paths are relative to ``src/chemgraph``.
ALLOWED = {
    "ChatOpenAI(": "models/protocols/openai_compatible.py",
    "ChatAnthropic(": "models/protocols/anthropic_native.py",
    "ChatGoogleGenerativeAI(": "models/protocols/google_native.py",
}


def _offenders(needle: str, allowed_rel: str) -> list[str]:
    allowed = (CHEMGRAPH_DIR / allowed_rel).resolve()
    hits: list[str] = []
    for scanned in SCANNED_DIRS:
        for path in scanned.rglob("*.py"):
            if path.resolve() == allowed:
                continue
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if needle in line:
                    rel = path.relative_to(CHEMGRAPH_DIR)
                    hits.append(f"{rel}:{lineno}: {line.strip()}")
    return hits


def test_no_direct_client_construction_in_models_and_agent():
    problems: list[str] = []
    for needle, allowed_rel in ALLOWED.items():
        problems.extend(_offenders(needle, allowed_rel))
    assert not problems, "Unexpected direct client construction:\n" + "\n".join(problems)
