import pytest

from contextlib import asynccontextmanager

import httpx

from tests.service.fixtures import FixtureAdapter
from chemgraph.service.config import Principal, Settings

TOKENS = {"alice": "a" * 40, "bob": "b" * 40, "operator": "o" * 40, "notifier": "n" * 40}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    principals = []
    for name, token in TOKENS.items():
        monkeypatch.setenv(f"TEST_{name}", token)
        principals.append(
            Principal(
                name=name,
                token_env=f"TEST_{name}",
                role=name if name in {"operator", "notifier"} else "peer",
            )
        )
    return Settings(state_dir=tmp_path, principals=tuple(principals))


@asynccontextmanager
async def running(settings, adapter=None, owner="alice"):
    from chemgraph.service.client import Client
    from chemgraph.service.server import create_app

    app = create_app(settings, adapter or FixtureAdapter(settings))
    async with app.router.lifespan_context(app):
        async with Client(
            "http://test", TOKENS[owner], transport=httpx.ASGITransport(app)
        ) as client:
            yield app.state.service, client, app


class ArtifactAdapter(FixtureAdapter):
    """Test-only file producer; core transport tests perform no chemistry."""

    async def run(self, record):
        import json

        from chemgraph.service.adapters import Outcome

        path = self.settings.state_dir / "artifacts" / record["id"] / "note.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"message": record["text"]}))
        return Outcome(
            text="Test artifact ready",
            artifacts=[
                {
                    "path": str(path),
                    "media_type": "application/json",
                }
            ],
        )
