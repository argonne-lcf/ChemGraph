import json
from dataclasses import asdict
from chemgraph.service.adapters import Outcome

class FixtureAdapter:
    """Deterministic text-only conversation fixture, never a scientific runtime."""

    def __init__(self, settings):
        self.settings = settings

    async def start(self):
        pass

    async def close(self):
        pass

    async def prepare(self, record):
        return None

    async def run(self, record):
        directory = self.settings.state_dir / "artifacts" / record["id"]
        # Message IDs are never used as paths.
        cache = directory / f"outcome-{record['digest']}.json"
        if cache.exists():
            return Outcome(**json.loads(cache.read_text()))
        text = record["text"]
        if record["action"] == "resume":
            outcome = Outcome(text=f"Clarification received: {text}")
        elif text.lower().startswith("clarify"):
            outcome = Outcome(
                state="input_required",
                text="What additional information should I use?",
                reason="clarification",
                interrupts=[
                    {
                        "id": record["id"],
                        "payload": {"question": "Please provide additional information."},
                    }
                ],
            )
        else:
            outcome = Outcome(text=f"Fixture received: {text}")
        directory.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(outcome)))
        temporary.replace(cache)
        return outcome

    async def recover(self, record):
        # All fixture operations are bounded, local and idempotent.
        return await self.run(record)
