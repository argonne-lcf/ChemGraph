import pytest
pytest.importorskip("a2a")



def test_remote_http_requires_https(settings):
    with pytest.raises(ValueError, match="HTTPS"):
        type(settings).model_validate(
            {**settings.model_dump(), "public_url": "http://remote.example"}
        )


@pytest.mark.parametrize("profile", [{"version": 1}, {"tools": []}, {"tools": ["emt_energy"]}])
def test_legacy_profile_rejected_with_migration(profile):
    from chemgraph.service.config import Profile

    with pytest.raises(ValueError, match="remove profile.tools"):
        Profile.model_validate(profile)


async def test_fixture_is_text_only(settings):
    from tests.service.conftest import running

    async with running(settings) as (_, client, _):
        task = (await client.send("Calculate hydrogen EMT energy", blocking=True)).task
        assert not task.artifacts
        assert task.status.message.parts[0].text.startswith("Fixture received:")


def test_model_switch_changes_checkpoint_identity():
    from chemgraph.service.config import Profile

    old = Profile(adapter="chemgraph", model="model-a")
    new = old.model_copy(update={"model": "model-b"})
    assert new.identity != old.identity


pytestmark = pytest.mark.asyncio


def test_production_fixture_mode_is_rejected():
    from chemgraph.service.config import Profile

    with pytest.raises(ValueError):
        Profile(adapter='fixture')


async def test_old_connect_state_is_retained_but_not_opened(settings):
    from chemgraph.service.service import Service

    legacy = settings.state_dir / 'connect.db'
    legacy.write_bytes(b'old state retained for inspection')
    with pytest.raises(RuntimeError, match='fresh service state'):
        await Service(settings).start()
    assert legacy.read_bytes() == b'old state retained for inspection'
    assert not (settings.state_dir / 'service.db').exists()
