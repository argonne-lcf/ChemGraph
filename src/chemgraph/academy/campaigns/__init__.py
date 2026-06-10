from __future__ import annotations

import dataclasses
from importlib import resources
from pathlib import Path


EXAMPLE_002 = 'example-002-mace-ensemble-screening'

CAMPAIGNS = {
    'mace-ensemble-screening-20': f'{EXAMPLE_002}/campaign.json',
}

LM_CONFIG_TEMPLATES = {
    'argo-gpt54-mace-template': f'{EXAMPLE_002}/lm_config.json',
}


@dataclasses.dataclass(frozen=True)
class CampaignLaunchDefaults:
    """Runtime defaults for a packaged ChemGraph Academy campaign."""

    lm_config_template: str
    agent_count: int
    agents_per_node: int
    max_decisions: int


CAMPAIGN_LAUNCH_DEFAULTS = {
    'mace-ensemble-screening-20': CampaignLaunchDefaults(
        lm_config_template='argo-gpt54-mace-template',
        agent_count=5,
        agents_per_node=1,
        max_decisions=24,
    ),
}


def _resolve_campaign_asset(
    path_or_name: str | Path,
    known_assets: dict[str, str],
) -> Path:
    value = str(path_or_name)
    path = Path(value)
    if path.exists():
        return path.resolve()
    relative = known_assets.get(value)
    if relative is None:
        return path
    return Path(str(resources.files(__package__).joinpath(relative)))


def resolve_campaign(path_or_name: str | Path) -> Path:
    return _resolve_campaign_asset(path_or_name, CAMPAIGNS)


def resolve_lm_config_template(path_or_name: str | Path) -> Path:
    return _resolve_campaign_asset(path_or_name, LM_CONFIG_TEMPLATES)


def list_campaigns() -> list[str]:
    return sorted(CAMPAIGNS)


def campaign_launch_defaults(campaign: str) -> CampaignLaunchDefaults:
    try:
        return CAMPAIGN_LAUNCH_DEFAULTS[campaign]
    except KeyError as exc:
        raise KeyError(
            f'No launch defaults for campaign {campaign!r}',
        ) from exc
