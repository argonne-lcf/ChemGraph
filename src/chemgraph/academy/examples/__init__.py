from __future__ import annotations

import dataclasses
from importlib import resources
from pathlib import Path


EXAMPLE_002 = 'example-002-mace-ensemble-screening'

BUILTIN_CAMPAIGNS = {
    'mace-ensemble-screening-20': f'{EXAMPLE_002}/campaign.jsonc',
}

BUILTIN_LM_CONFIG_TEMPLATES = {
    'argo-gpt54-template': f'{EXAMPLE_002}/lm_config.template.json',
    'argo-gpt54-mace-template': f'{EXAMPLE_002}/lm_config.template.json',
}


@dataclasses.dataclass(frozen=True)
class CampaignLaunchDefaults:
    """Runtime defaults for a built-in ChemGraph Academy campaign."""

    lm_config_template: str
    agent_count: int
    agents_per_node: int
    max_decisions: int


BUILTIN_CAMPAIGN_LAUNCH_DEFAULTS = {
    'mace-ensemble-screening-20': CampaignLaunchDefaults(
        lm_config_template='argo-gpt54-mace-template',
        agent_count=5,
        agents_per_node=1,
        max_decisions=24,
    ),
}


def _resolve_builtin(
    path_or_name: str | Path,
    builtins: dict[str, str],
) -> Path:
    value = str(path_or_name)
    path = Path(value)
    if path.exists():
        return path.resolve()
    relative = builtins.get(value)
    if relative is None:
        return path
    return Path(str(resources.files(__package__).joinpath(relative)))


def resolve_builtin_campaign(path_or_name: str | Path) -> Path:
    return _resolve_builtin(path_or_name, BUILTIN_CAMPAIGNS)


def resolve_builtin_lm_config_template(path_or_name: str | Path) -> Path:
    return _resolve_builtin(path_or_name, BUILTIN_LM_CONFIG_TEMPLATES)


def list_builtin_campaigns() -> list[str]:
    return sorted(BUILTIN_CAMPAIGNS)


def list_builtin_lm_config_templates() -> list[str]:
    return sorted(BUILTIN_LM_CONFIG_TEMPLATES)


def campaign_launch_defaults(campaign: str) -> CampaignLaunchDefaults:
    try:
        return BUILTIN_CAMPAIGN_LAUNCH_DEFAULTS[campaign]
    except KeyError as exc:
        raise KeyError(
            f'No built-in launch defaults for campaign {campaign!r}',
        ) from exc
