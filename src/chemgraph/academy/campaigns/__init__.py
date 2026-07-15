from __future__ import annotations

import dataclasses
import os
from importlib import resources
from pathlib import Path

# When set, resolve_campaign checks ``<root>/<name>/campaign.jsonc``
# before falling back to the shipped example. This is how dashboard
# authoring-mode edits reach the daemon: the dashboard writes the
# edited campaign into ``<root>/<name>/campaign.jsonc`` and exports
# this env var in the PBS script so the compute-side resolve picks
# it up. On the laptop the same var routes to
# ``~/.chemgraph-academy/user-campaigns/``.
USER_CAMPAIGNS_ROOT_ENV = 'CHEMGRAPH_USER_CAMPAIGNS_ROOT'


MOF_CRUX_AURORA = 'mof-crux-aurora'
MOF_CRUX_AURORA_MOCK = 'mof-crux-aurora-mock'

CAMPAIGNS = {
    'mof-crux-aurora': f'{MOF_CRUX_AURORA}/campaign.jsonc',
    'mof-crux-aurora-mock': f'{MOF_CRUX_AURORA_MOCK}/campaign.jsonc',
}

LM_CONFIG_TEMPLATES = {
    'argo-gpt5mini-federated-chat': f'{MOF_CRUX_AURORA}/lm_config.json',
}


@dataclasses.dataclass(frozen=True)
class CampaignLaunchDefaults:
    """Runtime defaults for a packaged ChemGraph Academy campaign.

    Loaded from a ``launch_defaults`` block inside each campaign.jsonc:

        "launch_defaults": {
            "lm_config_template": "argo-gpt54-mace-template",
            "agent_count": 5,
            "agents_per_node": 1,
            "max_decisions": 24
        }

    User-copied campaigns (created via the canvas) get to override
    these without a code edit.
    """

    lm_config_template: str
    agent_count: int
    agents_per_node: int
    max_decisions: int


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
    # User-edited copies at $USER_CAMPAIGNS_ROOT/<name>/campaign.jsonc
    # win over shipped examples. Canvas-created campaigns (not in the
    # shipped CAMPAIGNS registry) also live here -- pre-2026-07-07 the
    # user-copy check was gated on "name is in CAMPAIGNS" which meant
    # canvas-only campaigns silently fell through to a nonexistent
    # shipped-path lookup.
    value = str(path_or_name)
    explicit_path = Path(value)
    # Only apply the user-copy override for name-lookups, not for
    # absolute / explicit paths the caller passed in.
    is_name_lookup = not explicit_path.is_absolute() and not explicit_path.exists()
    if is_name_lookup:
        # Try each candidate root, in order:
        # 1. $CHEMGRAPH_USER_CAMPAIGNS_ROOT (set by the PBS script on
        #    compute nodes so daemons find rsync'd copies at the HPC
        #    remote_root/user-campaigns/).
        # 2. ~/.chemgraph-academy/user-campaigns/ (canonical laptop
        #    location; where the canvas writes and where the laptop-
        #    side inject subprocess needs to find canvas-created
        #    campaigns that don't exist in the shipped registry).
        # Pre-2026-07-07 the laptop path was skipped, and canvas-only
        # campaigns silently fell through to a shipped-package lookup
        # that returned garbage.
        candidate_roots = []
        env_root = os.environ.get(USER_CAMPAIGNS_ROOT_ENV)
        if env_root:
            candidate_roots.append(Path(env_root))
        candidate_roots.append(
            Path.home() / '.chemgraph-academy' / 'user-campaigns'
        )
        for root in candidate_roots:
            user_copy = root / value / 'campaign.jsonc'
            if user_copy.exists():
                return user_copy.resolve()
    return _resolve_campaign_asset(path_or_name, CAMPAIGNS)


def resolve_lm_config_template(path_or_name: str | Path) -> Path:
    return _resolve_campaign_asset(path_or_name, LM_CONFIG_TEMPLATES)


def list_campaigns() -> list[str]:
    return sorted(CAMPAIGNS)


def campaign_launch_defaults(campaign: str) -> CampaignLaunchDefaults:
    """Read launch_defaults from the campaign JSON.

    Fallback: if the block is missing (older user copies), returns a
    zero-agent-count sentinel so the caller must supply --agent-count
    explicitly rather than silently launching zero ranks.
    """
    from chemgraph.academy.core.campaign import _load_jsonc

    path = resolve_campaign(campaign)
    if not path.exists():
        raise KeyError(f'campaign {campaign!r} not found')
    data = _load_jsonc(path)
    block = data.get('launch_defaults') or {}
    return CampaignLaunchDefaults(
        lm_config_template=block.get('lm_config_template', ''),
        agent_count=int(block.get('agent_count') or 0),
        agents_per_node=int(block.get('agents_per_node') or 1),
        max_decisions=int(block.get('max_decisions') or 6),
    )
