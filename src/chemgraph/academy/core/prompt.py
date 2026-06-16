from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PromptStateLimits(BaseModel):
    """Visibility limits for state included in each logical-agent prompt."""

    model_config = ConfigDict(extra='forbid')

    received_messages_last_n: int = Field(ge=0)
    tool_results_last_n: int = Field(ge=0)
    actions_last_n: int = Field(ge=0)


class PromptProfile(BaseModel):
    """Prompt/rendering profile shared by logical agents in a campaign run."""

    model_config = ConfigDict(extra='forbid')

    prompt_version: str
    prompt_style: Literal['json_state']
    system_prompt: str
    protocol_prompt: str
    langchain_recursion_limit: int = Field(ge=4)
    state_limits: PromptStateLimits


def load_prompt_profile(path: str | Path) -> PromptProfile:
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return PromptProfile.model_validate(data)
