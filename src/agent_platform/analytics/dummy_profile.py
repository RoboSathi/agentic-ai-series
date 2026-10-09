"""What an agent's simulated traffic looks like: its `analytics/dummy.yaml` profile.

Each agent owns its profile, next to its own semantic layer, because only the agent
knows its task kinds, outcomes, tools and models; the platform's generator
(`analytics/dummy.py`) knows none of them. The outcome flags are not repeated
here: they are read from the agent's own overlay (`analytics.tasks.outcomes`), so
a profile can only name outcomes the agent actually declares.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.config.paths import resolve_path
from agent_platform.config.settings import OutcomeFlags

Range = tuple[int, int]


class TaskMix(BaseModel):
    """One kind of task: how often it is asked, how long it runs, how it ends."""

    weight: float = Field(gt=0)
    turns: Range
    outcomes: dict[str, float]
    # Share left open when the session ends (abandoned, or still open).
    open_rate: float = Field(default=0.0, ge=0, le=1)


class RoleModels(BaseModel):
    """A model role: its provider, primary model and backups, and its weight among steps."""

    provider: str
    primary: str
    backups: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, gt=0)
    latency_ms: float = Field(gt=0)
    prompt_tokens: Range
    completion_tokens: Range


class ToolMix(BaseModel):
    """One tool: how often it is proposed, whether it writes, how it fails."""

    weight: float = Field(gt=0)
    side_effecting: bool = False
    failure_rate: float = Field(default=0.0, ge=0, le=1)
    timeout_rate: float = Field(default=0.0, ge=0, le=1)
    duration_ms: float = Field(gt=0)


class CheckSpec(BaseModel):
    """One check: what answers it, how fast, and how often each verdict comes back."""

    backend: str
    model: str = ""
    latency_ms: float = Field(gt=0)
    block_rate: float = Field(default=0.0, ge=0, le=1)
    flag_rate: float = Field(default=0.0, ge=0, le=1)
    failure_rate: float = Field(default=0.0, ge=0, le=1)
    # A dependency is consulted this many times per step, on average.
    per_step: float = Field(default=0.0, ge=0)


class GroundingSpec(BaseModel):
    """The grounding check on replies: claims per reply and how many are unsupported."""

    backend: str
    model: str = ""
    latency_ms: float = Field(gt=0)
    claims: Range
    unsupported_rate: float = Field(ge=0, le=1)
    # Step 1 (token confidence): share of reply sentences written with low confidence, and
    # the language model that wrote them; None = the model returns no logprobs.
    low_confidence_rate: float | None = Field(default=None, ge=0, le=1)
    writer_model: str = ""


class Tier2Hook(BaseModel):
    """An agent's own facts for its tasks that ended a given way (`module:function`)."""

    hook: str
    outcomes: list[str]
    # Whatever the hook's own simulation needs (its mixes and timings), from the profile.
    params: dict[str, Any] = Field(default_factory=dict)


class DummyProfile(BaseModel):
    """One agent's simulated traffic."""

    agent_id: str
    # The agent's overlay, where its task outcomes and their flags are declared.
    overlay: str
    sessions_per_day: float = Field(gt=0)
    tasks: dict[str, TaskMix]
    steps: Range
    step_cap: int | None = None
    # Of turns that reach the cap, the share that end on it (a loop abort).
    cap_abort_rate: float = Field(default=0.0, ge=0, le=1)
    roles: dict[str, RoleModels]
    # Chance a step proposes tools, and how many it proposes when it does.
    tool_step_rate: float = Field(ge=0, le=1)
    tools_per_step: Range = (1, 1)
    tools: dict[str, ToolMix] = Field(default_factory=dict)
    # Share of proposals that stop before execution, by stage.
    stage_rates: dict[str, float] = Field(default_factory=dict)
    governed: bool = True
    # Of governed refusals, the share that escalate to a person (the rest are denied).
    escalate_share: float = Field(default=0.0, ge=0, le=1)
    screens: dict[str, CheckSpec] = Field(default_factory=dict)
    # Share of turns a declared screen does not run on (a halt before screening).
    unscreened_rate: float = Field(default=0.0, ge=0, le=1)
    dependencies: dict[str, CheckSpec] = Field(default_factory=dict)
    grounding: GroundingSpec | None = None
    turn_overhead_ms: float = Field(gt=0)
    tier2: Tier2Hook | None = None

    @cached_property
    def outcome_flags(self) -> dict[str, OutcomeFlags]:
        """The agent's declared outcomes (its overlay), with the platform's reserved ones."""
        raw: dict[str, Any] = yaml.safe_load(resolve_path(self.overlay).read_text()) or {}
        declared = raw.get("analytics", {}).get("tasks", {}).get("outcomes", {})
        flags = dict(get_settings().analytics.tasks.outcomes)
        flags.update({name: OutcomeFlags.model_validate(v) for name, v in declared.items()})
        return flags

    def check_outcomes(self) -> None:
        """Every outcome the profile names is one the agent declares (else a ValueError)."""
        named = {o for mix in self.tasks.values() for o in mix.outcomes}
        missing = sorted(named - set(self.outcome_flags))
        if missing:
            raise ValueError(f"{self.agent_id}: outcomes not declared in its overlay: {missing}")


def load_profiles() -> list[DummyProfile]:
    """Every agent's profile, found by convention (`analytics.dummy.profiles_glob`)."""
    from agent_platform.config.paths import REPO_ROOT

    pattern = get_settings().analytics.dummy.profiles_glob
    profiles = [_load(path) for path in sorted(REPO_ROOT.glob(pattern))]
    for profile in profiles:
        profile.check_outcomes()
    return profiles


def _load(path: Path) -> DummyProfile:
    return DummyProfile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


__all__ = [
    "CheckSpec",
    "DummyProfile",
    "GroundingSpec",
    "RoleModels",
    "TaskMix",
    "Tier2Hook",
    "ToolMix",
    "load_profiles",
]
