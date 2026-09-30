"""Configuration for the research agent.

These settings live with the agent, not with the platform: the section is
declared in this package's overlay and read here, so `agentic_commons`
never learns what a research run is. Every tunable the agent reads is a
field here — nothing is a literal in the code that uses it.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from agentic_commons.config.paths import resolve_path
from agentic_commons.config.settings import YamlBackedSettings
from research_agent.domain.budget import Budget

ModelBackend = Literal["fixture", "routed"]


class CommandNames(BaseModel):
    """The conversational slash commands, by role."""

    done: str
    waive: str
    rubric: str
    approve: str
    cancel: str
    history: str
    show: str
    rubric_add: str
    rubric_drop: str
    rubric_set: str


class MessageSettings(BaseModel):
    """Every agent-authored sentence the user sees (none are model-written)."""

    refuse: str
    reframe: str
    challenge_prefix: str
    propose_done: str
    turn_cap: str
    waived: str
    finish_scoping_first: str
    rubric_hint: str
    rubric_error: str
    approved: str
    run_busy: str
    no_active_run: str
    cancel_requested: str
    delivered: str
    new_scope: str
    history_empty: str
    unknown_run: str
    plan_failed: str
    brief_ready: str


class ScopingSettings(BaseModel):
    """The interview: its turn cap, its guard, and how answers are read."""

    max_turns: int = Field(ge=1)
    challenge_every_k_turns: int = Field(ge=1)
    checklists_path: str
    refuse_patterns: list[str]
    reframe_patterns: list[str]
    archetype_keywords: dict[str, list[str]]
    answer_split_pattern: str
    vague_answer_max_words: int = Field(ge=0)
    absolute_words: list[str]
    waive_all_keyword: str
    any_recency_words: list[str]
    none_answers: list[str]
    primary_sources_phrase: str

    @property
    def resolved_checklists_path(self) -> Path:
        return resolve_path(self.checklists_path)


class RubricSettings(BaseModel):
    templates_path: str

    @property
    def resolved_templates_path(self) -> Path:
        return resolve_path(self.templates_path)


class PlanningSettings(BaseModel):
    max_queries_per_node: int = Field(ge=1)


class MatchingSettings(BaseModel):
    """Deterministic term matching (research/matching.py)."""

    token_pattern: str
    stem_chars: int = Field(ge=1)
    min_token_chars: int = Field(ge=1)
    stopwords: list[str]
    synonyms: dict[str, list[str]]


class ResearchSettings(BaseModel):
    """The researcher worker and its defences."""

    max_results_per_search: int = Field(ge=1)
    max_fetches_per_node: int = Field(ge=1)
    max_claims_per_page: int = Field(ge=1)
    primary_source_markers: list[str]
    outbound_allow_patterns: list[str]
    outbound_deny_patterns: list[str]
    authority_rank: str
    # Why a search hit outside the approved recency window was not read.
    recency_rejection: str


class ReflectionSettings(BaseModel):
    max_followups_per_round: int = Field(ge=0)
    gap_fill_round_offset: int = Field(ge=1)


class SynthesisSettings(BaseModel):
    citation_pattern: str
    entry_id_prefix: str
    templates_path: str

    @property
    def resolved_templates_path(self) -> Path:
        return resolve_path(self.templates_path)


class AuditSettings(BaseModel):
    """How independent the Goldfish Audit's judge must be, and what its checks read."""

    independent_judge: bool
    # `platform`: the platform's configured NLI (local cross-encoder, hosted,
    # or lexical fallback); `lexical`: the deterministic fallback, always.
    entailment_model: Literal["platform", "lexical"]
    # What counts as a named entity in a claim (distinct-subject checks).
    entity_pattern: str


class ArchiveSettings(BaseModel):
    run_id_prefix: str
    history_limit: int = Field(ge=1)


class CancelSettings(BaseModel):
    key: str


class PanelSettings(BaseModel):
    """The console's presentation of this agent."""

    panel_title: str
    panel_width: int = Field(ge=10)
    stage_labels: dict[str, str]
    command_help: dict[str, str]
    panel_labels: dict[str, str]
    history_headers: list[str]


class ResearchAgentSettings(YamlBackedSettings):
    """The research agent's own wiring values."""

    _yaml_section = "research_agent"

    title: str
    placeholder: str
    greeting: str
    model_backend: ModelBackend
    fixture_world_path: str
    command_names: CommandNames
    messages: MessageSettings
    scoping: ScopingSettings
    rubric: RubricSettings
    budget: Budget
    planning: PlanningSettings
    matching: MatchingSettings
    research: ResearchSettings
    reflection: ReflectionSettings
    synthesis: SynthesisSettings
    audit: AuditSettings
    archive: ArchiveSettings
    cancel: CancelSettings
    tui: PanelSettings

    model_config = SettingsConfigDict(env_prefix="RESEARCH_AGENT__")

    @property
    def resolved_fixture_world_path(self) -> Path:
        return resolve_path(self.fixture_world_path)


_SETTINGS: ResearchAgentSettings | None = None


def get_agent_settings() -> ResearchAgentSettings:
    """The agent's settings, built once per process."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = ResearchAgentSettings()  # type: ignore[call-arg]
    return _SETTINGS


def reload_agent_settings() -> ResearchAgentSettings:
    """Rebuild from disk — used by tests that change the overlay."""
    global _SETTINGS
    _SETTINGS = ResearchAgentSettings()  # type: ignore[call-arg]
    return _SETTINGS
