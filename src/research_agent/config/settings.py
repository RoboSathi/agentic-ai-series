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
from research_agent.domain.scope import Mode

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
    qa: str
    research: str


class MessageSettings(BaseModel):
    """Every agent-authored sentence the user sees (none are model-written)."""

    refuse: str
    # `{options}` becomes the numbered menu, one `reframe_option` line each.
    reframe: str
    reframe_option: str
    # Said when a question goes straight to a quick answer, and after it.
    qa_started: str
    qa_followup: str
    # Added to a FAILED run in demo mode: the question is outside Kelvari.
    demo_off_world: str
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


class GreetingSettings(BaseModel):
    """The lines the greeting is assembled from (console/greeting.py)."""

    live_mode: str
    demo_mode: str
    topic_line: str
    modes_title: str
    qa_line: str
    research_line: str
    limits: str
    wont: str
    commands: str


class SetupMessages(BaseModel):
    """What start-up says when the live posture is missing a credential."""

    # Fatal: a routed role needs a model key that is not set.
    missing_model_key: str
    # Not fatal: without a Tavily key, search is DuckDuckGo only.
    no_search_key: str
    # The month's metered searches, for the greeting and the side panel.
    search_usage: str


class ReframeOption(BaseModel):
    """One choice in the reframe menu: a mode, how it is named, an example."""

    mode: Mode
    label: str
    example: str
    # Words a reply may use instead of the number or the label.
    aliases: list[str] = Field(default_factory=list)


class TimeWindowSettings(BaseModel):
    """How "in the last 1 month" becomes a day count (all patterns case-insensitive)."""

    # "3 weeks", "30 days": group 1 the amount, group 2 the unit.
    amount_pattern: str
    # "last month", "past week" (one unit): group 1 the unit.
    relative_pattern: str
    # "this week", "this month": group 1 the unit.
    current_pattern: str
    # "today", "yesterday": one day.
    today_pattern: str
    unit_days: dict[str, int]


class ScopingSettings(BaseModel):
    """The interview: its turn cap, its guard, and how answers are read."""

    max_turns: int = Field(ge=1)
    challenge_every_k_turns: int = Field(ge=1)
    checklists_path: str
    refuse_patterns: list[str]
    reframe_patterns: list[str]
    archetype_keywords: dict[str, list[str]]
    # A message naming no specialised kind but asking for a *research run*.
    research_cues: list[str]
    reframe_options: list[ReframeOption]
    time_window: TimeWindowSettings
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
    # A Q&A or general question bounded to this many days or fewer is news:
    # search the providers' news vertical (`news_topic`).
    news_window_days: int = Field(ge=1)
    news_topic: str
    # The budget stop recorded when every search provider is out of quota.
    quota_stop: str


class QaSettings(BaseModel):
    """The Q&A fast path: no interview, a small budget, one follow-up round."""

    budget: Budget


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
    # `make runs ra`: the archive read straight from the artifact store.
    list_headers: list[str]
    list_empty: str


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


class ReferenceJudgeSettings(BaseModel):
    """The offline judge the fixture backend grades soft quality with."""

    # Lines that open with these state a trade-off and its counterweight.
    tradeoff_markers: list[str]
    # The heading a decision report leads with (trade-offs only apply there).
    decision_heading: str
    # A PARTIAL report must say so under this heading.
    limits_heading: str
    partial_status: str
    # How the platform's judge prompt labels its parts, and the status line.
    conversation_label: str
    tool_calls_label: str
    evidence_label: str
    evidence_separator: str
    status_prefix: str


class ResearchEvalSettings(BaseModel):
    """Where the eval cases live and how each one is isolated and checked."""

    dataset_dir: str
    suite_files: dict[str, str]
    security_suite: str
    judge_rubric_path: str
    session_prefix: str
    workspace_root: str
    keep_case_workspaces: bool
    # A query no fixture records: replay must raise on it, never improvise.
    probe_query: str
    # Roles that must never see raw page text (only the extractor may).
    screened_roles: list[str]
    # Page sentences shorter than this are too generic to prove a leak.
    min_leak_sentence_chars: int = Field(ge=1)
    reference_judge: ReferenceJudgeSettings

    @property
    def resolved_dataset_dir(self) -> Path:
        return resolve_path(self.dataset_dir)

    @property
    def resolved_workspace_root(self) -> Path:
        return resolve_path(self.workspace_root)

    @property
    def resolved_judge_rubric_path(self) -> Path:
        return resolve_path(self.judge_rubric_path)


class ResearchAgentSettings(YamlBackedSettings):
    """The research agent's own wiring values."""

    _yaml_section = "research_agent"

    title: str
    placeholder: str
    greeting: GreetingSettings
    model_backend: ModelBackend
    fixture_world_path: str
    command_names: CommandNames
    messages: MessageSettings
    setup: SetupMessages
    scoping: ScopingSettings
    rubric: RubricSettings
    budget: Budget
    qa: QaSettings
    planning: PlanningSettings
    matching: MatchingSettings
    research: ResearchSettings
    reflection: ReflectionSettings
    synthesis: SynthesisSettings
    audit: AuditSettings
    archive: ArchiveSettings
    cancel: CancelSettings
    tui: PanelSettings
    evals: ResearchEvalSettings

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
