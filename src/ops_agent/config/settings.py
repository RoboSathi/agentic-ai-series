"""The ops agent's own configuration (`ops_agent:` in its overlay)."""

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from agent_platform.config.settings import YamlBackedSettings


class PolicySettings(BaseModel):
    """The authority this agent runs under (the Gatekeeper's policy)."""

    role: str
    authority_rank: int = Field(ge=1)
    permissions: list[str]


class Words(BaseModel):
    """What the agent says in its own replies."""

    stopped: str
    model_failed: str
    blocked: str
    no_tools: str
    refused_tool: str
    budget_spent: str
    tools_spent: str
    cut_off: str
    invalid_call: str
    query_heading: str
    rows_heading: str
    more_rows: str
    ungrounded: str
    withheld: str
    sources_heading: str


class Cards(BaseModel):
    """Card headings on the live feed."""

    result_rows: str
    result_metrics: str
    result_views: str
    result_columns: str
    result_done: str
    unverified: str
    rule: str
    figure_level: str
    figure_tally: str
    no_figures: str
    query_check: str
    query_where: str
    not_found: str
    queries_searched: str


class SqlReviewCard(BaseModel):
    """The REFLECT step card's words (`ops_agent/guardrails/sql_review.py`)."""

    heading: str
    issues_suffix: str
    failed_heading: str
    asked: str
    query_line: str
    decision_revise: str
    decision_capped: str
    decision_accept: str
    decision_failed: str
    unparsed: str
    deterministic_heading: str
    deterministic_by: str
    call_line: str
    deterministic_reasoning: list[str]
    deterministic_decision: str
    formula_line: str
    capped_heading: str
    capped_by: str


class SqlReviewSettings(BaseModel):
    """Reflection on the SQL the model wrote, before the answer is grounded."""

    data_tools: list[str]
    reviewed_tools: list[str]
    max_reviews: int = Field(ge=0)
    sample_rows: int = Field(ge=0)
    temperature: float
    max_tokens: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    role: str
    purpose: str
    accept: str
    revise: str
    instructions: str
    request: str
    query_block: str
    truncated_mark: str
    revise_message: str
    card: SqlReviewCard


class OpsBudget(BaseModel):
    """A turn's budgets beside its step cap (`loop_control.LoopBudget`)."""

    max_tool_calls: int = Field(ge=1)
    max_total_tokens: int = Field(ge=1)
    max_seconds: float = Field(gt=0)


class OpsAgentSettings(YamlBackedSettings):
    """The ops agent's wiring values."""

    _yaml_section = "ops_agent"

    title: str
    placeholder: str
    greeting: str
    policy: PolicySettings
    max_steps: int = Field(ge=1)
    budget: OpsBudget
    history_turns: int = Field(ge=0)
    temperature: float
    max_tokens: int = Field(ge=1)
    retry_max_tokens: int = Field(ge=1)
    cut_off_finish: str
    table_rows: int = Field(ge=1)
    cell_chars: int = Field(ge=4)
    number_tolerance: float = Field(ge=0)
    free_numbers: list[float]
    # The id a prefetch read's call is shown to the model with (`{tool}`).
    prefetch_call_id: str
    sql_review: SqlReviewSettings
    words: Words
    stage_labels: dict[str, str]
    cards: Cards

    model_config = SettingsConfigDict(env_prefix="OPS_AGENT__")


_SETTINGS: OpsAgentSettings | None = None


def get_agent_settings() -> OpsAgentSettings:
    """The agent's settings, built once per process."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = OpsAgentSettings()  # type: ignore[call-arg]
    return _SETTINGS


def reload_agent_settings() -> OpsAgentSettings:
    """Rebuild from disk — used by tests that change the overlay."""
    global _SETTINGS
    _SETTINGS = OpsAgentSettings()  # type: ignore[call-arg]
    return _SETTINGS


__all__ = ["OpsAgentSettings", "get_agent_settings", "reload_agent_settings"]
