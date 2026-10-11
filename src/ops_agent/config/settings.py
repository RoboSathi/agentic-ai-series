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
    invalid_call: str
    query_heading: str
    rows_heading: str
    more_rows: str
    ungrounded: str
    sources_heading: str


class Stages(BaseModel):
    """The feed's stage key for each node's cards."""

    server: str


class Cards(BaseModel):
    """Card headings on the live feed."""

    result_rows: str
    result_metrics: str
    result_views: str
    result_columns: str
    result_done: str
    not_run: str
    failed: str
    refused: str
    grounded: str
    ungrounded: str
    figure_level: str
    figure_tally: str
    no_figures: str
    figure_found: str
    confidence_signal: str
    queries_searched: str


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
    table_rows: int = Field(ge=1)
    cell_chars: int = Field(ge=4)
    number_tolerance: float = Field(ge=0)
    free_numbers: list[float]
    # The id a prefetch read's call is shown to the model with (`{tool}`).
    prefetch_call_id: str
    words: Words
    stages: Stages
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
