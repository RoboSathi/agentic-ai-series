"""Configuration for the customer support agent.

These settings live with the agent, not with the platform. `AppConfig`
used to own them, which meant `agentic_commons` could not load without
this agent's config file present — a shared library that depends on one
of its consumers.

The values still come from the same merged YAML: the agent's
`config/config.yaml` is an overlay on the platform's, so a machine-local
`local.yaml` override still reaches these too.
"""

from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

from agentic_commons.config.paths import resolve_path
from agentic_commons.config.settings import YamlBackedSettings


class IntentSettings(BaseModel):
    """One customer intent, and the tools it unlocks."""

    name: str
    phrases: list[str]
    tools: list[str]
    # Whether this intent may also be reached by similarity when no
    # phrase matched. Off by default: an intent that moves money is a
    # policy decision, and a policy stated as a threshold is a policy
    # nobody can read.
    semantic: bool = False


class SemanticIntentSettings(BaseModel):
    """The encoder fallback used when no phrase matched."""

    enabled: bool
    threshold: float
    max_words: int


class PlanStepSettings(BaseModel):
    """One configured step of an intent's plan (E06)."""

    description: str
    tool: str = ""


class FollowupSettings(BaseModel):
    """How a message that restates nothing continues the previous subject."""

    phrases: list[str]
    reference_pattern: str


class ScopeSettings(BaseModel):
    """What the agent will talk about — default-deny topic gate."""

    in_scope_topics: dict[str, list[str]]
    min_words_before_refusal: int
    reference_pattern: str
    refusal_message: str


class CustomerSupportAgentSettings(YamlBackedSettings):
    """The running product's own wiring values (LLD §21)."""

    _yaml_section = "customer_support_agent"
    enabled: bool
    max_tool_iterations: int
    intents: list[IntentSettings]
    default_intent: str
    plans: dict[str, list[PlanStepSettings]]
    semantic_intent: SemanticIntentSettings
    followup: FollowupSettings
    history_turns: int
    deliberation_markers: list[str]
    scope: ScopeSettings
    user_code_length: int
    user_code_alphabet: str
    demo_user_code: str
    checkpoint_path: str
    policy_path: str
    db_path: str

    model_config = SettingsConfigDict(env_prefix="CUSTOMER_SUPPORT_AGENT__")

    @property
    def resolved_checkpoint_path(self) -> Path:
        return resolve_path(self.checkpoint_path)

    @property
    def resolved_policy_path(self) -> Path:
        return resolve_path(self.policy_path)

    @property
    def resolved_db_path(self) -> Path:
        """SQLite database owned by this agent deployment."""
        return resolve_path(self.db_path)


_SETTINGS: CustomerSupportAgentSettings | None = None


def get_agent_settings() -> CustomerSupportAgentSettings:
    """The agent's settings, built once per process."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = CustomerSupportAgentSettings()  # type: ignore[call-arg]
    return _SETTINGS


def reload_agent_settings() -> CustomerSupportAgentSettings:
    """Rebuild from disk — used by tests that mutate configuration."""
    global _SETTINGS
    _SETTINGS = CustomerSupportAgentSettings()  # type: ignore[call-arg]
    return _SETTINGS
