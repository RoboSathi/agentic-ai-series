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
    description: str


class DeterministicRejectRule(BaseModel):
    """One explicit, high-confidence out-of-scope rejection rule."""

    rule_id: str
    pattern: str
    reason: str


class IntentRouterSettings(BaseModel):
    """Customer Support intent routing policy."""

    enabled: bool = True
    confidence_threshold: float
    clarification_top_k: int
    clarification_template: str
    deterministic_rejects: list[DeterministicRejectRule]
    model_label_mapping: dict[str, list[str]]


class PlanStepSettings(BaseModel):
    """One configured step of an intent's plan (E06)."""

    description: str
    tool: str = ""


class FollowupSettings(BaseModel):
    """How a message that restates nothing continues the previous subject."""

    phrases: list[str]
    reference_pattern: str


class CustomerSupportAgentSettings(YamlBackedSettings):
    """The running product's own wiring values (LLD §21)."""

    _yaml_section = "customer_support_agent"
    enabled: bool
    max_tool_iterations: int
    intents: list[IntentSettings]
    default_intent: str
    plans: dict[str, list[PlanStepSettings]]
    intent_router: IntentRouterSettings
    followup: FollowupSettings
    history_turns: int
    deliberation_markers: list[str]
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
