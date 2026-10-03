"""Configuration for the customer support agent.

These settings live with the agent, not with the platform. `AppConfig`
used to own them, which meant `agent_platform` could not load without
this agent's config file present — a shared library that depends on one
of its consumers.

The values still come from the same merged YAML: the agent's
`config/config.yaml` is an overlay on the platform's, so a machine-local
`local.yaml` override still reaches these too.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from agent_platform.config import get_settings
from agent_platform.config.paths import resolve_path
from agent_platform.config.settings import YamlBackedSettings


class IntentSettings(BaseModel):
    """One customer intent, its capability class, and the tools it unlocks."""

    name: str
    phrases: list[str]
    tools: list[str]
    description: str
    mode: Literal["read", "write"] = "read"


class DeterministicRejectRule(BaseModel):
    """One explicit, high-confidence out-of-scope rejection rule."""

    rule_id: str
    pattern: str
    reason: str


class IntentRouterSettings(BaseModel):
    """Customer Support intent routing policy."""

    confidence_threshold: float
    clarification_top_k: int
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


class CsaEvalSettings(BaseModel):
    """Where this agent's evals keep their dataset, fixture and graders' inputs."""

    dataset_path: str
    fixture_seed_path: str
    # The fixture is built once into this template; each case runs on a copy.
    fixture_db_path: str
    workspace_dir: str
    judge_rubric_path: str
    quality_metrics_path: str
    security_config_path: str
    session_prefix: str
    # Message roles the model was shown that a reply may be grounded on.
    grounding_roles: list[str]
    # Keep each case's database copy after the run, for debugging a failure.
    keep_case_workspaces: bool
    # How the oracle finds order references and money in a reply.
    order_reference_pattern: str
    money_pattern: str
    # A reply sharing a run of this many words with the system instructions
    # is treated as leaking them (the refund policy itself may be quoted).
    prompt_leak_ngram: int

    @property
    def resolved_dataset_path(self) -> Path:
        return resolve_path(self.dataset_path)

    @property
    def resolved_fixture_seed_path(self) -> Path:
        return resolve_path(self.fixture_seed_path)

    @property
    def resolved_fixture_db_path(self) -> Path:
        return resolve_path(self.fixture_db_path)

    @property
    def resolved_workspace_dir(self) -> Path:
        return resolve_path(self.workspace_dir)


class OrderReferenceSettings(BaseModel):
    """How a customer's words are read as naming an order."""

    # A number tied to the word "order", however loosely the customer wrote
    # it ("order: N", "this order -> N", "the order is N"). Group `id`.
    anchored_patterns: list[str]
    # The compact request form ("refund N"). Group `id`.
    compact_patterns: list[str]
    # What an id looks like when nothing anchors it. Such a number counts only
    # if it is one of the customer's own orders: shape alone proves nothing.
    id_shape: str


class AuthMessages(BaseModel):
    """What a customer reads while proving who they are (`{length}` = code length)."""

    ask_for_code: str
    unknown_code: str


class CsaMessages(BaseModel):
    """Fixed wording this agent says itself, rather than a model."""

    auth: AuthMessages
    # A turn the operator cancelled (`{step}` = the node it resumes at).
    cancelled: str


class OrderTableSettings(BaseModel):
    """How a customer's orders read: the table's columns and the words for each state."""

    headers: list[str] = Field(min_length=7, max_length=7)
    # How much of a product name one row shows before it is elided.
    item_width: int = Field(ge=1)
    ellipsis: str
    status_words: dict[str, str]
    refund_words: dict[str, str]
    # What the refund column says when none was requested (list, and one order's detail).
    no_refund: str
    no_refund_detail: str


class ConsoleActionSettings(BaseModel):
    """One console button: its label and Textual variant (`{node}` = where a resume starts)."""

    label: str
    variant: str = "default"


class ConsoleSettings(BaseModel):
    """This agent's operator-console controls."""

    # The key that cancels the turn in flight.
    cancel_key: str
    cancel: ConsoleActionSettings
    resume: ConsoleActionSettings


class CustomerSupportAgentSettings(YamlBackedSettings):
    """The running product's own wiring values (LLD §21)."""

    _yaml_section = "customer_support_agent"
    max_tool_iterations: int
    # The most orders one `list_orders` call returns, newest first.
    list_orders_max_rows: int
    intents: list[IntentSettings]
    default_intent: str
    # Which tool is charged against which budget class. The platform stores
    # budgets by class and does not know this agent's verbs, so the mapping
    # is declared here rather than as a literal in the composition root.
    action_classes: dict[str, str]
    plans: dict[str, list[PlanStepSettings]]
    intent_router: IntentRouterSettings
    order_reference: OrderReferenceSettings
    followup: FollowupSettings
    history_turns: int
    deliberation_markers: list[str]
    user_code_length: int
    user_code_alphabet: str
    user_code_placeholder: str
    messages: CsaMessages
    console: ConsoleSettings
    order_table: OrderTableSettings
    demo_user_code: str
    policy_path: str
    grounding_facts: dict[str, str]
    # Who the console acts as when it approves or denies a queued refund.
    console_reviewer: str
    evals: CsaEvalSettings
    # What `make dummy-data` generates (counts, seed, catalogue, weights).
    dummy_data_generator_path: str

    model_config = SettingsConfigDict(env_prefix="CUSTOMER_SUPPORT_AGENT__")

    @property
    def resolved_policy_path(self) -> Path:
        return resolve_path(self.policy_path)

    @property
    def resolved_dummy_data_generator_path(self) -> Path:
        return resolve_path(self.dummy_data_generator_path)

    @property
    def resolved_db_path(self) -> Path:
        """SQLite database owned by this agent deployment (`persistence.db_path`)."""
        return get_settings().persistence.resolved_db_path


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
