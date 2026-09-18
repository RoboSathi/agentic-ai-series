"""Configuration for every subsystem — the single source of truth for values.

Nothing is hardcoded anywhere else in the repo: every value here is read from
config.yaml (overridable by the gitignored local.yaml, and by SECTION__FIELD
env vars), never from a Python default. API keys are the deliberate
exception: they are read from bare lowercase env vars and never appear in a
settings dump (see LLMProviderSettings.api_key).
"""

import os
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_commons.config.paths import resolve_path
from agentic_commons.config.yaml_config import load_section

RedactionMode = Literal["mask", "tokenize", "remove"]
GuardrailMode = Literal["reject", "auto_repair", "constrained"]
RetrievalMode = Literal["lexical", "dense", "hybrid", "graph"]
ProviderName = Literal["nim", "local"]
SurfaceClassName = Literal["locked", "editable", "append_only", "human_controlled"]


class YamlBackedSettings(BaseSettings):
    """Base for settings sourced from config.yaml.

    Subclasses set `_yaml_section` to their (optionally dotted) path in
    config.yaml. Any field not given explicitly via env var or constructor
    kwarg is filled from that section; a field with no Python default and no
    matching YAML key fails fast rather than silently defaulting.
    """

    _yaml_section: ClassVar[str] = ""

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    @model_validator(mode="before")
    @classmethod
    def _fill_from_yaml(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return {**load_section(cls._yaml_section), **data}


class ConfigSettings(YamlBackedSettings):
    """Meta toggle for the config subsystem itself."""

    _yaml_section = "config"
    enabled: bool

    model_config = SettingsConfigDict(env_prefix="CONFIG__")


class LoggingSettings(YamlBackedSettings):
    """File and console logging (LLD §2)."""

    _yaml_section = "logging"
    enabled: bool
    level: str
    console_enabled: bool
    console_level: str
    file_enabled: bool
    file_level: str
    dir: str
    per_subsystem_color: bool
    max_bytes: int
    backup_count: int

    model_config = SettingsConfigDict(env_prefix="LOGGING__")

    @property
    def resolved_dir(self) -> Path:
        return resolve_path(self.dir)


class LLMProviderSettings(BaseModel):
    """One provider's connection and pricing parameters."""

    base_url: str
    model: str
    timeout_seconds: float
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float


class LLMCascadeSettings(BaseModel):
    """Model-cascading policy (E44) — cheapest sufficient tier first."""

    enabled: bool
    tiers: list[ProviderName]
    escalate_on_finish_reasons: list[str]
    escalate_on_empty_content: bool
    escalate_on_error: bool


class EmbeddingSettings(BaseModel):
    """Embedding backend used by the NOLLM cascade, RAG and memory scoring."""

    provider: ProviderName
    nim_model: str
    local_dimensions: int


class LLMSettings(YamlBackedSettings):
    """LLM adapters — the only place a provider SDK is used (LLD §3)."""

    _yaml_section = "llm"
    enabled: bool
    provider: ProviderName
    max_tokens: int
    log_payloads: bool
    payload_max_chars: int
    providers: dict[str, LLMProviderSettings]
    cascade: LLMCascadeSettings
    embedding: EmbeddingSettings

    model_config = SettingsConfigDict(env_prefix="LLM__")

    def provider_settings(self, name: ProviderName | None = None) -> LLMProviderSettings:
        """Connection settings for one provider (defaults to the active one)."""
        return self.providers[name or self.provider]

    def api_key_for(self, name: ProviderName) -> str | None:
        """Resolve a provider's key from its bare env var, never from config.

        Keys are deliberately not fields: they never appear in model_dump()
        and so can never leak into a log line or settings dump.
        """
        return os.environ.get("nvidia_api_key") if name == "nim" else None


class PersistenceSettings(YamlBackedSettings):
    """SQLite access (LLD §4)."""

    _yaml_section = "persistence"
    enabled: bool
    db_path: str

    model_config = SettingsConfigDict(env_prefix="PERSISTENCE__")

    @property
    def resolved_db_path(self) -> Path:
        return resolve_path(self.db_path)


class AuditSettings(YamlBackedSettings):
    """Append-only decision ledger (LLD §5)."""

    _yaml_section = "audit"
    enabled: bool
    policy_version: str
    data_version: str

    model_config = SettingsConfigDict(env_prefix="AUDIT__")


class ReviewQueueSettings(BaseModel):
    """Little's Law queue model inputs (E38)."""

    minutes_per_hour: float


class ObservabilitySettings(YamlBackedSettings):
    """Tracing, SLOs, cost stack, queue math (LLD §6)."""

    _yaml_section = "observability"
    enabled: bool
    langsmith_enabled: bool
    langsmith_project: str
    persist_traces: bool
    slo_targets: dict[str, float]
    review_queue: ReviewQueueSettings

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY__")


class ContextStrategySettings(BaseModel):
    """Per-strategy toggles for the six context-management strategies (E18)."""

    offload: bool
    reduce: bool
    retrieve: bool
    isolate: bool
    cache: bool
    consolidate: bool


class ContextFailureDetectionSettings(BaseModel):
    """Context-failure detection thresholds (E17)."""

    enabled: bool
    checkpoints_tokens: list[int]
    degradation_threshold: float


class ContextSettings(YamlBackedSettings):
    """Five-stage assembly pipeline, retrieval and strategies (LLD §7)."""

    _yaml_section = "context"
    enabled: bool
    token_budget: int
    chars_per_token: int
    retrieval_mode: RetrievalMode
    top_k: int
    hybrid_lexical_weight: float
    bm25_k1: float
    bm25_b: float
    strategies: ContextStrategySettings
    reduce_keep_ratio: float
    failure_detection: ContextFailureDetectionSettings

    model_config = SettingsConfigDict(env_prefix="CONTEXT__")


class MemoryStoreToggle(BaseModel):
    """Independent toggle for one CoALA store."""

    enabled: bool


class EpisodicMemorySettings(BaseModel):
    """Episodic store — append-only, one file per session (E14)."""

    enabled: bool
    path: str

    @property
    def resolved_path(self) -> Path:
        return resolve_path(self.path)


class SemanticMemorySettings(BaseModel):
    """Semantic store — structured facts with fuzzy key lookup."""

    enabled: bool
    fuzzy_match_threshold: float


class MemoryWritePipelineSettings(BaseModel):
    """Write-path routing thresholds (E15)."""

    enabled: bool
    auto_commit_threshold: float
    discard_threshold: float
    decay_if_unconfirmed_days: int
    authority_changing_categories: list[str]
    classification_hints: dict[str, list[str]]


class MemoryRetrievalWeights(BaseModel):
    """Composite retrieval scoring weights (Park et al. extended, E13)."""

    relevance: float
    recency: float
    importance: float
    authority: float
    temporal_validity: float


class MemorySettings(YamlBackedSettings):
    """All four CoALA stores plus the write pipeline (LLD §8)."""

    _yaml_section = "memory"
    enabled: bool
    working: MemoryStoreToggle
    episodic: EpisodicMemorySettings
    semantic: SemanticMemorySettings
    procedural: MemoryStoreToggle
    write_pipeline: MemoryWritePipelineSettings
    retrieval_weights: MemoryRetrievalWeights
    provenance_weights: dict[str, float]
    recency_half_life_days: float

    model_config = SettingsConfigDict(env_prefix="MEMORY__")


class GuardrailsSettings(YamlBackedSettings):
    """Structured-output guarantees (LLD §9)."""

    _yaml_section = "guardrails"
    enabled: bool
    mode: GuardrailMode
    max_repair_attempts: int
    constrained_backend: Literal["llamacpp_grammar", "nim_guided_json"]

    model_config = SettingsConfigDict(env_prefix="GUARDRAILS__")


class HallucinationProtectionSettings(YamlBackedSettings):
    """Detection ladder thresholds (LLD §10)."""

    _yaml_section = "hallucination_protection"
    enabled: bool
    confidence_threshold: float
    consistency_samples: int
    consistency_agreement_threshold: float
    grounding_overlap_threshold: float

    model_config = SettingsConfigDict(env_prefix="HALLUCINATION_PROTECTION__")


class LedgerSettings(BaseModel):
    """Cumulative-spend ledger that defeats structuring attacks (E02)."""

    enabled: bool
    window_hours: int
    action_classes: dict[str, str]
    thresholds_usd: dict[str, float]


class Tier2Settings(BaseModel):
    """Semantic advisor — advisory only, structurally unable to block (E12)."""

    enabled: bool


class SurfacePolicySettings(BaseModel):
    """Surface classification per action (E12)."""

    enabled: bool
    surfaces: dict[str, SurfaceClassName]


class AutomationSettings(BaseModel):
    """Sheridan automation level per action class (E03)."""

    enabled: bool
    human_approval_below: int
    levels: dict[str, int]
    per_action_amount_escalation_usd: dict[str, float]


class GatekeeperSettings(YamlBackedSettings):
    """The always-invoked mediation point (LLD §11).

    There is deliberately no top-level `enabled` flag: PRD G1 requires that
    no execution path bypasses the Gatekeeper. Its sub-features are each
    independently toggleable instead.
    """

    _yaml_section = "gatekeeper"
    authority_grant_max_rank: int
    authority_hierarchy: list[str]
    ledger: LedgerSettings
    tier2: Tier2Settings
    surface_policy: SurfacePolicySettings
    automation: AutomationSettings

    model_config = SettingsConfigDict(env_prefix="GATEKEEPER__")


class AuthSettings(BaseModel):
    """RBAC + just-in-time privilege elevation (E10)."""

    enabled: bool
    elevation_max_ttl_seconds: int
    roles: dict[str, list[str]]
    action_permissions: dict[str, str]


class PiiSettings(BaseModel):
    """PII detection/redaction/minimization (E32)."""

    enabled: bool
    mode: RedactionMode
    mask_token: str
    ner_enabled: bool
    patterns: dict[str, str]
    tool_call_allowed_fields: list[str]


class PromptInjectionSettings(BaseModel):
    """Instruction/data separation and tool scoping (E30)."""

    enabled: bool
    data_open_tag: str
    data_close_tag: str
    suspicious_phrases: list[str]


class ContentModerationSettings(BaseModel):
    """Tier-2 advisory content classifier."""

    enabled: bool
    blocked_terms: list[str]


class SecuritySettings(YamlBackedSettings):
    """RBAC, PII, prompt injection, moderation (LLD §12)."""

    _yaml_section = "security"
    enabled: bool
    auth: AuthSettings
    pii: PiiSettings
    prompt_injection: PromptInjectionSettings
    content_moderation: ContentModerationSettings

    model_config = SettingsConfigDict(env_prefix="SECURITY__")


class CircuitBreakerSettings(BaseModel):
    enabled: bool
    failure_threshold: int
    half_open_after_seconds: float


class RetrySettings(BaseModel):
    enabled: bool
    max_attempts: int
    base_delay_seconds: float
    max_delay_seconds: float


class TimeoutSettings(BaseModel):
    enabled: bool
    default_seconds: float


class BulkheadSettings(BaseModel):
    enabled: bool
    pool_size: int


class ResilienceSettings(YamlBackedSettings):
    """The seven Nygard patterns (LLD §13)."""

    _yaml_section = "resilience"
    enabled: bool
    circuit_breaker: CircuitBreakerSettings
    retry: RetrySettings
    timeout: TimeoutSettings
    bulkhead: BulkheadSettings
    idempotency: MemoryStoreToggle
    saga: MemoryStoreToggle

    model_config = SettingsConfigDict(env_prefix="RESILIENCE__")


class CascadeTierToggles(BaseModel):
    """Per-tier toggles for the NOLLM scoring cascade (E10)."""

    exact_assertion: bool
    deterministic_rule: bool
    embedding_similarity: bool
    nli_entailment: bool
    calibrated_llm_judge: bool
    human_adjudication: bool


class NliSettings(BaseModel):
    """Lexical entailment fallback used when no NLI endpoint is wired."""

    min_overlap: float
    negation_markers: list[str]


class JudgeSettings(BaseModel):
    """Sampling parameters for the calibrated LLM judge tier (E11)."""

    temperature: float
    max_tokens: int


class EvalsSettings(YamlBackedSettings):
    """Eval pyramid and NOLLM cascade (LLD §18)."""

    _yaml_section = "evals"
    enabled: bool
    manifest_path: str
    cascade: CascadeTierToggles
    embedding_similarity_threshold: float
    embedding_dissimilarity_threshold: float
    nli_entailment_threshold: float
    judge_confidence_threshold: float
    kappa_acceptable: float
    nli: NliSettings
    judge: JudgeSettings

    model_config = SettingsConfigDict(env_prefix="EVALS__")

    @property
    def resolved_manifest_path(self) -> Path:
        return resolve_path(self.manifest_path)


class ToolsSettings(YamlBackedSettings):
    """Tool registry and retrieval-based selection (LLD §19)."""

    _yaml_section = "tools"
    enabled: bool
    selection_k: int
    selection_min_score: float

    model_config = SettingsConfigDict(env_prefix="TOOLS__")


class TuiSettings(YamlBackedSettings):
    """Live status feed (LLD §20)."""

    _yaml_section = "tui"
    refresh_interval_ms: int
    show_token_count: bool
    copy_method: Literal["osc52", "file"]
    max_feed_lines: int
    spinner_frames: str
    stage_labels: dict[str, str]
    tool_labels: dict[str, str]
    stage_styles: dict[str, str]
    default_stage_style: str

    model_config = SettingsConfigDict(env_prefix="TUI__")


class IntentSettings(BaseModel):
    """One customer intent, and the tools it unlocks."""

    name: str
    phrases: list[str]
    tools: list[str]


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
    followup: FollowupSettings
    history_turns: int
    default_actor: str
    default_role: str
    default_authority_rank: int
    checkpoint_path: str
    policy_path: str

    model_config = SettingsConfigDict(env_prefix="CUSTOMER_SUPPORT_AGENT__")

    @property
    def resolved_checkpoint_path(self) -> Path:
        return resolve_path(self.checkpoint_path)

    @property
    def resolved_policy_path(self) -> Path:
        return resolve_path(self.policy_path)


class AppConfig(BaseSettings):
    """Root config — one settings model per bounded context (LLD §1)."""

    # Each nested model fills its required fields from config.yaml via
    # YamlBackedSettings; mypy can't see that, hence the ignores.
    config: ConfigSettings = ConfigSettings()  # type: ignore[call-arg]
    logging: LoggingSettings = LoggingSettings()  # type: ignore[call-arg]
    llm: LLMSettings = LLMSettings()  # type: ignore[call-arg]
    persistence: PersistenceSettings = PersistenceSettings()  # type: ignore[call-arg]
    audit: AuditSettings = AuditSettings()  # type: ignore[call-arg]
    observability: ObservabilitySettings = ObservabilitySettings()  # type: ignore[call-arg]
    context: ContextSettings = ContextSettings()  # type: ignore[call-arg]
    memory: MemorySettings = MemorySettings()  # type: ignore[call-arg]
    guardrails: GuardrailsSettings = GuardrailsSettings()  # type: ignore[call-arg]
    hallucination_protection: HallucinationProtectionSettings = (
        HallucinationProtectionSettings()  # type: ignore[call-arg]
    )
    gatekeeper: GatekeeperSettings = GatekeeperSettings()  # type: ignore[call-arg]
    security: SecuritySettings = SecuritySettings()  # type: ignore[call-arg]
    resilience: ResilienceSettings = ResilienceSettings()  # type: ignore[call-arg]
    evals: EvalsSettings = EvalsSettings()  # type: ignore[call-arg]
    tools: ToolsSettings = ToolsSettings()  # type: ignore[call-arg]
    tui: TuiSettings = TuiSettings()  # type: ignore[call-arg]
    customer_support_agent: CustomerSupportAgentSettings = (
        CustomerSupportAgentSettings()  # type: ignore[call-arg]
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


_config_instance: AppConfig | None = None


def get_settings() -> AppConfig:
    """Get the mutable config singleton, read live on every access.

    Deliberately a singleton (the one place in the repo that uses the
    pattern): the TUI and tests flip feature toggles on this object mid-run
    and every governance checkpoint must observe the change without a
    restart.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance


def reload_settings() -> AppConfig:
    """Drop the cached singleton and re-read config.yaml from disk."""
    global _config_instance
    _config_instance = None
    return get_settings()
