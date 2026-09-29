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

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_commons.config.features import FeatureSet
from agentic_commons.config.paths import resolve_path
from agentic_commons.config.yaml_config import deep_merge, load_section

RedactionMode = Literal["mask", "tokenize", "remove"]
GuardrailMode = Literal["reject", "auto_repair", "constrained"]
RetrievalMode = Literal["lexical", "dense", "hybrid", "graph"]
ProvenanceKindName = Literal["verified_system_of_record", "agent_inference", "self_reported"]
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
        # Deep, not shallow: `RESILIENCE__RETRY__MAX_ATTEMPTS=7` arrives as
        # {"retry": {"max_attempts": "7"}}, and a shallow merge would drop
        # every sibling of `max_attempts` that YAML supplied.
        return deep_merge(load_section(cls._yaml_section), data)


class AgentIdentitySettings(YamlBackedSettings):
    """Identity stamped onto shared platform observability records."""

    _yaml_section = "agent"

    id: str
    version: str
    # `module.path:attribute` naming this agent's `AgentHost`. Empty in the
    # platform defaults on purpose: the platform must not name an agent, so
    # an agent declares its own host in its own overlay.
    host: str = ""

    model_config = SettingsConfigDict(env_prefix="AGENT__")

class LoggingSettings(YamlBackedSettings):
    """File and console logging (LLD §2)."""

    _yaml_section = "logging"
    level: str
    console_enabled: bool
    console_level: str
    file_enabled: bool
    file_level: str
    dir: str
    file_format: Literal["tree", "json"]
    payload_max_chars: int
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


class EmbeddingInputTypes(BaseModel):
    """What an asymmetric embedding model is told each text is.

    Retrieval models embed a short question and a long passage
    differently; embedding a passage as if it were a query costs recall.
    """

    query: str
    passage: str


class EmbeddingSettings(BaseModel):
    """Embedding backend used by the NOLLM cascade, RAG and memory scoring."""

    provider: ProviderName
    nim_model: str
    local_dimensions: int
    input_types: EmbeddingInputTypes
    truncate: str


class EncoderSettings(BaseModel):
    """BERT-family encoders for entailment and reranking (LLD §3).

    NLI runs locally or hosted (`nli_provider`); the reranker is hosted.
    """

    enabled: bool
    base_url: str
    nli_provider: Literal["huggingface_local", "huggingface_hosted"]
    nli_model: str
    nli_local_model_dir: str
    nli_max_length: int
    download_if_missing: bool
    reranker_model: str
    timeout_seconds: float
    wait_for_model: bool

    @property
    def resolved_nli_local_model_dir(self) -> Path:
        return resolve_path(self.nli_local_model_dir)

    @property
    def api_key(self) -> str:
        """The Hugging Face token, from its bare environment variable.

        A property rather than a field, like every other credential here:
        it never appears in `model_dump()`, so it cannot reach a log line
        or a settings dump.
        """
        return os.environ.get("hf_api_key", "")

    @property
    def usable(self) -> bool:
        """Whether a hosted encoder can be called at all."""
        return self.enabled and bool(self.api_key)




class IntentClassifierSettings(BaseModel):
    """Reusable model/provider settings for semantic intent classification.

    Agent-specific labels and the confidence threshold stay with the
    consuming agent. This section only describes how the shared platform
    obtains and runs the classifier.
    """

    enabled: bool
    provider: Literal["huggingface_local"]
    model: str
    local_model_dir: str
    download_if_missing: bool
    max_length: int

    @property
    def resolved_local_model_dir(self) -> Path:
        return resolve_path(self.local_model_dir)


class DialogueSummarizerSettings(BaseModel):
    """A local seq2seq model that summarises a conversation (memory dreams).

    Local on purpose: a dream summarises every session it reads, and a
    remote LLM call per session is exactly the rate-limited cost this
    avoids. Its output is never trusted as stated — the dream keeps a
    sentence only when the NLI checker confirms it against the events.
    """

    enabled: bool
    model: str
    local_model_dir: str
    download_if_missing: bool
    max_input_tokens: int
    max_summary_tokens: int
    num_beams: int

    @property
    def resolved_local_model_dir(self) -> Path:
        return resolve_path(self.local_model_dir)


class LLMCheckSettings(BaseModel):
    """The smallest real request `make llm-check` sends to each endpoint."""

    prompt: str
    max_tokens: int


class OutputNormalisationSettings(BaseModel):
    """Typography undone in model output before anything reads it."""

    enabled: bool
    space_categories: list[str]
    replacements: dict[str, str]
    remove: list[str]


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
    encoders: EncoderSettings
    intent_classifier: IntentClassifierSettings
    dialogue_summarizer: DialogueSummarizerSettings
    check: LLMCheckSettings
    output_normalisation: OutputNormalisationSettings

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
    db_path: str

    model_config = SettingsConfigDict(env_prefix="PERSISTENCE__")

    @property
    def resolved_db_path(self) -> Path:
        return resolve_path(self.db_path)


DurabilityMode = Literal["sync", "async", "exit"]


class CheckpointEncryptionSettings(BaseModel):
    """Encryption at rest for checkpoints, which hold the conversation."""

    # The *name* of the environment variable holding the AES key — the
    # key itself never appears in configuration.
    key_env: str
    # Refuse to start without a key. Off for development, on in any
    # deployment whose checkpoints hold real customers' words.
    required: bool


class DurabilitySettings(YamlBackedSettings):
    """Durable execution: checkpoints, resumable runs (LLD §13, E35)."""

    _yaml_section = "durability"
    db_path: str
    # LangGraph's durability mode. `sync` writes each checkpoint before
    # the next node starts, so a crash loses at most the node in flight.
    mode: DurabilityMode
    retention_days: int
    # How much of a run's query the run registry keeps for display.
    preview_chars: int
    encryption: CheckpointEncryptionSettings

    model_config = SettingsConfigDict(env_prefix="DURABILITY__")

    @property
    def resolved_db_path(self) -> Path:
        return resolve_path(self.db_path)


class MoneySettings(YamlBackedSettings):
    """How amounts are held and shown (see `agentic_commons.money`)."""

    _yaml_section = "money"
    currency: str
    # Decimal places of the minor unit: 2 for cents.
    minor_units: int
    symbol: str

    model_config = SettingsConfigDict(env_prefix="MONEY__")


class AuditSettings(YamlBackedSettings):
    """Append-only decision ledger (LLD §5)."""

    _yaml_section = "audit"
    enabled: bool
    policy_version: str
    data_version: str
    prompt_version: str
    tool_schema_version: str

    model_config = SettingsConfigDict(env_prefix="AUDIT__")


class ReviewQueueSettings(BaseModel):
    """Little's Law queue model inputs (E38)."""

    minutes_per_hour: float
    minutes_per_review: float


class ObservabilitySettings(YamlBackedSettings):
    """Tracing, SLOs, cost stack, queue math (LLD §6)."""

    _yaml_section = "observability"
    enabled: bool
    analytics_db_path: str
    langsmith_enabled: bool
    persist_traces: bool
    slo_targets: dict[str, float]
    review_queue: ReviewQueueSettings
    # How a card in the tool-use loop names its iteration. `{step}` is the
    # model call being made, `{max}` the cap on tool iterations.
    loop_step_prefix: str
    loop_step_detail: str

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY__")


class RerankSettings(BaseModel):
    """Cross-encoder reranking of the retrieved shortlist (LLD §7)."""

    shortlist: int
    min_candidates: int


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

    checkpoints_tokens: list[int]
    degradation_threshold: float


class ContextSettings(YamlBackedSettings):
    """Five-stage assembly pipeline, retrieval and strategies (LLD §7)."""

    _yaml_section = "context"
    token_budget: int
    chars_per_token: int
    retrieval_mode: RetrievalMode
    top_k: int
    hybrid_lexical_weight: float
    rerank: RerankSettings
    bm25_k1: float
    bm25_b: float
    strategies: ContextStrategySettings
    reduce_keep_ratio: float
    failure_detection: ContextFailureDetectionSettings

    model_config = SettingsConfigDict(env_prefix="CONTEXT__")


class EnabledToggle(BaseModel):
    """A section whose only setting is whether it runs."""

    enabled: bool


class EpisodicMemorySettings(BaseModel):
    """Episodic store — append-only, one file per session (E14)."""

    path: str

    @property
    def resolved_path(self) -> Path:
        return resolve_path(self.path)


class RagChunkingSettings(BaseModel):
    """Semantic chunking — headings first, then embedding breakpoints."""

    # A sentence boundary becomes a passage boundary when the cosine
    # distance across it is above this percentile of the section's gaps.
    breakpoint_percentile: float
    min_tokens: int
    max_tokens: int


class RagRetrievalSettings(BaseModel):
    """Hybrid retrieval: sparse + dense legs, RRF fusion, cross-encoder rerank."""

    sparse_k: int
    dense_k: int
    # Cormack et al.'s constant: damps how much a first place outweighs a
    # fifth, so one leg's confident outlier cannot swamp the other.
    rrf_k: int
    rerank_shortlist: int
    top_k: int


class RagSettings(BaseModel):
    """One agent's knowledge index: where it lives and what feeds it."""

    db_path: str
    corpus_dir: str
    corpus_glob: str
    ingest_on_startup: bool
    embed_batch_size: int
    chunking: RagChunkingSettings
    retrieval: RagRetrievalSettings

    @property
    def resolved_db_path(self) -> Path:
        return resolve_path(self.db_path)

    @property
    def resolved_corpus_dir(self) -> Path:
        return resolve_path(self.corpus_dir)


class SemanticMemorySettings(BaseModel):
    """Semantic store — structured facts, plus document RAG over a corpus."""

    fuzzy_match_threshold: float
    rag: RagSettings


class MemoryWritePipelineSettings(BaseModel):
    """Write-path routing thresholds (E15)."""

    enabled: bool
    interrogative_markers: list[str]
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
    # How often a fact has been recalled — reinforcement, not popularity:
    # a fact the agent keeps needing is one it should keep finding.
    frequency: float


MemoryPrincipalRole = Literal[
    "customer", "agent", "system_of_record", "curator", "data_protection"
]


class MemoryGovernanceSettings(BaseModel):
    """Who may do what to memory, and how much what they say is believed."""

    # Provenance is derived from *who* is speaking, never passed in.
    provenance_by_role: dict[MemoryPrincipalRole, ProvenanceKindName]
    # Platform-default grants per role, as `memory:<class>:<op>:<category>`
    # with `*` wildcards. The agent principal also holds whatever its
    # persisted AgentPolicy grants.
    role_permissions: dict[MemoryPrincipalRole, list[str]]


class MemoryExtractorSettings(BaseModel):
    """One deterministic claim pattern: named groups `key` and/or `value`."""

    name: str
    pattern: str
    # A fixed key, for patterns that capture only a value ("I prefer …").
    key: str = ""


class TextSearchSettings(BaseModel):
    """Grep over file-backed memory (episodic logs, skills)."""

    backend: Literal["ripgrep", "python"]
    binary: str
    timeout_seconds: float


class MemoryDeterministicSettings(BaseModel):
    """The deterministic tier that runs before any model or embedding."""

    extractors: list[MemoryExtractorSettings]
    # Shapes of identifiers in a query; any match is looked up exactly (R1).
    identifier_patterns: list[str]
    # Text that must never become memory, whoever says it (poisoning).
    reject_patterns: list[str]
    text_search: TextSearchSettings
    # R1 answers on its own once it finds at least this many facts.
    min_exact_hits: int
    # The dense tier (R3) runs only when R1 + R2 found fewer than this.
    dense_below: int


class MemoryImportanceSettings(BaseModel):
    """Importance assigned at write time, by category (deterministic)."""

    default: float
    by_category: dict[str, float]


class DreamSettings(BaseModel):
    """Offline consolidation of the dream inbox (`make dream`)."""

    # Consolidated turn events older than this are pruned by each dream.
    retention_days: int = Field(gt=0)
    # How many identical refusals across sessions make a procedure candidate.
    procedure_threshold: int = Field(gt=0)
    # Add a locally generated intent sentence to each session summary.
    summarize_intent: bool
    # The principal a dream acts as. An agent role, so a proposed procedure
    # is refused by governance and lands with a curator.
    actor: str
    # How much of a tool result one outcome line quotes.
    outcome_chars: int = Field(gt=0)
    # Prefixes that mark a tool result as a refusal.
    refusal_markers: list[str]
    # A procedure key is a name, not a sentence: longer keys are cut here.
    procedure_key_chars: int = Field(gt=0)
    # Words a generated summary sentence may not contain (gendered pronouns:
    # the summarizer cannot know them). Such a sentence is dropped.
    forbidden_words: list[str]
    # Share of real words below which a generated sentence is decoding noise.
    min_word_ratio: float = Field(ge=0.0, le=1.0)
    # Read-only tools, named by the agent. A summary condenses their calls
    # to "looked up …" and quotes only what changed something.
    lookup_tools: list[str] = Field(default_factory=list)


class MemorySettings(YamlBackedSettings):
    """All four CoALA stores plus the write pipeline (LLD §8)."""

    _yaml_section = "memory"
    schema_version: str
    episodic: EpisodicMemorySettings
    semantic: SemanticMemorySettings
    write_pipeline: MemoryWritePipelineSettings
    retrieval_weights: MemoryRetrievalWeights
    provenance_weights: dict[str, float]
    recency_half_life_days: float
    governance: MemoryGovernanceSettings
    deterministic: MemoryDeterministicSettings
    importance: MemoryImportanceSettings
    dream: DreamSettings

    model_config = SettingsConfigDict(env_prefix="MEMORY__")


class GuardrailsSettings(YamlBackedSettings):
    """Output/schema guardrail settings."""

    _yaml_section = "guardrails"
    enabled: bool
    mode: Literal["reject", "auto_repair", "constrained"]
    max_repair_attempts: int
    constrained_backend: Literal["llamacpp_grammar", "nim_guided_json"]

    model_config = SettingsConfigDict(env_prefix="GUARDRAILS__")


class GroundingTierSettings(BaseModel):
    """One stage of the grounding cascade."""

    name: str
    kinds: list[str]
    claims: list[str]
    may_refute: bool


class RulePhraseSettings(BaseModel):
    """A surface phrase in an outcome, and the rule it restates."""

    pattern: str
    statement: str


class StatePhraseSettings(BaseModel):
    """A way of saying where a refund stands, and the status it takes to be true."""

    pattern: str
    reaches: str
    # True only while the refund is exactly there ("pending review"), rather
    # than once it has got there ("submitted").
    exact: bool = False


class RefundStatesSettings(BaseModel):
    """The refund state machine a state claim is judged against."""

    ladder: list[str]
    branches: dict[str, str]
    settled_from: str
    observed_as: dict[str, list[str]]
    context_pattern: str
    phrases: list[StatePhraseSettings]


class HallucinationProtectionSettings(YamlBackedSettings):
    """Detection ladder thresholds (LLD)."""

    _yaml_section = "hallucination_protection"
    enforce_grounding: bool
    confidence_threshold: float
    # Grounding-specific NLI threshold. Keep the generic confidence threshold
    # for compatibility with the rest of the evaluation configuration.
    nli_entailment_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    consistency_enabled: bool
    consistency_samples: int
    consistency_agreement_threshold: float
    grounding_overlap_threshold: float
    # Number of evidence chunks retrieved for each claim before NLI verification.
    # This is a retrieval parameter, not a claim count. Keep it small for the
    # support agent; the same primitive can later be reused by larger RAG corpora.
    embedding_top_k: int = Field(default=3, ge=1, le=20)
    # Retrieved evidence below this similarity is shown but never sent to
    # NLI: text unrelated to a claim can neither support nor refute it.
    evidence_min_similarity: float
    # What an identifier looks like to this agent. When set, evidence may
    # refute a claim that names identifiers only if it names one of them:
    # NLI reads a record about order B as contradicting a claim about
    # order A. Unset: every retrieved item may refute.
    refute_identifier_pattern: str | None = None
    # The grounding cascade, strongest first. Each tier names the evidence
    # kinds it reads, the claim classes it may ground, and whether it may
    # refute as well as support. Order is the order credit is given in.
    tiers: list[GroundingTierSettings]
    # How a claim is told apart from conversation and from a promise.
    # Business vocabulary, so an agent overrides it; the platform ships the
    # shape and one worked default.
    conversational_openers: list[str]
    record_identifier_pattern: str
    outcome_pattern: str
    promissory_pattern: str
    # Settled-refund wording, the record field that settles it, and the
    # record wording that says it is still waiting on a person.
    refusal_pattern: str
    refund_state_pattern: str
    refund_states: RefundStatesSettings
    # A phrase carved out of an outcome and checked as the rule it states.
    rule_phrases: list[RulePhraseSettings]
    # A rule claim whose content is this fraction restated by one rule of
    # the tier is supported, whatever the NLI model made of an instruction.
    rule_restatement_coverage: float = Field(default=0.7, ge=0.0, le=1.0)
    # Hallucination repair is deliberately bounded. A model that repeats an
    # unsupported claim must not be given an unbounded generative retry loop.
    max_repair_attempts: int = Field(default=1, ge=0, le=3)

    model_config = SettingsConfigDict(env_prefix="HALLUCINATION_PROTECTION__")


class LedgerSettings(BaseModel):
    """Cumulative-spend ledger that defeats structuring attacks (E02)."""

    enabled: bool
    window_hours: int
    action_classes: dict[str, str]
    # Cumulative caps per budget class, in minor units (cents).
    thresholds_cents: dict[str, int]


class MandateSettings(BaseModel):
    """Binding a proposed write to what was actually requested (LLD §11.1)."""

    # Which call parameter names the thing being acted on. The platform
    # compares targets; it does not know they are called orders.
    target_param: str


class Tier2Settings(BaseModel):
    """Semantic advisor — advisory only, structurally unable to block (E12).

    The prompt is configuration because it has to name the kind of agent
    being reviewed — "unusual for a *what*?" is the whole question — and
    the platform does not know what kind of agent it is running.
    """

    enabled: bool
    prompt: str


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
    mandate: MandateSettings
    surface_policy: SurfacePolicySettings
    automation: AutomationSettings

    model_config = SettingsConfigDict(env_prefix="GATEKEEPER__")


class AuthSettings(BaseModel):
    """RBAC + just-in-time privilege elevation (E10)."""

    enabled: bool


class PiiSettings(BaseModel):
    """PII detection/redaction/minimization (E32)."""

    mode: RedactionMode
    mask_token: str
    ner_enabled: bool
    patterns: dict[str, str]
    tool_call_allowed_fields: list[str]


class SecurityGuardModelSettings(BaseModel):
    """Hosted security-model endpoint settings; credentials stay in env vars."""

    enabled: bool
    provider: Literal["nvidia"]
    base_url: str
    endpoint: str
    model: str
    timeout_seconds: float
    max_tokens: int = 16

    @property
    def api_key(self) -> str:
        """Return the existing NVIDIA credential from the bare environment."""
        return os.environ.get("nvidia_api_key") or os.environ.get("NVIDIA_API_KEY", "")

    @property
    def usable(self) -> bool:
        """Whether this hosted guard can be invoked."""
        return self.enabled and bool(self.api_key)


class SecurityGuardSettings(BaseModel):
    """Input/output guard-model configuration."""

    enabled: bool
    fail_open_on_unavailable: bool
    jailbreak: SecurityGuardModelSettings
    content_safety: SecurityGuardModelSettings


class PromptInjectionSettings(BaseModel):
    """Instruction/data separation and tool scoping (E30)."""

    data_open_tag: str
    data_close_tag: str
    suspicious_phrases: list[str]


class ContentModerationSettings(BaseModel):
    """Tier-2 advisory content classifier.

    `system_prompt` describes the application being protected, which is
    the agent's to say: "ordinary requests are safe" means something
    different for a support desk than for a medical triage line.
    """

    blocked_terms: list[str]
    system_prompt: str


class SecuritySettings(YamlBackedSettings):
    """RBAC, PII, prompt injection, moderation (LLD §12)."""

    _yaml_section = "security"
    auth: AuthSettings
    pii: PiiSettings
    prompt_injection: PromptInjectionSettings
    content_moderation: ContentModerationSettings
    guards: SecurityGuardSettings

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


class DeadLetterQueueSettings(BaseModel):
    """Durable quarantine for requests blocked before model execution."""

    enabled: bool


class RateLimitBucketSettings(BaseModel):
    """One shared request budget, and the hosts that draw on it."""

    max_requests: int
    hosts: list[str]


class RateLimitSettings(BaseModel):
    """Sliding-window rate limiting for hosted model endpoints.

    The window is kept in a SQLite file rather than in memory because the
    budget belongs to the provider account, not to one process: the TUI, an
    eval run and Promptfoo's provider processes all spend the same quota.
    """

    enabled: bool
    store_path: str
    window_seconds: float
    max_wait_seconds: float
    lock_timeout_seconds: float
    default_retry_after_seconds: float
    buckets: dict[str, RateLimitBucketSettings]

    @property
    def resolved_store_path(self) -> Path:
        return resolve_path(self.store_path)


class ResilienceSettings(YamlBackedSettings):
    """Resilience controls, including durable dead-letter quarantine."""

    _yaml_section = "resilience"
    circuit_breaker: CircuitBreakerSettings
    retry: RetrySettings
    timeout: TimeoutSettings
    bulkhead: BulkheadSettings
    idempotency: EnabledToggle
    saga: EnabledToggle
    dead_letter_queue: DeadLetterQueueSettings
    rate_limit: RateLimitSettings

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
    """The LLM judge: which model grades, and how it samples (E11).

    The judge's model is its own setting, deliberately not the agent's: a
    model grading its own output is the self-preference bias a judge
    exists to avoid.
    """

    provider: ProviderName
    model: str
    temperature: float
    max_tokens: int
    # Cap on the evidence text a rubric judge is shown (0 = all of it).
    max_evidence_chars: int


class EvalStoreSettings(BaseModel):
    """Where eval runs, transcripts and scores are kept (DuckDB)."""

    db_path: str
    # DuckDB admits one process per file; a reader that meets a running
    # writer retries instead of failing.
    lock_retry_attempts: int
    lock_retry_delay_seconds: float

    @property
    def resolved_db_path(self) -> Path:
        return resolve_path(self.db_path)


class EvalGenerationSettings(BaseModel):
    """How the agent is driven over a suite."""

    max_concurrency: int


class EvalFingerprintSettings(BaseModel):
    """What counts as "the agent changed" — hashed from disk, not git."""

    source_roots: list[str]
    include_suffixes: list[str]
    # Directory names skipped wherever they appear.
    exclude_parts: list[str]
    # Paths (relative to the repo) whose edits cannot change what the agent
    # does — the eval machinery itself — so editing a report never forces
    # the agent to be re-run.
    exclude_paths: list[str]
    # Config sections that cannot change agent behaviour.
    exclude_config_sections: list[str]


class EvalGateSettings(BaseModel):
    """When a run fails, and when the pipeline stops."""

    stop_on_invariant_failure: bool
    soft_pass_rate: float


class EvalReportingSettings(BaseModel):
    """The Markdown reports every stage writes."""

    output_dir: str
    confidence_z: float
    max_cases_listed: int
    # Transcript metadata key holding the runtime hallucination guard's
    # rate, compared against the judge's verdicts in the judge report.
    runtime_guard_key: str
    # Report file per stage (plus `summary` and `diff`), inside a run's folder.
    files: dict[str, str]

    @property
    def resolved_output_dir(self) -> Path:
        return resolve_path(self.output_dir)


class EvalDeepEvalSettings(BaseModel):
    """DeepEval, run as an optional quality stage through the harness's own client."""

    # Reasons cost an extra model call per metric; off keeps the stage cheap.
    include_reason: bool
    telemetry_opt_out: bool


class EvalPromptfooSettings(BaseModel):
    """Promptfoo, run as a Node tool through npx — never a Python dependency."""

    runner: list[str]
    package: str
    suite: str
    max_concurrency: int
    # How long one test may run in a Promptfoo Python worker. A multi-turn
    # attack is several full agent turns, each behind the shared rate limit.
    provider_timeout_ms: int
    provider_scheme: str
    output_dir: str
    extra_args: list[str]
    # Environment the Promptfoo process (and its Python workers) gets.
    env: dict[str, str]

    @property
    def resolved_output_dir(self) -> Path:
        return resolve_path(self.output_dir)


class EvalsSettings(YamlBackedSettings):
    """Eval pyramid and NOLLM cascade (LLD §18)."""

    _yaml_section = "evals"
    enabled: bool
    manifest_version: str
    manifest_path: str
    # Further manifests merged over the platform's, in order. Empty here:
    # a golden-task eval is necessarily about some product's behaviour, so
    # an agent declares its own rather than adding to everyone's.
    additional_manifest_paths: list[str] = []
    cascade: CascadeTierToggles
    embedding_similarity_threshold: float
    embedding_dissimilarity_threshold: float
    nli_entailment_threshold: float
    judge_confidence_threshold: float
    kappa_acceptable: float
    nli: NliSettings
    judge: JudgeSettings
    # `module.path:attribute` of the running agent's EvalPlugin. Empty
    # here: the platform names no agent.
    plugin: str
    suites: list[str]
    store: EvalStoreSettings
    generation: EvalGenerationSettings
    fingerprint: EvalFingerprintSettings
    gate: EvalGateSettings
    reporting: EvalReportingSettings
    deepeval: EvalDeepEvalSettings
    promptfoo: EvalPromptfooSettings

    model_config = SettingsConfigDict(env_prefix="EVALS__")

    @property
    def resolved_manifest_path(self) -> Path:
        return resolve_path(self.manifest_path)

    @property
    def resolved_manifest_paths(self) -> list[Path]:
        """Every manifest that contributes, platform first."""
        return [
            self.resolved_manifest_path,
            *(resolve_path(entry) for entry in self.additional_manifest_paths),
        ]


class ToolsSettings(YamlBackedSettings):
    """Tool registry and retrieval-based selection (LLD §19)."""

    _yaml_section = "tools"
    enabled: bool
    selection_k: int
    selection_min_score: float

    model_config = SettingsConfigDict(env_prefix="TOOLS__")


class CommandSettings(YamlBackedSettings):
    """Console slash commands (LLD §20)."""

    _yaml_section = "commands"
    default_limit: int
    value_preview_chars: int
    # Memory trees (`make show-memory`, `/memory`, `make dream`): how much of
    # a value one line shows, and how many events each session lists.
    tree_preview_chars: int
    memory_tree_events_per_session: int
    recent_window: int

    model_config = SettingsConfigDict(env_prefix="COMMANDS__")


class CommandOutputSettings(BaseModel):
    """How slash-command output is drawn (LLD §20)."""

    title_style: str
    header_style: str
    row_style: str
    note_style: str
    error_style: str
    column_gap: int
    max_column_width: int
    prompt_style: str


class TuiSettings(YamlBackedSettings):
    """Live status feed (LLD §20)."""

    _yaml_section = "tui"
    refresh_interval_ms: int
    show_token_count: bool
    copy_method: Literal["osc52", "file"]
    # Commands that put text on the system clipboard, tried in order; the
    # first one installed wins. OSC 52 alone is not enough: Apple Terminal
    # ignores it, so a selection copied that way silently goes nowhere.
    system_clipboard_commands: list[list[str]]
    # How long the "Copied N characters" toast stays up.
    copy_toast_seconds: float
    max_feed_lines: int
    spinner_frames: str
    command_output: CommandOutputSettings

    model_config = SettingsConfigDict(env_prefix="TUI__")


class AppConfig(BaseSettings):
    """Root config — one settings model per bounded context (LLD §1).

    Every section is built by a factory rather than assigned a default
    instance. A default instance is evaluated once, when this module is
    imported, which would freeze the whole configuration at import time:
    `reload_settings()` would rebuild `AppConfig` and hand back the same
    seventeen section objects it built the first time, having re-read
    nothing. Config files would then only ever be read once per process,
    and a run-scoped overlay applied afterwards — an episode's, for
    instance — would appear in `config_sources()` and change nothing.
    """

    # Each nested model fills its required fields from config.yaml via
    # YamlBackedSettings; mypy can't see that, hence the ignores.
    agent: AgentIdentitySettings = Field(default_factory=lambda: AgentIdentitySettings())   # type: ignore[call-arg]
    logging: LoggingSettings = Field(default_factory=lambda: LoggingSettings())  # type: ignore[call-arg]
    llm: LLMSettings = Field(default_factory=lambda: LLMSettings())  # type: ignore[call-arg]
    persistence: PersistenceSettings = Field(
        default_factory=lambda: PersistenceSettings()  # type: ignore[call-arg]
    )
    audit: AuditSettings = Field(default_factory=lambda: AuditSettings())  # type: ignore[call-arg]
    observability: ObservabilitySettings = Field(
        default_factory=lambda: ObservabilitySettings()  # type: ignore[call-arg]
    )
    context: ContextSettings = Field(default_factory=lambda: ContextSettings())  # type: ignore[call-arg]
    memory: MemorySettings = Field(default_factory=lambda: MemorySettings())  # type: ignore[call-arg]
    guardrails: GuardrailsSettings = Field(default_factory=lambda: GuardrailsSettings())  # type: ignore[call-arg]
    hallucination_protection: HallucinationProtectionSettings = Field(
        default_factory=lambda: HallucinationProtectionSettings()  # type: ignore[call-arg]
    )
    gatekeeper: GatekeeperSettings = Field(default_factory=lambda: GatekeeperSettings())  # type: ignore[call-arg]
    security: SecuritySettings = Field(default_factory=lambda: SecuritySettings())  # type: ignore[call-arg]
    resilience: ResilienceSettings = Field(default_factory=lambda: ResilienceSettings())  # type: ignore[call-arg]
    durability: DurabilitySettings = Field(default_factory=lambda: DurabilitySettings())  # type: ignore[call-arg]
    money: MoneySettings = Field(default_factory=lambda: MoneySettings())  # type: ignore[call-arg]
    evals: EvalsSettings = Field(default_factory=lambda: EvalsSettings())  # type: ignore[call-arg]
    tools: ToolsSettings = Field(default_factory=lambda: ToolsSettings())  # type: ignore[call-arg]
    tui: TuiSettings = Field(default_factory=lambda: TuiSettings())  # type: ignore[call-arg]
    commands: CommandSettings = Field(default_factory=lambda: CommandSettings())  # type: ignore[call-arg]
    # Which capabilities this deployment was given. Declared by the agent
    # rather than the platform, and merged here by the same overlay chain
    # as everything else — so an episode narrows the posture by declaring
    # a `features:` block, not by calling a setter.
    features: FeatureSet = Field(default_factory=lambda: FeatureSet(tree=load_section("features")))

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
