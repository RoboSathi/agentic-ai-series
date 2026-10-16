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

from agent_platform.config.environment import load_environment
from agent_platform.config.features import FeatureSet
from agent_platform.config.paths import resolve_path
from agent_platform.config.yaml_config import deep_merge, load_section

RedactionMode = Literal["mask", "tokenize", "remove"]
GuardrailMode = Literal["reject", "auto_repair", "constrained"]
RetrievalMode = Literal["lexical", "dense", "hybrid", "graph"]
ProvenanceKindName = Literal["verified_system_of_record", "agent_inference", "self_reported"]
# A key of `llm.providers`: which providers exist, and what each is called, is configuration.
ProviderName = str
# How a provider's endpoint is spoken to: `hosted` (OpenAI-compatible, structured output
# validated after generation) or `grammar` (also sends `response_format.json_schema`).
ProviderAdapter = Literal["hosted", "grammar"]
SurfaceClassName = Literal["locked", "editable", "append_only", "human_controlled"]


def _first_env(names: list[str]) -> str | None:
    """The first of these bare environment variables that is set, or None."""
    return next((os.environ[name] for name in names if os.environ.get(name)), None)


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


class SubjectFieldSettings(BaseModel):
    """A field naming what an agent acts on (`order_id` → "order"), shown on cards."""

    field: str
    label: str
    # Whether a one-line card summary names it ("tool requested · order 10042").
    summarise: bool = False


class AgentIdentitySettings(YamlBackedSettings):
    """Identity stamped onto shared platform observability records."""

    _yaml_section = "agent"

    id: str
    # How dashboards and reports title the agent ("" → its id).
    display_name: str
    version: str
    # `module.path:attribute` naming this agent's `AgentHost`. Empty in the
    # platform defaults on purpose: the platform must not name an agent, so
    # an agent declares its own host in its own overlay.
    host: str = ""
    # The fields that name what this agent acts on. Agent vocabulary, so the
    # platform default lists none and each agent's overlay lists its own.
    subject_fields: list[SubjectFieldSettings] = Field(default_factory=list)
    # A session id is `<agent id><separator><hex chars>`: named for the agent
    # that owns the conversation, never for the screen that opened it.
    session_separator: str
    session_hex_chars: int

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
    # Optional sub-folder under the agent's own, for runs that are not the live agent (an offline
    # profile, an eval): its records stay out of the live logs. Empty: the agent's folder itself.
    run_label: str = ""
    file_format: Literal["tree", "json"]
    payload_max_chars: int
    max_bytes: int
    backup_count: int
    # The console's colours, as Rich theme entries (`logging.level.info`, `log.time`, ...), and
    # its line format (`%(name)s` the subsystem). The file log stays plain text.
    console_theme: dict[str, str] = Field(default_factory=dict)
    console_format: str = "%(message)s"
    # Console colour of each subsystem's call traces; an agent's overlay adds its own.
    subsystem_colors: dict[str, str] = Field(default_factory=dict)
    subsystem_default_color: str

    model_config = SettingsConfigDict(env_prefix="LOGGING__")

    @property
    def resolved_dir(self) -> Path:
        return resolve_path(self.dir)


class LLMProviderSettings(BaseModel):
    """One provider's connection and pricing parameters."""

    adapter: ProviderAdapter
    # Bare environment variables holding the provider's key, first present wins; empty for
    # a provider that needs none. Keys are never config fields, so they never reach a dump.
    key_env_vars: list[str] = Field(default_factory=list)
    base_url: str
    model: str
    timeout_seconds: float
    input_cost_per_1k_tokens: float
    # Prompt tokens the provider served from its prefix cache (`prompt_tokens_details`).
    cached_input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float


class LLMCascadeSettings(BaseModel):
    """Model-cascading policy — cheapest sufficient tier first."""

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

    # `hosted`: `model` on the `llm.providers` entry named by `provider`; `hashing`: the
    # deterministic local fallback (no network).
    backend: Literal["hosted", "hashing"]
    provider: ProviderName
    model: str
    local_dimensions: int
    input_types: EmbeddingInputTypes
    truncate: str


class EncoderSettings(BaseModel):
    """BERT-family cross-encoders for entailment and reranking, run in-process (LLD §3)."""

    enabled: bool
    nli_model: str
    nli_local_model_dir: str
    nli_max_length: int
    reranker_model: str
    reranker_local_model_dir: str
    # Query plus one candidate passage, in tokens.
    reranker_max_length: int
    download_if_missing: bool

    @property
    def resolved_nli_local_model_dir(self) -> Path:
        return resolve_path(self.nli_local_model_dir)

    @property
    def resolved_reranker_local_model_dir(self) -> Path:
        return resolve_path(self.reranker_local_model_dir)


class LayaIntentSettings(BaseModel):
    """A Laya decision checkpoint asked one choice question per message.

    The labels and their descriptions are not here: they are the consuming
    agent's intents, injected when the classifier is built.
    """

    model: str
    # One checkpoint of a repository that bundles several (`typed-decisions`).
    subfolder: str
    local_model_dir: str
    device: str
    # The question's id in the request, and the key the message is sent under.
    question_id: str
    state_key: str

    @property
    def resolved_local_model_dir(self) -> Path:
        return resolve_path(self.local_model_dir)


class IntentClassifierSettings(BaseModel):
    """Reusable model/provider settings for semantic intent classification.

    Agent-specific labels and the confidence threshold stay with the
    consuming agent. This section only describes how the shared platform
    obtains and runs the classifier.
    """

    enabled: bool
    provider: Literal["huggingface_local", "laya_local"]
    model: str
    local_model_dir: str
    download_if_missing: bool
    max_length: int
    laya: LayaIntentSettings

    @property
    def resolved_local_model_dir(self) -> Path:
        return resolve_path(self.local_model_dir)


class LocalEmbedderSettings(BaseModel):
    """A local sentence-embedding model (mean-pooled encoder), for an agent that wants
    similarity without a network call per text — the exemplar intent classifier."""

    enabled: bool
    model: str
    local_model_dir: str
    download_if_missing: bool
    max_length: int = Field(ge=8)
    # Text put before each input by purpose ("query: " for e5); empty for most models.
    prefixes: dict[str, str]

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


class LocalCheckSettings(BaseModel):
    """The one tiny input each in-process model is run on by `make model-check`."""

    nli_premise: str
    nli_hypothesis: str
    # The reranker scores each passage against the query; the first should win.
    reranker_query: str
    reranker_passages: list[str]
    intent_text: str
    # A classifier that reads label descriptions (`laya_local`) is asked this, over these
    # options: the probe's own, since the platform knows no agent's intents.
    intent_question: str
    intent_choices: dict[str, str]
    dialogue: str


class BenchmarkSettings(BaseModel):
    """`make model-check BENCH=1`: repeated, judge-sized calls to each chat model."""

    samples: int = Field(ge=1)
    prompt: str
    max_tokens: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)


class CheckStyleSettings(BaseModel):
    """How `make model-check` colours its tables: each cell by what it is."""

    title: str
    border: str
    header: str
    # A probe's status (ok, failed, skipped, downloaded, pending) -> style.
    status: dict[str, str]
    name: str
    model: str
    target: str
    required: str
    optional: str
    detail: str
    failure: str
    # Latency coloured against two bars: under `fast_ms` fast, over `slow_ms` slow.
    fast: str
    medium: str
    slow: str
    fast_ms: float
    slow_ms: float
    # The same bands for the benchmark's p50/p95, in seconds.
    bench_fast_seconds: float
    bench_slow_seconds: float


class LLMCheckSettings(BaseModel):
    """The smallest real request `make llm-check` sends to each endpoint."""

    prompt: str
    max_tokens: int
    # An endpoint that has not answered in this long has failed: the probe is cancelled (and the SDK's
    # own retries with it), so one stalled model cannot hold the whole check for minutes.
    timeout_seconds: float
    local: LocalCheckSettings
    # Use key (chat, judge_fallback, nli, …) → how `make model-check` describes it.
    uses: dict[str, str]
    platform_layer: str
    benchmark: BenchmarkSettings
    styles: CheckStyleSettings


class OutputNormalisationSettings(BaseModel):
    """Typography undone in model output before anything reads it."""

    enabled: bool
    space_categories: list[str]
    replacements: dict[str, str]
    remove: list[str]
    # Regexes removed outright: model-specific markup with no meaning to a reader.
    strip_patterns: list[str]


class ModelRef(BaseModel):
    """One model on one configured provider, and the request fields it is called with."""

    provider: ProviderName
    model: str
    # Provider-specific request fields for this model only (`CompletionRequest.extra_body`).
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        """`provider:model` — what "the same model" means."""
        return f"{self.provider}:{self.model}"


class ModelChainSettings(BaseModel):
    """How one model call is protected: its backups, each attempt's limit, when to hedge.

    `fallbacks` are tried in order when the model before them fails or is slow. Each model's
    attempts are bounded by `timeout_seconds` (unset: `resilience.timeout.default_seconds`)
    and retried by the platform's resilience stack. With `hedge_after_seconds` set, a backup
    starts *beside* a model that has not answered by then, and the first answer wins; unset,
    a backup starts only when the model before it has failed.
    """

    fallbacks: list[ModelRef] = Field(default_factory=list)
    timeout_seconds: float | None = Field(default=None, gt=0)
    hedge_after_seconds: float | None = Field(default=None, gt=0)


class RoleModelSettings(ModelRef, ModelChainSettings):
    """Which model plays one named role, and how its calls are protected."""


class TokenizerSettings(BaseModel):
    """The tokenizer a request's size is counted with, before the provider counts it.

    One `tokenizer.json` from a model family, in a repository folder, read by the Rust
    `tokenizers` library. Unset or unloadable, sizes fall back to characters per token —
    an estimate the CONTEXT card labels as one, and reconciles with the provider's count.
    """

    enabled: bool
    model: str
    local_dir: str
    file: str
    download_if_missing: bool

    @property
    def resolved_local_dir(self) -> Path:
        return resolve_path(self.local_dir)


class ContextWindowSizes(BaseModel):
    """How many tokens each model can read: what a request's size is a share of."""

    default_tokens: int = Field(gt=0)
    by_model: dict[str, int] = Field(default_factory=dict)

    def for_model(self, model: str) -> int:
        """The model's window, or the default for a model not listed."""
        return self.by_model.get(model, self.default_tokens)


class LogprobSettings(BaseModel):
    """Whether completions ask for token log-probabilities, and how many alternatives each."""

    enabled: bool
    top_logprobs: int = Field(ge=0, le=20)


class LLMSettings(YamlBackedSettings):
    """LLM adapters — the only place a provider SDK is used (LLD §3)."""

    _yaml_section = "llm"
    enabled: bool
    provider: ProviderName
    max_tokens: int
    log_payloads: bool
    payload_max_chars: int
    # Where a provider returns the model's reasoning beside its answer
    # (`reasoning_content` on NIM). Read in order; the first present wins.
    reasoning_fields: list[str] = Field(default_factory=list)
    logprobs: LogprobSettings
    tokenizer: TokenizerSettings
    context_windows: ContextWindowSizes
    providers: dict[str, LLMProviderSettings]
    # The default client's protection (`create_llm_client`): backups, timeout, hedge.
    chain: ModelChainSettings = Field(default_factory=ModelChainSettings)
    cascade: LLMCascadeSettings
    embedding: EmbeddingSettings
    encoders: EncoderSettings
    intent_classifier: IntentClassifierSettings
    dialogue_summarizer: DialogueSummarizerSettings
    local_embedder: LocalEmbedderSettings
    check: LLMCheckSettings
    output_normalisation: OutputNormalisationSettings
    # Warnings a third-party library raises while a local model loads, matched by message
    # (a regex) and silenced for that load only: a dependency's own deprecation, not ours.
    suppressed_load_warnings: list[str] = Field(default_factory=list)
    # Role name -> model. Empty in the platform: role names are an agent's
    # vocabulary (`RoleModelRouter`), declared in its own overlay.
    roles: dict[str, RoleModelSettings]

    model_config = SettingsConfigDict(env_prefix="LLM__")

    def provider_settings(self, name: ProviderName | None = None) -> LLMProviderSettings:
        """Connection settings for one provider (defaults to the active one)."""
        return self.providers[name or self.provider]

    def api_key_for(self, name: ProviderName) -> str | None:
        """Resolve a provider's key from its bare env var, never from config.

        Keys are deliberately not fields: they never appear in model_dump()
        and so can never leak into a log line or settings dump.
        """
        return _first_env(self.providers[name].key_env_vars)


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


class StopReportWording(BaseModel):
    """What a stopped run says, the same for every agent (`durability/stop_report.py`)."""

    # `{step}`: the stage the run resumes at, in capitals.
    heading: str
    choices: str
    aborted: str
    nothing_to_abort: str
    # What `step` says when the run stopped after its last stage.
    the_end: str


class StagnationSettings(BaseModel):
    semantic: bool
    semantic_threshold: float = Field(gt=0, le=1)
    cycle_max_period: int = Field(ge=1)
    cycle_repeats: int = Field(ge=2)
    history_window: int = Field(ge=2)


class ConvergenceSettings(BaseModel):
    enabled: bool
    window: int = Field(ge=1)
    min_gain: float = Field(ge=0, le=1)


class HistoricalBaselineSettings(BaseModel):
    enabled: bool
    min_samples: int = Field(ge=2)
    sample_size: int = Field(ge=2)
    z_threshold: float = Field(gt=0)


class SuppressionMessages(BaseModel):
    suppressed_success: str
    suppressed_refused: str
    suppressed_similar: str


class LoopCardWords(BaseModel):
    summary: str
    layer_suffix: str
    intervention: str
    sections: dict[str, str]
    budget_line: str
    check_line: str


class LoopControlSettings(YamlBackedSettings):
    """The loop governor's mechanisms and wording (`loop_control/governor.py`)."""

    _yaml_section = "loop_control"
    call_timeout_seconds: float = Field(gt=0)
    min_call_seconds: float = Field(ge=0)
    stagnation: StagnationSettings
    convergence: ConvergenceSettings
    historical: HistoricalBaselineSettings
    messages: SuppressionMessages
    reasons: dict[str, str]
    card: LoopCardWords

    model_config = SettingsConfigDict(env_prefix="LOOP_CONTROL__")


class DurabilitySettings(YamlBackedSettings):
    """Durable execution: checkpoints, resumable runs (LLD §13)."""

    _yaml_section = "durability"
    db_path: str
    # LangGraph's durability mode. `sync` writes each checkpoint before
    # the next node starts, so a crash loses at most the node in flight.
    mode: DurabilityMode
    retention_days: int
    # How much of a run's query the run registry keeps for display.
    preview_chars: int
    encryption: CheckpointEncryptionSettings
    # Recorded as the error of a run an operator cancelled, or aborted after stopping it.
    cancel_reason: str
    abort_reason: str
    stop_report: StopReportWording

    model_config = SettingsConfigDict(env_prefix="DURABILITY__")

    @property
    def resolved_db_path(self) -> Path:
        return resolve_path(self.db_path)


class MoneySettings(YamlBackedSettings):
    """How amounts are held and shown (see `agent_platform.core.primitives.money`)."""

    _yaml_section = "money"
    currency: str
    # Decimal places of the minor unit: 2 for cents.
    minor_units: int
    symbol: str

    model_config = SettingsConfigDict(env_prefix="MONEY__")


class AuditViewSettings(BaseModel):
    """How the audit trail reads in a terminal and the console (`audit/view.py`)."""

    decision_colours: dict[str, str]
    standing_colours: dict[str, str]
    since_pattern: str
    since_units: dict[str, str]
    trace_entry: str
    trace_detail: str
    trace_details: list[str]
    locked: str
    title: str
    trace_title: str


class AuditSettings(YamlBackedSettings):
    """Append-only decision ledger (LLD §5)."""

    _yaml_section = "audit"
    enabled: bool
    policy_version: str
    data_version: str
    prompt_version: str
    tool_schema_version: str
    # Where the versioned prompt and policy live, as `/versions` reports them.
    # The agent names its own; the platform default names none.
    prompt_source: str
    policy_source: str
    view: AuditViewSettings

    model_config = SettingsConfigDict(env_prefix="AUDIT__")


class ReviewQueueSettings(BaseModel):
    """Little's Law queue model inputs."""

    minutes_per_hour: float
    minutes_per_review: float


class ModelEventWording(BaseModel):
    """Card wording for a model chain's backup events."""

    backup_started: str
    # Why a backup started, in words, by `StartReason`.
    reasons: dict[str, str]
    primary_label: str
    backup_label: str


class StepCardSettings(BaseModel):
    """The one layout every step card uses (`observability/step_card.py`)."""

    # Section key -> title, in the order a card shows them.
    sections: dict[str, str]
    model_line: str
    backup_line: str
    # When the step was decided without a model (`{by}`: the rule that decided).
    no_model_line: str
    # Appended to the model line when it stands for several calls.
    calls_suffix: str
    # Where a structured answer states its own rationale, read when the provider gave no reasoning.
    rationale_fields: list[str]


class LangSmithSettings(BaseModel):
    """Exporting each turn's trace to LangSmith (`observability/langsmith.py`)."""

    enabled: bool
    endpoint: str
    web_url: str
    # `{agent_id}`: each agent's turns land in a project of their own.
    project: str
    root_run_name: str
    # The metadata key LangSmith groups a conversation's turns by (a thread).
    thread_key: str
    # Send each graph step's full input state (true) or only what it changed (false).
    full_node_state: bool
    # Share of turns exported, 0.0–1.0, decided per turn; an agent's overlay may lower it.
    sample_rate: float = Field(ge=0.0, le=1.0)
    # Bare env vars tried in order. The key is never a field, so it never
    # reaches model_dump() or a log line.
    key_env_vars: list[str]
    # `{web_url}`, `{tenant_id}`, `{project_id}`, `{run_id}`.
    run_url: str
    # The turn summary's field that carries the link.
    url_field: str

    def api_key(self) -> str | None:
        """The first configured key variable that is set."""
        return _first_env(self.key_env_vars)


class LoopControlWords(BaseModel):
    """The LOOP CONTROL card: its heading, and why the loop chose its next step."""

    summary: str
    # Reason code → sentence; an agent adds the reasons only its loop has.
    because: dict[str, str]


class LlmPurposeWords(BaseModel):
    """Why a model call was made, as its LLM REQUEST / RESPONSE cards name it.

    `next_action` is a loop step's call, which may ask for a tool or answer: its request says
    so, and its response names what it produced (`tool_selection` or `reply`).
    """

    next_action: str
    tool_selection: str
    reply: str
    reply_revision: str
    hallucination_repair: str


class DreamInboxWords(BaseModel):
    """The turn summary's dream-inbox line (`observability/turn_tally.py`)."""

    queued: str
    waiting: str


class ContextCardWords(BaseModel):
    """The CONTEXT card drawn before every model request (`observability/context_card.py`)."""

    heading: str
    heading_estimated: str
    sections_title: str
    request_title: str
    window_title: str
    findings_title: str
    prefix_reused: str
    prefix_new: str
    model_line: str
    section_line: str
    part_line: str
    source_line: str
    tools_line: str
    tools_title: str
    prefix_line: str
    prefix_none: str
    tokenizer_line: str
    # Many requests of one purpose drawn as one card (`StageReporterModelObserver(aggregate=…)`):
    # `{purpose}`, `{count}`, `{low}`, `{high}` tokens, `{peak_share}` of `{window}`.
    aggregate_heading: str
    aggregate_title: str
    # One request of an aggregated card: `{tokens}`, `{share}`, `{model}`.
    aggregate_line: str
    # Section name → what a person reads for it; an unlisted name is shown as it is.
    labels: dict[str, str]


class StepEventWords(BaseModel):
    """Headings of the cards every agent's tool-use loop draws (`observability/step_events.py`)."""

    tool_requested: str
    tool_prefetched: str
    tool_served: str
    gatekeeper: str
    # What `{subject}` in the Gatekeeper heading reads as when the call is about something
    # ("order 10532"); empty when it is about nothing.
    gatekeeper_subject: str
    # TOOL RESULT headings by how far the call got: `{tool}`, and `{stage}` for one not run.
    tool_done: str
    tool_failed: str
    tool_refused: str
    tool_not_run: str
    loop_control: LoopControlWords


class ObservabilitySettings(YamlBackedSettings):
    """Tracing, SLOs, cost stack, queue math (LLD §6)."""

    _yaml_section = "observability"
    enabled: bool
    langsmith: LangSmithSettings
    persist_traces: bool
    # One row per model call, tool call and turn (`observability/facts.py`):
    # what the analytics source and the dashboards aggregate.
    persist_facts: bool
    slo_targets: dict[str, float]
    review_queue: ReviewQueueSettings
    step_card: StepCardSettings
    # How a card in the tool-use loop names its iteration. `{step}` is the
    # model call being made, `{max}` the cap on tool iterations.
    loop_step_prefix: str
    loop_step_detail: str
    # The model-call cards' headings: `{purpose}` is why the call was made,
    # `{tool}` the tool the model asked for.
    llm_request_heading: str
    context_card: ContextCardWords
    dream_inbox: DreamInboxWords
    llm_response_heading: str
    llm_tool_requested_heading: str
    # What a response card shows for cached input tokens when the provider did not say.
    cache_not_reported: str
    # The turn summary's prompt-cache line: `{hits}`/`{calls}` calls, `{cached}`/`{prompt}` tokens.
    prompt_cache_line: str
    llm_purposes: LlmPurposeWords
    step_events: StepEventWords
    # What a tool card shows when the model's arguments are not JSON.
    unparseable_arguments: str
    # An `amount` argument that is not money, as the card shows it.
    invalid_amount: str
    # What a model call's backup cards say (`ModelCallObserver`): `{label}` the role or
    # client, `{reason}` slow/failed. Headings never name a model; the body does.
    model_events: "ModelEventWording"

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY__")


class RedactPattern(BaseModel):
    """One regex the analytics source masks in free text before it leaves the database."""

    pattern: str
    replacement: str


class QueryRejectionWords(BaseModel):
    """Why the query guard refused a query, in words a model can act on."""

    unparseable: str
    single_statement: str
    not_a_select: str
    table_function: str
    qualified_name: str
    unknown_view: str
    denied_function: str
    unknown_metric: str
    unknown_dimension: str
    # `{error}`, `{columns}`, `{hint}` (`did_you_mean` with `{names}`, or nothing).
    not_runnable: str
    did_you_mean: str
    # `{agent}` the id asked for, `{agents}` the ones the source holds.
    unknown_agent: str


class QueryNoteWords(BaseModel):
    """What a result says when it may not answer the question: empty, or cut (`{rows}`)."""

    empty: str
    truncated: str


class AnalyticsQuerySettings(BaseModel):
    """The limits every NL-to-SQL query over the source runs under (`analytics/query_guard.py`)."""

    dialect: str
    # The session time zone results are reported in (the source's timestamps are UTC).
    timezone: str
    agent_column: str
    max_rows: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    max_result_bytes: int = Field(ge=1024)
    sample_rows: int = Field(ge=0)
    # Functions no query may call, even over the curated views (lower case).
    denied_functions: list[str]
    rejections: QueryRejectionWords
    notes: QueryNoteWords
    timed_out: str


class AnalyticsMcpServerSettings(BaseModel):
    """How the analytics MCP server introduces itself to a client."""

    name: str
    version: str
    instructions: str
    manifest_uri: str


class PlatformDashboardSettings(BaseModel):
    """The Agent Platform dashboard: every agent together, and side by side."""

    name: str
    display_name: str
    about: str


class ByAgentSettings(BaseModel):
    """The platform page's per-section table of agents."""

    dimension: str
    title: str
    measures: int = Field(ge=1)


class DashboardRanking(BaseModel):
    """A leaderboard: one view, ranked by one dimension, showing some of its measures."""

    view: str
    by: str
    measures: list[str]


class DashboardKpis(BaseModel):
    """A titled KPI grid: some measures of one view, shown when the agent runs `requires`."""

    title: str
    view: str
    measures: list[str]
    # Check names at least one of which the scope must have run (e.g. its screens); empty: always.
    requires: list[str] = Field(default_factory=list)


class DashboardTable(BaseModel):
    """A table: one view, a row per combination of `by`, a column per measure."""

    view: str
    by: list[str]
    measures: list[str]
    title: str = ""


class DashboardGroup(BaseModel):
    """One dashboard section: its heading, KPI grids, trend charts, rankings and table."""

    title: str
    about: str = ""
    # Explicit KPI grids; without them, one grid per view the section's metrics read.
    kpis: list[DashboardKpis] = Field(default_factory=list)
    trends: list[str] = Field(default_factory=list)
    rankings: list[DashboardRanking] = Field(default_factory=list)
    table: DashboardTable | None = None


class DashboardLayout(BaseModel):
    """Sizes and KPI options of the generated canvases."""

    columns: int = Field(ge=1)
    title_height: str
    header_height: str
    kpi_height: str
    chart_height: str
    ranking_rows: int = Field(ge=1)
    comparison: list[str]
    sparkline: str


class DummyIncidentSettings(BaseModel):
    """The one bad day in the generated world."""

    days_ago: int = Field(ge=0)
    crash_rate: float = Field(ge=0, le=1)
    retry_rate: float = Field(ge=0, le=1)
    backup_rate: float = Field(ge=0, le=1)
    latency_factor: float = Field(gt=0)
    guard_failure_rate: float = Field(ge=0, le=1)


class DummyAnalyticsSettings(BaseModel):
    """A generated analytics world for demos (`analytics/dummy.py`)."""

    # The analytics source (`analytics.sources`) the world is written to.
    source: str
    profiles_glob: str
    days: int = Field(ge=1)
    seed: int
    eval_share: float = Field(ge=0, le=1)
    growth: float = Field(ge=0)
    weekend_factor: float = Field(gt=0)
    active_hours: tuple[int, int]
    turn_gap_seconds: tuple[int, int]
    crash_rate: float = Field(ge=0, le=1)
    cancel_rate: float = Field(ge=0, le=1)
    retry_rate: float = Field(ge=0, le=1)
    backup_rate: float = Field(ge=0, le=1)
    incident: DummyIncidentSettings


class DashboardDurations(BaseModel):
    """Every duration measure in one unit and one format, so a row of tiles never mixes units."""

    ending: str
    divisor: float = Field(gt=0)
    unit: str
    format_d3: str
    title: str
    note: str


class DashboardSettings(BaseModel):
    """The Rill project generated from the semantic layer (`analytics/dashboards.py`)."""

    project_dir: str
    # The first line of every generated file: what marks it safe to regenerate.
    marker: str
    data_link: str
    time_range: str
    platform: PlatformDashboardSettings
    agent_configs_glob: str
    agent_about: str
    by_agent: ByAgentSettings
    groups: dict[str, DashboardGroup]
    layout: DashboardLayout
    durations: DashboardDurations
    # Format preset by the metric name's ending.
    formats: dict[str, str]
    default_format: str

    @property
    def resolved_project_dir(self) -> Path:
        return resolve_path(self.project_dir)


class AnalyticsWords(BaseModel):
    """What `make analytics-export` says (`analytics/cli.py`)."""

    usage: str
    dashboards_written: str
    dropped: str
    dummy_written: str
    title: str
    rows: str
    skipped: str
    # A fact-backed view with no rows yet, and the line closing a source that has none at all.
    empty_hint: str
    no_facts: str
    locked: str
    time_format: str


class OutcomeFlags(BaseModel):
    """What an outcome means for the fleet (ADR 0009 D4)."""

    # Did the person get what they asked for.
    success: bool
    # Does the task end here.
    terminal: bool


class ReservedOutcomes(BaseModel):
    """The outcomes the platform assigns itself, never an agent (UL: implicit task, abandoned)."""

    unclassified: str
    crashed: str
    cancelled: str
    abandoned: str


class TaskSettings(BaseModel):
    """Task kinds and outcomes: the platform's reserved ones, plus each agent's (D2, D4).

    An agent's overlay adds its kinds and outcomes under `analytics.tasks`; the
    outcome mapping merges with the platform's reserved outcomes.
    """

    implicit_kind: str
    kinds: list[str]
    reserved: ReservedOutcomes
    outcomes: dict[str, OutcomeFlags]
    # An open task older than this is reported `abandoned` by the source.
    abandon_after_hours: float = Field(gt=0)


class AnalyticsSource(BaseModel):
    """One analytics dataset: the DuckDB facts go to, and the Parquet its views export to."""

    db_path: str
    parquet_dir: str

    @property
    def resolved_db_path(self) -> Path:
        return resolve_path(self.db_path)

    @property
    def resolved_parquet_dir(self) -> Path:
        return resolve_path(self.parquet_dir)


class AnalyticsSettings(YamlBackedSettings):
    """Analytics sources: curated views over the facts, exported as Parquet (LLD §6)."""

    _yaml_section = "analytics"
    # The source readers read, by name (`ANALYTICS__SOURCE`, `make … SOURCE=dummy`).
    source: str
    # The source a running agent records its facts and audit to. Never moved by `SOURCE`:
    # pointing a reader at the demo world must not send a live run's records there.
    record_source: str
    sources: dict[str, AnalyticsSource]
    schema_version: int
    manifest_file: str
    # How often a running console re-exports its source's Parquet; 0 exports only on demand.
    export_interval_seconds: float = Field(ge=0)
    semantic_layer: str
    # Agents' Tier-2 semantic layers, found by convention (D7): each file names its agent.
    agent_layers_glob: str
    evals_alias: str
    slo_windows_days: list[int]
    # The channel a fact is stamped with when no surface bound one (a script, a make target).
    default_channel: str
    tasks: TaskSettings
    redact_patterns: list[RedactPattern]
    query: AnalyticsQuerySettings
    mcp_server: AnalyticsMcpServerSettings
    legacy_tables: list[str]
    dummy: DummyAnalyticsSettings
    dashboards: DashboardSettings
    words: AnalyticsWords

    model_config = SettingsConfigDict(env_prefix="ANALYTICS__")

    @property
    def active(self) -> AnalyticsSource:
        """The source readers read (an unknown name fails loudly, naming the ones there are)."""
        return self._named("source", self.source)

    @property
    def recording(self) -> AnalyticsSource:
        """The source a running agent records to, whatever readers were pointed at."""
        return self._named("record_source", self.record_source)

    def _named(self, key: str, name: str) -> AnalyticsSource:
        if name not in self.sources:
            raise ValueError(f"analytics.{key} '{name}' is not one of {sorted(self.sources)}")
        return self.sources[name]

    @property
    def resolved_db_path(self) -> Path:
        return self.active.resolved_db_path

    @property
    def resolved_parquet_dir(self) -> Path:
        return self.active.resolved_parquet_dir

    @property
    def resolved_semantic_layer(self) -> Path:
        return resolve_path(self.semantic_layer)


class RerankSettings(BaseModel):
    """Cross-encoder reranking of the retrieved shortlist (LLD §7)."""

    shortlist: int
    min_candidates: int


class ContextStrategySettings(BaseModel):
    """Per-strategy toggles for the six context-management strategies."""

    offload: bool
    reduce: bool
    retrieve: bool
    isolate: bool
    cache: bool
    consolidate: bool


class ContextFailureDetectionSettings(BaseModel):
    """Context-failure detection thresholds."""

    checkpoint_fractions: list[float]
    degradation_threshold: float = Field(gt=0, lt=1)
    min_observations: int = Field(ge=1)
    max_observations: int = Field(ge=1)
    mode_ratio: float = Field(gt=0, le=1)
    min_relevance: float = Field(ge=0, le=1)
    max_tools: int = Field(ge=1)
    max_tool_schema_share: float = Field(gt=0, le=1)
    # Requests smaller than this are never flagged for their schemas' share.
    min_request_tokens_for_share: int = Field(ge=0)
    nli: bool
    max_pairs: int = Field(ge=0)
    max_sentences: int = Field(ge=1)
    min_sentence_chars: int = Field(ge=1)
    contradiction_score: float = Field(ge=0, le=1)
    # A line that states where the harness is, not something about the world
    # (`order_under_discussion: order 10538`): it is never a fact a sentence is checked against.
    state_line_pattern: str
    # Modes whose cut lines are also struck from the source that serves them next turn.
    retract_modes: list[str]


class FindingTemplate(BaseModel):
    rule: str
    detail: str


class ContextFindingWords(BaseModel):
    """What a context finding says: the action taken, and each rule's lines (`context/failure/`)."""

    actions: dict[str, str]
    # Action key → the tone its source is drawn in (`tui.tone_styles`: pass, warn, fail).
    action_tones: dict[str, str]
    templates: dict[str, FindingTemplate]


class ContextSimilaritySettings(BaseModel):
    """Which local model answers "how alike are these texts" (`context/similarity.py`)."""

    backend: Literal["local_embedder", "hashing", "lexical"]


class ContextConsolidationSettings(BaseModel):
    near_duplicate_overlap: float = Field(gt=0, le=1)
    semantic_threshold: float = Field(gt=0, le=1)


class ContextOffloadSettings(BaseModel):
    """Offload: what leaves the window, the placeholder left behind, the tool that fetches it."""

    min_tokens_fraction: float = Field(gt=0, le=1)
    handle_chars: int = Field(ge=6)
    max_entries: int = Field(ge=1)
    placeholder: str
    tool_namespace: str
    tool_name: str
    tool_permission: str
    tool_description: str
    not_found: str


ReduceMode = Literal["excerpt", "tail", "head"]


class ContextReduceSettings(BaseModel):
    keep_ratio: float = Field(gt=0, lt=1)
    by_kind: dict[str, ReduceMode]
    excerpt_window_chars: int = Field(gt=0)
    marker: str


TrustClass = Literal["system", "verified", "retrieved", "user", "model"]


class ContextTrustSettings(BaseModel):
    """Who is believed: a chunk's trust class by source prefix, then by kind, and the rank."""

    by_source_prefix: dict[str, TrustClass]
    by_kind: dict[str, TrustClass]
    rank: dict[TrustClass, int]
    model_line_prefix: str


class ContextWindowWording(BaseModel):
    """What the `context_assembly` card says (`context/window.py`), for every agent."""

    started: str
    assembled: str
    # Said (quietly) when an agent withholds context assembly.
    withheld: str
    # `{modes}` the failure modes found, `{action}` what mitigation did.
    unhealthy: str
    # `{count}` sources dropped or cut by mitigation, `{sources}` their names.
    removed: str
    nothing_removed: str
    # One finding: `{mode}`, `{source}`, `{rule}`, `{detail}`, `{action}`.
    finding_line: str
    finding_heading: str
    size_title: str
    size_line: str
    findings_title: str
    marks: dict[str, str]
    # One finding as a tree: `{mark} {mode} · {rule}` over the lines below. `{line}` is the
    # offending text, `{source}` the chunk it was in, `{action}` what was done with a kept one.
    finding_card: str
    # How long a quoted line may be, and how many removed lines a finding names before
    # "and {count} more" (`more_removed`).
    snippet_chars: int = Field(gt=0)
    max_removed_lines: int = Field(gt=0)
    more_removed: str
    flagged_line: str
    statement_line: str
    # A statement with the chunk it came from: `{source}`, `{line}`.
    sourced_statement_line: str
    # A chunk kind -> how the card names where a line came from ("history" -> "conversation").
    kind_words: dict[str, str]
    removed_chunk: str
    removed_line: str
    struck_line: str
    kept_source: str
    detail_title: str
    detail_note: str
    # Keys of one finding's entry on the card: statements, detail, sources.
    finding_labels: dict[str, str]
    # Failure modes that are retrieval noise the check already removed: no warning for them.
    routine_modes: list[str]
    # A finding's `action` when mitigation did nothing, and the prefix of one it dropped.
    not_acted_on: str
    dropped_action: str
    # A clean window is logged and counted but draws no card of its own.
    quiet_when_healthy: bool


class PromptWindowParts(BaseModel):
    """How a context window's chunks are named on the CONTEXT card: by source prefix, then kind."""

    by_source: dict[str, str]
    by_kind: dict[str, str]


class TurnContextSettings(BaseModel):
    """The message that carries a request's per-turn sections when it has history."""

    role: Literal["system", "user"]
    heading: str


class PromptSettings(BaseModel):
    """How a request is built from named sections (`llm/prompt.py`)."""

    separator: str
    window_heading: str
    signature_chars: int = Field(gt=0)
    # How many (model, prefix) pairs are remembered, to say whether a prefix is reused.
    prefix_memory: int = Field(gt=0)
    turn_context: TurnContextSettings
    window_parts: PromptWindowParts
    # A message role → the section a request no builder made puts it in.
    role_sections: dict[str, str]


class ContextSettings(YamlBackedSettings):
    """Five-stage assembly pipeline, retrieval and strategies (LLD §7)."""

    _yaml_section = "context"
    prompt: PromptSettings
    token_budget: int
    chars_per_token: int
    retrieval_mode: RetrievalMode
    top_k: int
    hybrid_lexical_weight: float
    rerank: RerankSettings
    bm25_k1: float
    bm25_b: float
    strategies: ContextStrategySettings
    # What the CONTEXT card says each strategy did; `{count}` (and `{kinds}`, `{handles}`).
    strategy_notes: dict[str, str]
    similarity: ContextSimilaritySettings
    consolidation: ContextConsolidationSettings
    # A task kind → the chunk kinds its window may hold (Isolate); unlisted kinds keep all.
    isolation: dict[str, list[str]]
    offload: ContextOffloadSettings
    reduce: ContextReduceSettings
    trust: ContextTrustSettings
    findings: ContextFindingWords
    failure_detection: ContextFailureDetectionSettings
    window: ContextWindowWording

    model_config = SettingsConfigDict(env_prefix="CONTEXT__")


class EnabledToggle(BaseModel):
    """A section whose only setting is whether it runs."""

    enabled: bool


class MemoryIndexWords(BaseModel):
    """How each store's MEMORY.md reads: a title, then one line per memory."""

    title: str
    line: str
    empty: str
    description_chars: int = Field(gt=0)


class MemoryFilesSettings(BaseModel):
    """Where an agent's long-term memory lives on disk, and how its files are named (ADR 0012)."""

    root: str
    dream_events_dir: str
    episodic_dir: str
    semantic_dir: str
    procedural_dir: str
    pending_dir: str
    consolidated_dir: str
    history_dir: str
    index_file: str
    derived_index_file: str
    lock_file: str
    sequence_file: str
    record_digits: int = Field(gt=0)
    record_suffix: str
    event_suffix: str
    session_log_suffix: str
    frontmatter_delimiter: str
    shared_subject_dir: str
    name_digest_chars: int = Field(gt=0)
    index: MemoryIndexWords

    @property
    def resolved_root(self) -> Path:
        return resolve_path(self.root)

    @property
    def dream_events_root(self) -> Path:
        return self.resolved_root / self.dream_events_dir

    @property
    def episodic_root(self) -> Path:
        return self.resolved_root / self.episodic_dir

    @property
    def semantic_root(self) -> Path:
        return self.resolved_root / self.semantic_dir

    @property
    def procedural_root(self) -> Path:
        return self.resolved_root / self.procedural_dir


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
    """Write-path routing thresholds."""

    enabled: bool
    interrogative_markers: list[str]
    auto_commit_threshold: float
    discard_threshold: float
    authority_changing_categories: list[str]
    classification_hints: dict[str, list[str]]


class MemoryRetrievalWeights(BaseModel):
    """Composite retrieval scoring weights (Park et al. extended)."""

    relevance: float
    recency: float
    importance: float
    authority: float
    temporal_validity: float
    # How often a fact has been recalled — reinforcement, not popularity:
    # a fact the agent keeps needing is one it should keep finding.
    frequency: float


MemoryPrincipalRole = Literal["customer", "agent", "system_of_record", "curator", "data_protection"]


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


class LearnedSkillWords(BaseModel):
    """How a dream writes a skill it learned: a SKILL.md, worded by signal kind."""

    artifact_type: str
    name: str
    description: str
    unit: str
    precondition_scoped: str
    precondition_any: str
    body: str
    default_lesson: str
    lessons: dict[str, str] = Field(default_factory=dict)
    keyword_min_chars: int = Field(gt=0)
    max_keywords: int = Field(gt=0)


class DreamSettings(BaseModel):
    """Offline consolidation of an agent's dream events (`make dream <agent>`)."""

    # Consolidated events older than this are pruned by each dream.
    retention_days: int = Field(gt=0)
    # The kind a completed turn's event carries.
    turn_event_kind: str
    # strftime pattern naming a pending event's file, so a listing is arrival order.
    event_file_time_format: str
    # Add a locally generated intent sentence to each session summary.
    summarize_intent: bool
    # The principal a dream acts as. An agent role, so a proposed skill is
    # refused by governance and lands with a curator.
    actor: str
    # How much of a tool result one outcome line quotes.
    outcome_chars: int = Field(gt=0)
    # Words a generated summary sentence may not contain (gendered pronouns:
    # the summarizer cannot know them). Such a sentence is dropped.
    forbidden_words: list[str]
    # Share of real words below which a generated sentence is decoding noise.
    min_word_ratio: float = Field(ge=0.0, le=1.0)
    # Read-only tools, named by the agent. A summary condenses their calls
    # to "looked up …" and quotes only what changed something.
    lookup_tools: list[str] = Field(default_factory=list)
    # Prefixes that mark a tool result as a refusal, and the signal kind one becomes.
    refusal_markers: list[str]
    refusal_signal_kind: str
    # In how many distinct sessions (or runs) a signal recurs before it is a skill candidate.
    skill_threshold: int = Field(gt=0)
    learned_skill: LearnedSkillWords


class SkillSelectionSettings(BaseModel):
    """How many skills one prompt carries, and when a query names one."""

    max_active: int = Field(gt=0)
    min_lexical_score: float = Field(ge=0.0)
    # Words that carry no task ("is", "can", "you", "it"). They are dropped from a query before it
    # is scored, so a skill is never chosen because a question was phrased politely.
    function_words: list[str]


class SkillRenderSettings(BaseModel):
    """How skills and their refs are written into a prompt and a record."""

    skill_heading: str
    reference_heading: str
    separator: str
    sections_sent: list[str]
    ref: str
    profile_ref: str


class SkillReasonWords(BaseModel):
    """Why a skill was activated, as recorded on its card and audit entry."""

    always: str
    trigger: str
    lexical: str


class SkillObservabilitySettings(BaseModel):
    """The stage names and summaries of the activation and sync events."""

    activation_stage: str
    sync_stage: str
    activated: str
    joiner: str
    offered: str
    offered_none: str
    synced: str
    unchanged: str


class SkillAuditSettings(BaseModel):
    """Who the audit trail says activated or synced skills, and as what action."""

    actor: str
    activation_action: str
    sync_action: str


class SkillSettings(YamlBackedSettings):
    """The profile (AGENT.md) and authored skills (SKILL.md) — ADR 0010."""

    _yaml_section = "skills"
    profile_path: str
    skills_dir: str
    skill_file: str
    references_dir: str
    references_glob: str
    frontmatter_delimiter: str
    section_pattern: str
    step_pattern: str
    internal_tool_reference: str
    name_pattern: str
    max_name_chars: int = Field(gt=0)
    max_description_chars: int = Field(gt=0)
    always_trigger: str
    profile_sections: list[str]
    skill_sections: list[str]
    authored_artifact_type: str
    selectable_artifact_types: list[str]
    digest_chars: int = Field(gt=0)
    selection: SkillSelectionSettings
    max_prompt_chars: int = Field(ge=0)
    render: SkillRenderSettings
    reasons: SkillReasonWords
    observability: SkillObservabilitySettings
    audit: SkillAuditSettings

    model_config = SettingsConfigDict(env_prefix="SKILLS__")

    @property
    def resolved_profile_path(self) -> Path:
        return resolve_path(self.profile_path)

    @property
    def resolved_skills_dir(self) -> Path:
        return resolve_path(self.skills_dir)


class MemoryCardWording(BaseModel):
    """What the memory-write card says: only what a person needs to decide if it was right."""

    # Why a message was refused by the poisoning screen: `{pattern}` is the rule it matched.
    screen_reason: str
    # How much of the message the card quotes.
    said_chars: int = Field(gt=0)
    # The review-queue entry the write is waiting in (`{id}`), shown only when there is one.
    review_queue: str
    # Gate outcomes that carry no extractor name: a tier path entry ending in one of these
    # says where the write stopped, not how a fact was read.
    gate_tiers: list[str]


class MemorySettings(YamlBackedSettings):
    """All four CoALA stores plus the write pipeline (LLD §8)."""

    _yaml_section = "memory"
    schema_version: str
    files: MemoryFilesSettings
    semantic: SemanticMemorySettings
    write_pipeline: MemoryWritePipelineSettings
    retrieval_weights: MemoryRetrievalWeights
    provenance_weights: dict[str, float]
    recency_half_life_days: float
    governance: MemoryGovernanceSettings
    deterministic: MemoryDeterministicSettings
    importance: MemoryImportanceSettings
    subject_label: str
    no_subject_label: str
    card: MemoryCardWording
    dream: DreamSettings

    model_config = SettingsConfigDict(env_prefix="MEMORY__")


class ReviewKinds(BaseModel):
    """The platform's own review kinds."""

    memory_fact: str
    skill: str
    quarantine: str


class ReviewWords(BaseModel):
    """How `/review` and the Review tab word the queue."""

    summary: str
    usage: str
    title: str
    empty: str
    show_title: str
    decided: str
    hints: dict[str, str]
    kind_title: str
    # A kind's key -> what a person reads for it ("reply" -> "reply cut by the hallucination
    # check"). The key is what is stored and typed; only what is shown changes.
    kind_labels: dict[str, str] = Field(default_factory=dict)
    time_format: str
    none: str
    columns: dict[str, list[str]]
    show_headers: list[str]
    needs_id: str
    needs_comment: str
    unknown_item: str
    already_decided: str
    unknown_kind: str
    quarantine_released: str
    quarantine_upheld: str
    tab_title: str
    comment_placeholder: str
    approve_button: str
    deny_button: str
    nothing_selected: str

    def label(self, kind: str) -> str:
        """The words for a kind; an unlabelled kind is shown as its key."""
        return self.kind_labels.get(kind, kind)


class ReviewSettings(YamlBackedSettings):
    """One human-review queue for every kind (ADR 0012)."""

    _yaml_section = "review"
    table: str
    min_comment_chars: int = Field(gt=0)
    kinds: ReviewKinds
    audit_action: str
    trace_id: str
    console_reviewer: str
    words: ReviewWords

    model_config = SettingsConfigDict(env_prefix="REVIEW__")


class GuardrailOutcomeSettings(BaseModel):
    """What happened to a checked reply, as a card's result line says it.

    Named by what the customer received: the reply as written, a rewrite, the reply with its
    refuted sentences removed, or nothing of it. The first word is the state a card colours.
    """

    passed: str
    repaired: str
    trimmed: str
    withheld: str


class GuardrailsSettings(YamlBackedSettings):
    """Output/schema guardrail settings."""

    _yaml_section = "guardrails"
    enabled: bool
    mode: Literal["reject", "auto_repair", "constrained"]
    max_repair_attempts: int
    constrained_backend: Literal["grammar", "guided_json"]
    # Why a reply was blocked, by finding code, in the words a card shows.
    block_reasons: dict[str, str] = Field(default_factory=dict)
    outcomes: GuardrailOutcomeSettings

    model_config = SettingsConfigDict(env_prefix="GUARDRAILS__")


class GroundingTierSettings(BaseModel):
    """One stage of the grounding cascade."""

    name: str
    kinds: list[str]
    claims: list[str]
    may_refute: bool


class RefutingPremiseSettings(BaseModel):
    """What a line must look like before it may contradict a claim."""

    min_words: int = Field(ge=1)
    max_capitalised_share: float = Field(gt=0.0, le=1.0)


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


class RecordStatesSettings(BaseModel):
    """The state machine a state claim is judged against (an agent's record, e.g. a refund)."""

    # What the status is the status of, as a rationale says it ("refund").
    subject: str
    ladder: list[str]
    branches: dict[str, str]
    settled_from: str
    observed_as: dict[str, list[str]]
    context_pattern: str
    phrases: list[StatePhraseSettings]


class ConfidenceCardWords(BaseModel):
    """Level 1's wording: its summary and one line per sentence (`confidence.py`)."""

    heading: str
    heading_clean: str
    skipped: str
    no_logprobs: str
    no_match: str
    sentence_line: str
    low_mark: str
    ok_mark: str


class HallucinationMethodWords(BaseModel):
    """What decided a claim, as its card line says it (`guardrails/response.py`)."""

    record: str
    logprobs: str
    restated: str
    restated_source: str
    nothing: str
    nli: str
    # A claim the second-opinion model confirmed: `{model}`, `{prob}`. (`nli`: `{model}`, `{score}`.)
    confirmed: str


class EvidenceSourceNames(BaseModel):
    """How a line of evidence is named on the card and in the log: by its source's prefix,
    else by its kind. `{source}` is the whole source, `{id}` what follows the prefix."""

    prefixes: dict[str, str] = Field(default_factory=dict)
    kinds: dict[str, str] = Field(default_factory=dict)
    default: str


class ReadingWords(BaseModel):
    """How a claim's check steps read on the card and in the log (`hallucination_check/readings.py`)."""

    # The order the check runs in, atop the card: `{nli}`, `{confirmer}`, `{bar}`; `{nli}`.
    rule: str
    rule_nli_only: str
    supported: str
    contradicted: str
    undecided: str
    nothing_related: str
    confirmed: str
    not_confirmed: str
    # A score against the second opinion's bar: `{bar}`.
    at_least: str
    below: str
    # The second opinion chose another answer than "supported": `{answer}`, `{score}`.
    its_choice: str
    # A report's repair: the rewrite (`{role}`), and the sentence taken out.
    rewrite: str
    rewritten: str
    removed: str
    # The line under a rewrite: what the sentence became.
    rewritten_to: str
    # The evidence line of a step that read no line, only a rule: its rationale.
    why: str
    # How an evidence line opens, by what its step decided: supported, contradicted, undecided,
    # confirmed, not_confirmed, rewritten, removed, not_found.
    leads: dict[str, str]


class HallucinationCardWords(BaseModel):
    """The HALLUCINATION CHECK card's shared wording (`hallucination_check/card.py`)."""

    verdict_names: dict[str, str]
    tally: str
    plural: str
    no_claims: str
    lexical: str
    lexical_note: str
    label_names: dict[str, str]
    tier_names: dict[str, str]
    provenance_found: str
    provenance_missing: str
    nothing_searched: str
    retrieval: str
    retrieval_default_method: str


class HallucinationLevelWords(BaseModel):
    """How the hallucination check's levels are named (`hallucination_check/levels.py`)."""

    row: str
    confidence: str
    entailment: str


class ConfidenceSettings(BaseModel):
    """Step 1 of the hallucination check: sentence confidence from token logprobs."""

    enabled: bool
    sentence_mean_prob: float = Field(ge=0, le=1)
    min_token_prob: float = Field(ge=0, le=1)
    max_low_token_share: float = Field(ge=0, le=1)
    ignore_token_pattern: str
    # Tokens that carry grammar, not content ("the", "and", "please"): left out of the
    # content mean, where a low probability means another phrasing, not a guess.
    function_token_pattern: str
    # A sentence whose content tokens average below this is degenerate and is cut.
    degenerate_mean_prob: float = Field(ge=0, le=1)
    # An id, amount, date or number written below this, and confirmed by no record, is cut.
    slot_min_prob: float = Field(ge=0, le=1)
    card: ConfidenceCardWords


class FactAttributeSettings(BaseModel):
    """How a claim talks about one typed fact (`hallucination_check/ledger.py`)."""

    type: Literal["text", "money", "date", "duration", "number", "flag"]
    # Patterns that say a claim is about this attribute ("delivered", "settle").
    terms: list[str] = Field(default_factory=list)
    # A flag: wording that asserts it, and wording that denies it (checked first).
    affirm: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)
    # Whether a reply that got this fact wrong may be corrected with its sentence.
    correctable: bool = True


class FactSettings(BaseModel):
    """The fact ledger: an agent's typed facts, and the words that compare them."""

    # Attribute name -> how a claim states it. The agent's overlay names its own.
    attributes: dict[str, FactAttributeSettings] = Field(default_factory=dict)
    # Operator (`>`, `<`, `>=`, `<=`) -> phrases that state it between two values.
    comparisons: dict[str, list[str]] = Field(default_factory=dict)
    # Duration unit pattern -> seconds ("minutes?" -> 60).
    duration_units: dict[str, int] = Field(default_factory=dict)
    # Attribute name -> phrases that refer to its value without writing it ("the amount I can
    # refund directly" -> the limit), so a comparison against it is computed from the fact.
    references: dict[str, list[str]] = Field(default_factory=dict)
    # The attribute whose fact restates the record's state, for a corrected reply.
    state_attribute: str = ""


class ClaimConfirmerSettings(BaseModel):
    """A second opinion on a claim NLI left undecided: it may confirm, never refute.

    A local decision model reads the claim beside its source's text and answers one choice
    question. Only its "supported" answer, at or above `min_score`, changes anything: the claim
    becomes supported. Contradictions and removals stay with NLI and the deterministic checks.
    """

    enabled: bool
    # The checkpoint (an agent's overlay names it; the platform names none).
    model: str
    subfolder: str
    local_model_dir: str
    device: str
    download_if_missing: bool
    # The question: its id, the state keys the claim and its evidence go under, the
    # instruction, and each option's description. `supported_label` is the one that confirms.
    question_id: str
    claim_key: str
    evidence_key: str
    instructions: str
    criteria: dict[str, str]
    supported_label: str
    min_score: float = Field(ge=0.0, le=1.0)
    # The source text handed over with a claim: the lines most like it, in source order, up to
    # this many characters (the model reads 512 tokens of state and question together).
    max_evidence_chars: int = Field(ge=1)
    # Sources asked, best first, before the claim is left unverified.
    max_sources: int = Field(ge=1)

    @property
    def resolved_local_model_dir(self) -> Path:
        return resolve_path(self.local_model_dir)


class JoinedPremiseSettings(BaseModel):
    """One more NLI call for a claim a tier left undecided: its relevant evidence from one
    source, joined in the order the source gave it. It may support, never refute."""

    enabled: bool
    max_chars: int = Field(ge=1)


class CascadeTierToggles(BaseModel):
    """Per-tier toggles for the NOLLM scoring cascade."""

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


class ScoringCascadeSettings(BaseModel):
    """The NOLLM scoring cascade: which tiers run, and where each one decides."""

    tiers: CascadeTierToggles
    embedding_similarity_threshold: float
    embedding_dissimilarity_threshold: float
    nli_entailment_threshold: float
    judge_confidence_threshold: float
    nli: NliSettings


class HallucinationCheckSettings(YamlBackedSettings):
    """Detection ladder thresholds (LLD)."""

    _yaml_section = "hallucination_check"
    enforce_grounding: bool
    confidence: ConfidenceSettings
    levels: HallucinationLevelWords
    card: HallucinationCardWords
    methods: HallucinationMethodWords
    facts: FactSettings
    # Whether a verifiable claim no evidence grounds is a hallucination (cut and queued),
    # or an abstention left in the reply.
    unsupported_is_hallucination: bool
    # The reply when the gate refuses every sentence of an answer.
    withheld_reply: str
    # The same, by the reason the reply was cut; a reason not listed gets `withheld_reply`.
    withheld_replies: dict[str, str] = Field(default_factory=dict)
    # Whether an uncalibrated detector (the lexical fallback) may refute a claim, or only
    # support it. Off: its contradictions leave the claim unverified.
    uncalibrated_may_refute: bool
    # Whether a claim refuted by a rule (an unsourced figure) still gets the bounded rewrite.
    repair_direct_refutations: bool
    # Words after a figure that make it a size, not an exact amount ("$1.2 M", "4 lakh").
    money_scales: list[str]
    # Connectives that split one sentence into separately checked claims.
    claim_conjunctions: list[str]
    claim_min_content_tokens: int = Field(ge=1)
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
    # Consecutive sentences of one tool result read as one premise as well as one by one
    # (1: one by one only). A passage may support a claim, never refute it.
    passage_window: int = Field(ge=1)
    # Tool results are web pages: wrapped lines rejoined, markdown marks off, a repeated line
    # read once. Off for records, whose every line is a fact as written.
    tool_results_are_pages: bool
    joined_premise: JoinedPremiseSettings
    confirmer: ClaimConfirmerSettings
    # The card's and the log's name for a model, by its id (`model_labels`); unlisted: its
    # id's last part.
    model_labels: dict[str, str] = Field(default_factory=dict)
    readings: ReadingWords
    evidence_sources: EvidenceSourceNames
    # Appended to a kept sentence that holds a claim nothing verified ("" leaves it unmarked).
    undetermined_marker: str
    # What an identifier looks like to this agent. When set, evidence may
    # refute a claim that names identifiers only if it names one of them:
    # NLI reads a record about order B as contradicting a claim about
    # order A. Unset: every retrieved item may refute.
    refute_identifier_pattern: str | None = None
    # A premise that may refute must be a statement: at least `min_words` words, and not mostly
    # capitalised words (a heading, a menu, a row of links). NLI reads "Topics Trending
    # Collections Enterprise" as contradicting anything; a fragment can still support.
    refuting_premise: RefutingPremiseSettings
    # The grounding cascade, strongest first. Each tier names the evidence
    # kinds it reads, the claim classes it may ground, and whether it may
    # refute as well as support. Order is the order credit is given in.
    tiers: list[GroundingTierSettings]
    # How a claim is told apart from conversation and from a promise.
    # Business vocabulary, so an agent overrides it; the platform ships the
    # shape and one worked default.
    conversational_openers: list[str]
    # An offer of help ("If you'd like …, I can help with that"): conversation whatever it names,
    # because it asserts nothing. Matched at the claim's start.
    offer_pattern: str
    record_identifier_pattern: str
    # What a record id is called in a finding or a provenance line ("order 10042").
    record_identifier_label: str
    outcome_pattern: str
    promissory_pattern: str
    # The promissory markers that commit ("will"), as against those that offer ("once", "can").
    commitment_pattern: str
    # Settled-refund wording, the record field that settles it, and the
    # record wording that says it is still waiting on a person.
    refusal_pattern: str
    record_state_pattern: str
    record_states: RecordStatesSettings
    # A phrase carved out of an outcome and checked as the rule it states.
    rule_phrases: list[RulePhraseSettings]
    # A rule claim whose content is this fraction restated by one rule of
    # the tier is supported, whatever the NLI model made of an instruction.
    rule_restatement_coverage: float = Field(default=0.7, ge=0.0, le=1.0)
    # Hallucination repair is deliberately bounded. A model that repeats an
    # unsupported claim must not be given an unbounded generative retry loop.
    max_repair_attempts: int = Field(default=1, ge=0, le=3)
    scoring: ScoringCascadeSettings

    model_config = SettingsConfigDict(env_prefix="HALLUCINATION_CHECK__")


class LedgerSettings(BaseModel):
    """Cumulative-spend ledger that defeats structuring attacks."""

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
    """Semantic advisor — advisory only, structurally unable to block.

    The prompt is configuration because it has to name the kind of agent
    being reviewed — "unusual for a *what*?" is the whole question — and
    the platform does not know what kind of agent it is running.
    """

    enabled: bool
    prompt: str


class SurfacePolicySettings(BaseModel):
    """Surface classification per action."""

    enabled: bool
    surfaces: dict[str, SurfaceClassName]


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

    model_config = SettingsConfigDict(env_prefix="GATEKEEPER__")


class AuthSettings(BaseModel):
    """RBAC + just-in-time privilege elevation."""

    enabled: bool


class PresidioSettings(BaseModel):
    """Presidio's analyzer over a local spaCy pipeline: names and places no regex can find."""

    # The spaCy pipeline, its release, where the release wheel is published (`{model}`,
    # `{version}`), and the repository folder it is unpacked into and loaded from (once, by
    # `make models` or the first load; never fetched again at run time).
    model: str
    version: str
    download_url: str
    download_timeout_seconds: float = Field(gt=0)
    local_model_dir: str
    download_if_missing: bool
    language: str
    # A span Presidio scores below this is not personal data.
    score_threshold: float = Field(ge=0.0, le=1.0)
    # The entity types redacted (Presidio's names). An agent narrows or widens this in its own
    # config: names and places are personal data to one agent and the subject matter of another.
    entities: list[str]
    # The score a configured `patterns` match carries inside Presidio.
    pattern_score: float = Field(ge=0.0, le=1.0)

    @property
    def resolved_local_model_dir(self) -> Path:
        return resolve_path(self.local_model_dir)


class PiiVaultSettings(BaseModel):
    """The reversible map from a redaction token to its value, encrypted, for audit only."""

    # The *name* of the environment variable holding the key (32 bytes, hex or raw) — never
    # the key itself. The key also keys the tokens (HMAC), so they cannot be brute-forced.
    key_env: str
    # Refuse to start without a key; off for development.
    required: bool
    # Vault rows older than this are deleted when a vault opens.
    retention_days: int = Field(ge=1)


class PiiKnownValueSettings(BaseModel):
    """Values the harness read from its own records, found by exact match (`known_values.py`)."""

    # Shorter values are not registered: they could not be told from ordinary words.
    min_chars: int = Field(gt=0)
    # How many values the process remembers; the oldest go first.
    max_entries: int = Field(gt=0)


class PiiSourceSettings(BaseModel):
    """Tokenising at the source (`source.py`): what is remembered, in memory only."""

    # Token → value pairs held so a tool call and a reply can be restored; oldest go first.
    max_tokens: int = Field(gt=0)
    # Texts already tokenised (a prompt part is admitted on every turn): text → result.
    cache_entries: int = Field(gt=0)


class PiiSettings(BaseModel):
    """PII detection/redaction/minimization."""

    # `presidio`: Presidio's NER and recognizers plus the configured patterns; `pattern`: the
    # configured patterns alone (also the fallback when Presidio cannot load).
    engine: Literal["presidio", "pattern"]
    mode: RedactionMode
    mask_token: str
    # `tokenize` writes `[CATEGORY:token]`: the category, then this many characters of the token.
    token_format: str
    token_chars: int = Field(ge=8, le=64)
    boundary_cache_entries: int = Field(gt=0)
    # The deployment's own values that only look like personal data, left as written.
    allow_list: list[str] = Field(default_factory=list)
    patterns: dict[str, str]
    tool_call_allowed_fields: list[str]
    presidio: PresidioSettings
    vault: PiiVaultSettings
    # Recognizers that are a model (NER); every other recognizer is a deterministic rule.
    model_recognizers: list[str]
    known_values: PiiKnownValueSettings
    # Table → column → entity: the columns of an agent's records that hold personal data,
    # registered as known values the moment a query reads them.
    known_columns: dict[str, dict[str, str]] = Field(default_factory=dict)
    source: PiiSourceSettings


class GuardFallbackSettings(BaseModel):
    """One chat model asked in turn when a hosted guard does not answer."""

    model: str
    base_url: str
    endpoint: str
    timeout_seconds: float
    max_tokens: int
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # The dependency name of this fallback's own retry/circuit-breaker stack.
    resilience_key: str


class SecurityGuardModelSettings(BaseModel):
    """Hosted security-model endpoint settings; credentials stay in env vars.

    Everything that names the vendor or the model — the provider, the key's variable, the
    check's name on a card, the verdict's wording — is here, so swapping the guard is a
    config edit and the adapter never names who it calls.
    """

    enabled: bool
    provider: str
    # Bare environment variables holding the key, first present wins.
    key_env_vars: list[str]
    base_url: str
    endpoint: str
    model: str
    timeout_seconds: float
    max_tokens: int = 16
    # Provider request extras merged into the payload, e.g. a chat-template switch that makes a
    # content-safety model name the categories it found.
    extra_body: dict[str, Any] = Field(default_factory=dict)
    # The dependency name of this guard's retry/circuit-breaker stack.
    resilience_key: str
    # The check's name on a card, and what the verdict says (`{env}` the key's variables).
    check_name: str
    flagged_reason: str
    # The flagged reason when the model named what it found (`{categories}`); unset, the plain
    # `flagged_reason` is used. A content-safety model names categories; a jailbreak one does not.
    flagged_categories_reason: str = ""
    # The flagged reason when the model named nothing (`{output}`: its reply, verbatim).
    flagged_uncategorised_reason: str = ""
    allowed_reason: str
    missing_key_reason: str
    # Chat models asked in turn when this guard does not answer; every one unavailable is
    # what the fail-closed policy applies to.
    fallbacks: list[GuardFallbackSettings] = Field(default_factory=list)
    # A chat fallback for a guard whose primary is not a chat model (the jailbreak classifier):
    # the prompt it is asked with, and the JSON keys its verdict and score are read from.
    classifier_prompt: str = ""
    verdict_key: str = ""
    score_key: str = ""

    def as_fallback(self, fallback: GuardFallbackSettings) -> "SecurityGuardModelSettings":
        """These settings, pointed at one fallback model (same key, same wording)."""
        return self.model_copy(update=fallback.model_dump())

    @property
    def api_key(self) -> str:
        """The credential from the first configured bare environment variable."""
        return _first_env(self.key_env_vars) or ""

    @property
    def usable(self) -> bool:
        """Whether this hosted guard can be invoked."""
        return self.enabled and bool(self.api_key)


class RuleCheckNames(BaseModel):
    """The deterministic checks' names on a card: the phrase scan and the term scan."""

    injection: str
    moderation: str


class SecurityGuardSettings(BaseModel):
    """Input/output guard-model configuration."""

    enabled: bool
    fail_open_on_unavailable: bool
    jailbreak: SecurityGuardModelSettings
    content_safety: SecurityGuardModelSettings
    rule_checks: RuleCheckNames
    # What an assessment says when a hosted guard flagged the text, or was unavailable and
    # the policy fails closed (`{guard}` the guard's check name).
    hosted_flagged_reason: str
    # The same, when the guard named categories (`{categories}`).
    hosted_flagged_categories_reason: str
    # A flag in advisory categories only: shown, not blocked (`{guard}`, `{categories}`).
    hosted_advisory_reason: str
    fail_closed_reason: str
    # A verdict a fallback gave (`{reason}`, `{model}` that answered, `{primary}` that did not).
    fallback_reason: str


class PromptInjectionSettings(BaseModel):
    """Instruction/data separation and tool scoping."""

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
    # How the classifier's JSON answer is read: the keys `system_prompt` asks for, and the
    # words that mean safe and unsafe. Agrees with the prompt, so both live here.
    response_safety_key: str
    # Where the categories may be, first found wins: models differ in the key they use, and give
    # a comma-separated string or a list.
    response_categories_keys: list[str]
    safe_word: str
    unsafe_word: str
    # A flag naming only these categories is a warning, not a block: personal data has its own
    # screen (PII redaction), which has already run on the message.
    advisory_categories: list[str]


class ScreeningMessages(BaseModel):
    """What each screen's card says (`security/screening.py`), the same for every agent."""

    pii_started: str
    # `{categories}` the kinds redacted, `{count}` how many spans.
    pii_redacted: str
    # One redaction on the card and in the log: `{category}`, `{detector}`, `{score}`, `{token}`
    # (the masked form has no token).
    pii_entity: str
    pii_entity_masked: str
    pii_clear: str
    # The card's `reason`, as the other screens give one: what was found and by what (`{rule}`
    # rule-based, `{model}` model spans, `{categories}`), nothing found, or the model unavailable.
    pii_reason_found: str
    pii_reason_clear: str
    pii_reason_unavailable: str
    # Shown as the model when detection is patterns alone.
    pii_no_model: str
    injection_started: str
    injection_blocked_rule: str
    injection_blocked_model: str
    injection_unavailable: str
    injection_passed_model: str
    injection_passed_rule: str
    moderation_started: str
    # A flag in advisory categories only (PII/Privacy): shown as a warning, not blocked.
    moderation_advisory: str
    moderation_blocked_rule: str
    moderation_blocked_model: str
    moderation_unavailable: str
    moderation_passed_model: str
    moderation_passed_rule: str
    # A blocked message: the quarantine card's heading, and what the person is told instead.
    quarantined: str
    quarantine_failed: str
    refusal: str


class SecuritySettings(YamlBackedSettings):
    """RBAC, PII, prompt injection, moderation (LLD §12)."""

    _yaml_section = "security"
    auth: AuthSettings
    pii: PiiSettings
    prompt_injection: PromptInjectionSettings
    content_moderation: ContentModerationSettings
    guards: SecurityGuardSettings
    screening: ScreeningMessages

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
    # Requests allowed per calendar month (UTC), across every process; None
    # means no monthly cap. A free tier is a monthly quota, not a rate.
    monthly_quota: int | None = None


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


class SagaSettings(BaseModel):
    """The compensating saga: on, and what a step stopped by the operator is recorded as."""

    enabled: bool
    stopped_reason: str


class ResilienceSettings(YamlBackedSettings):
    """Resilience controls, including durable dead-letter quarantine."""

    _yaml_section = "resilience"
    circuit_breaker: CircuitBreakerSettings
    retry: RetrySettings
    timeout: TimeoutSettings
    bulkhead: BulkheadSettings
    idempotency: EnabledToggle
    saga: SagaSettings
    dead_letter_queue: DeadLetterQueueSettings
    rate_limit: RateLimitSettings

    model_config = SettingsConfigDict(env_prefix="RESILIENCE__")


class JudgeSettings(ModelRef, ModelChainSettings):
    """The LLM judge: which model grades, its backups, and how it samples.

    The judge's model is its own setting, deliberately not the agent's: a
    model grading its own output is the self-preference bias a judge
    exists to avoid. Its backups and protection are a model chain like any
    role's (`ModelChainBuilder`), so a stalled judge call gets an answer.
    """

    temperature: float
    max_tokens: int
    # Cap on the evidence text a judge is shown (0 = all of it).
    max_evidence_chars: int


class JudgeSectionSettings(YamlBackedSettings, JudgeSettings):
    """`judge:` — the platform's one judge model, and how a verdict is asked for."""

    _yaml_section = "judge"
    # How a verdict that fails its schema is treated before the judge abstains.
    validation_mode: Literal["reject", "auto_repair"]
    kappa_acceptable: float = Field(ge=0.0, le=1.0)

    model_config = SettingsConfigDict(env_prefix="JUDGE__")


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
    # An agent's additions to `exclude_paths` (an overlay's list would replace the platform's).
    agent_exclude_paths: list[str] = Field(default_factory=list)
    # Config sections that cannot change agent behaviour.
    exclude_config_sections: list[str]

    @property
    def all_exclude_paths(self) -> list[str]:
        return [*self.exclude_paths, *self.agent_exclude_paths]


class EvalCoreSettings(BaseModel):
    """The core subset `make eval` runs by default; `full` runs everything.

    An agent marks its core golden cases with `tag` (in the case's `tags`), its core attacks
    with `tag` under `attack_metadata_key` in their Promptfoo metadata, and its core DeepEval
    metrics with `core: true`.
    """

    tag: str
    attack_metadata_key: str


class EvalGateSettings(BaseModel):
    """When a run fails, and when the pipeline stops."""

    soft_pass_rate: float


class EvalOverviewSettings(BaseModel):
    """The colour-coded box tables `make eval` prints after each stage."""

    max_failures: int
    reason_chars: int
    # Section key (agent, oracle, deepeval, promptfoo, intent, total) → its name in the summary.
    sections: dict[str, str]
    section_error: str
    interrupted: str
    default_invocation: str
    marked_interrupted: str
    unfinished_section: str
    # Outcome words (`pass`, `fail`, `warn`, `defended`, `breached`, `incomplete`), coloured by
    # `tui.command_output.state_styles`.
    states: dict[str, str]


class EvalReportingSettings(BaseModel):
    """The Markdown report each framework writes per run, and the terminal overview."""

    output_dir: str
    # Framework (`deepeval`, `promptfoo`, `intent`) → file name, with `{run_id}`.
    files: dict[str, str]
    confidence_z: float
    max_cases_listed: int
    excerpt_chars: int
    # Transcript metadata key with the runtime guard's hallucination rate, and the metric
    # whose verdicts are compared with it.
    runtime_guard_key: str
    guard_metric: str
    overview: EvalOverviewSettings

    @property
    def resolved_output_dir(self) -> Path:
        return resolve_path(self.output_dir)


class EvalDeepEvalSettings(BaseModel):
    """DeepEval, run as the golden suite's quality grader through the platform judge."""

    include_reason: bool
    telemetry_opt_out: bool
    max_concurrent: int
    batch_size: int = Field(ge=1)
    geval_params: list[str]
    quiet: bool
    metric_timeout_seconds: float = Field(gt=0)


class EvalPromptfooSettings(BaseModel):
    """Promptfoo, run as a Node tool through npx — never a Python dependency."""

    runner: list[str]
    package: str
    max_concurrency: int
    output_dir: str
    extra_args: list[str]
    # Environment the Promptfoo process (and its Python workers) gets.
    env: dict[str, str]
    # Where each finished attack is ticked (an env var naming a file), and how often it is read.
    progress_env: str
    progress_poll_seconds: float
    # The attack metadata keys (Promptfoo's red-team taxonomy) the report groups by.
    plugin_key: str
    strategy_key: str
    severity_key: str

    @property
    def resolved_output_dir(self) -> Path:
        return resolve_path(self.output_dir)


class EvalSuiteNames(BaseModel):
    """The two suites every agent with evals has."""

    golden: str
    security: str


class EvalsSettings(YamlBackedSettings):
    """The eval framework: oracle, DeepEval and Promptfoo over an agent's suites."""

    _yaml_section = "evals"
    enabled: bool
    # The channel eval traffic is stamped with, so runtime metrics can exclude it.
    channel: str
    # The evaluator's version, recorded with every audit entry.
    version: str
    # `module.path:attribute` of the running agent's EvalPlugin. Empty
    # here: the platform names no agent.
    plugin: str
    run_id_prefix: str
    suites: EvalSuiteNames
    core: EvalCoreSettings
    trials: int = Field(ge=1)
    trial_separator: str
    # The console's log level during `make eval` (the log files are unchanged).
    console_level: str
    # The file (`<name>.log`, in the agent's log folder) every record of an eval run goes to.
    log_file: str
    store: EvalStoreSettings
    generation: EvalGenerationSettings
    fingerprint: EvalFingerprintSettings
    gate: EvalGateSettings
    reporting: EvalReportingSettings
    deepeval: EvalDeepEvalSettings
    promptfoo: EvalPromptfooSettings

    model_config = SettingsConfigDict(env_prefix="EVALS__")


class ToolsSettings(YamlBackedSettings):
    """Tool registry and retrieval-based selection (LLD §19)."""

    _yaml_section = "tools"
    enabled: bool
    selection_k: int
    selection_min_score: float
    # Qualified name → the name the model and the cards use (ADR 0010).
    model_names: dict[str, str]

    model_config = SettingsConfigDict(env_prefix="TOOLS__")


class McpToolEnvelope(BaseModel):
    """The governance a remote server's tools are granted here (never what the server claims)."""

    surface: Literal["locked", "editable", "append_only", "human_controlled"]
    reversible: bool
    side_effecting: bool
    requires_authority_rank: int = Field(ge=1)
    required_permission: str | None = None


class McpServerSettings(BaseModel):
    """One MCP server an agent connects to, and the envelope its tools run under."""

    name: str
    transport: Literal["stdio", "http"] = "stdio"
    # stdio: the process to launch; `{python}` is this interpreter. http: `url`.
    command: list[str] = Field(default_factory=list)
    url: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    namespace: str
    envelope: McpToolEnvelope | None = None
    # A tool's own envelope, by the server's tool name, over the server's.
    tools: dict[str, McpToolEnvelope] = Field(default_factory=dict)


class McpClientSettings(BaseModel):
    """The client side of an MCP session (`mcp/client/sdk_session.py`)."""

    name: str
    version: str
    mode: str
    discover_from: str
    discover_method: str
    initialize_method: str
    log_level: Literal[
        "debug", "info", "notice", "warning", "error", "critical", "alert", "emergency"
    ]
    log_level_meta_key: str
    card_log_levels: list[str]
    failed_log_levels: list[str]


class McpCardWords(BaseModel):
    """How an MCP server is drawn on a card (`mcp/client/presentation.py`)."""

    heading: str
    handshake_title: str
    handshake_lines: list[str]
    server_title: str
    server_lines: list[str]
    client_title: str
    client_lines: list[str]
    capabilities_title: str
    server_capability_line: str
    client_capability_line: str
    no_flags: str
    no_capabilities: str
    client_none: str
    server_side: str
    client_side: str
    instructions_title: str
    no_instructions: str
    tools_title: str
    tool_line: str
    offered_line: str
    argument_line: str
    required: str
    optional: str
    output_line: str
    no_output: str
    hints_line: str
    hint: str
    no_hints: str
    meta_line: str
    governance_line: str
    execution_title: str
    execution_lines: list[str]
    whole_result: str
    result_limit: str
    # The MCP CALL card, one per call (`observability/step_events.py`): any field of
    # the exchange may be named (server, method, tool, duration_ms, is_error).
    call_heading: str
    call_failed_heading: str
    call_raised_heading: str
    connect_failed_heading: str
    tools_changed_heading: str
    server_log_heading: str
    session_failed_heading: str
    error_title: str
    message_title: str


class McpSettings(YamlBackedSettings):
    """MCP servers an agent consumes as tools (`agent_platform.knowledge.mcp`)."""

    _yaml_section = "mcp"
    servers: list[McpServerSettings]
    # Applied to a server that declares no envelope: an undeclared remote tool is
    # treated as a write nobody may undo, so the Gatekeeper sees the worst case.
    default_envelope: McpToolEnvelope
    # Per protocol round trip: a server that stops answering fails the call.
    call_timeout_seconds: float = Field(gt=0)
    python_placeholder: str
    # Characters of a tool's text result handed on; 0 keeps all of it.
    result_text_limit: int = Field(ge=0)
    client: McpClientSettings
    card: McpCardWords

    model_config = SettingsConfigDict(env_prefix="MCP__")


class BoxGlyphSettings(BaseModel):
    """The characters a box-drawn table is drawn with: corners and joins, top to bottom."""

    horizontal: str
    vertical: str
    top: list[str] = Field(min_length=3, max_length=3)
    middle: list[str] = Field(min_length=3, max_length=3)
    bottom: list[str] = Field(min_length=3, max_length=3)


class PresentationSettings(YamlBackedSettings):
    """Plain-text renderers shared by `make` output and the TUI."""

    _yaml_section = "presentation"
    box: BoxGlyphSettings
    # Spaces either side of a cell's text.
    cell_padding: int = Field(ge=0)

    model_config = SettingsConfigDict(env_prefix="PRESENTATION__")


class StatsCommandWords(BaseModel):
    """What `/stats` says (`commands/stats.py`)."""

    summary: str
    usage: str
    title: str
    no_session: str
    tasks: str
    task_line: str
    turns: str
    models: str
    model_line: str
    tools: str
    tool_line: str
    checks: str
    check_line: str
    decisions: str
    slos: str
    slo_line: str
    met: str
    missed: str
    none: str


class AuditCommandWords(BaseModel):
    """What `/audit` says (`commands/inspection.py`)."""

    summary: str
    usage: str
    trace_keyword: str
    list_title: str
    narrowed_by: str
    trace_title: str
    headers: list[str]
    empty: str
    trace_hint: str
    needs_trace_id: str
    unknown_filter: str
    time_format: str
    unknown: str
    rationale_chars: int = Field(ge=8)


class SkillsCommandWords(BaseModel):
    """What `/skills` says (LLD §20)."""

    summary: str
    usage: str
    group: str
    list_title: str
    list_headers: list[str]
    history_title: str
    history_headers: list[str]
    pending_title: str
    empty: str
    no_catalog: str
    unknown: str
    needs_name: str
    description_chars: int = Field(gt=0)
    none: str
    active: str
    inactive: str


class MemoryTablesWords(BaseModel):
    """How `/memory` and `make show-memory` title and head their tables."""

    title: str
    working_title: str
    working_headers: list[str]
    history_key: str
    history_value: str
    episodic_title: str
    episodic_headers: list[str]
    summary_kind: str
    event_kind: str
    semantic_title: str
    semantic_headers: list[str]
    skills_title: str
    skills_headers: list[str]
    inbox_title: str
    inbox_headers: list[str]
    pending: str
    consolidated: str
    empty: str
    episodic_empty: str
    time_format: str
    queue_note: str
    dream_note: str
    inbox_events: int = Field(gt=0)


class CommandSettings(YamlBackedSettings):
    """Console slash commands (LLD §20)."""

    _yaml_section = "commands"
    hidden: list[str] = Field(default_factory=list)
    default_limit: int
    value_preview_chars: int
    # Memory trees (`make show-memory`, `/memory`, `make dream`): how many events
    # each session lists.
    memory_tree_events_per_session: int
    memory_tables: MemoryTablesWords
    recent_window: int
    # What a scoped command says when the session has nobody verified, and
    # when asked for someone else's records (`ConsoleScope`).
    verify_first: str
    not_your_records: str
    audit: AuditCommandWords
    stats: StatsCommandWords
    skills: SkillsCommandWords

    model_config = SettingsConfigDict(env_prefix="COMMANDS__")


class ResponseHighlight(BaseModel):
    """One colouring rule for the final answer: a regex, and the style of what it matches.

    `line`, when set, scopes the rule to lines matching it — a table's
    heading row, say — so a heading word in a data row is left alone.
    """

    pattern: str
    style: str
    line: str | None = None


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
    table_title_style: str
    grid_style: str
    table_header_style: str
    id_style: str
    empty_style: str
    state_styles: dict[str, str]


class RunControl(BaseModel):
    """One run button: its label (`{node}` = where a resume starts) and Textual variant."""

    label: str
    variant: str


class GroundingCardSettings(BaseModel):
    """The HALLUCINATION CHECK card's own words, and how its confidences are coloured."""

    result: str
    confidence: str
    confidence_skipped: str
    claims: str
    corrected: str
    no_evidence: str
    no_evidence_either_way: str
    # Beside a model that read several lines at once (the second opinion): `{count}`.
    lines_read: str
    # The tree under a claim, one line per piece of evidence a step read: `├` before every
    # line but the last, `└` before it. A step that did not decide the claim says what it
    # decided after its reading (`{decision}`); the claim line already carries the verdict.
    step_branch: str
    step_last: str
    step_outcome: str
    # Level 1 in one line atop the card: `{mean}`, `{lowest}`, `{sentence}`.
    logprobs_summary: str
    # The row naming the order the claims were checked in.
    rule: str
    # A sentence's content-token mean at or above `pass` reads as sure, at or above
    # `warn` as hesitant, below as guessed; a token below `warn` is marked in its line.
    confidence_bands: dict[str, float]


class TuiSettings(YamlBackedSettings):
    """Live status feed (LLD §20)."""

    _yaml_section = "tui"
    # The channel this console's turns are logged under.
    channel: str
    # One style per console role (banner, user, assistant, metrics, notice, muted, ...).
    palette: dict[str, str]
    refresh_interval_ms: int
    # A stage's one-line summary for an agent's own reason code (DeterministicCheck's, say).
    reason_summaries: dict[str, str] = Field(default_factory=dict)
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
    # The optional status block an agent may offer (`StatusHost`).
    status_refresh_ms: int
    status_collapsed_symbol: str
    status_expanded_symbol: str
    status_expand_hint: str
    # The run buttons every agent shows: stop while running; resume or abort once stopped.
    run_controls: dict[str, RunControl]
    # The line a resumed turn opens with, where a typed turn shows `USER  <message>`.
    resume_banner: str
    status_history_hint: str
    # The live status line's colour by state: running, complete, failed.
    status_styles: dict[str, str]
    # Shown on a menu of fixed answers; free text can always be typed instead.
    choice_menu_hint: str
    # Event kinds whose `started` event draws a pending card, replaced in place when done.
    pending_card_kinds: list[str]
    # Title colour of a card by outcome (its `status`): started, completed, warning, ...
    outcome_colors: dict[str, str]
    # Style of a card's section titles, its keys and values (`key`, `separator`, `value`,
    # `muted`), its lines by tone (pass, fail, warn, dim), and the words that name a state.
    section_title_style: str
    # What follows a step's duration on a narrative card (`took 1968 ms`).
    timing_unit: str
    # Said on stderr when the agent cannot be built at start-up: `{agent}`, `{error}`, `{logs}`.
    startup_failed: str
    # Said in the feed when a command, or the agent rebuild a command asked for, fails; the
    # agent already on screen keeps running. `{error}` is the exception type and message.
    command_failed: str
    rebuild_failed: str
    body_styles: dict[str, str]
    tone_styles: dict[str, str]
    state_words: dict[str, str]
    grounding_card: GroundingCardSettings
    # How the final answer's light Markdown is coloured (h1, h2, bold, bullet, citation, ...).
    response_styles: dict[str, str]
    # More of the answer coloured by pattern, in order: an agent's own vocabulary ("[must]",
    # its slash commands, a next-step line). The first rule to claim a span wins.
    response_highlights: list[ResponseHighlight]
    command_output: CommandOutputSettings

    model_config = SettingsConfigDict(env_prefix="TUI__")


WebMode = Literal["live", "record", "replay"]


class TavilySettings(BaseModel):
    """The Tavily search API (`POST {base_url}{endpoint}`, bearer auth)."""

    base_url: str
    endpoint: str
    search_depth: Literal["basic", "advanced"]
    # Provider-written summaries are never evidence; asking for one only
    # spends tokens on text the gateway will discard.
    include_answer: bool
    timeout_seconds: float
    # Smallest `time_range` covering a request's `recent_days` (days → name).
    time_ranges: dict[int, str]

    @property
    def api_key(self) -> str:
        """From the bare `tavily_api_key` environment variable, never config."""
        return os.environ.get("tavily_api_key", "")


class DuckDuckGoSettings(BaseModel):
    """DuckDuckGo through the `ddgs` package: no key, no official API, best effort."""

    backend: str
    region: str
    safesearch: str
    timeout_seconds: float
    # Smallest `timelimit` covering a request's `recent_days` (days → d/w/m/y).
    time_limits: dict[int, str]
    # The rate-limit bucket it spends (it does not go through the HTTP transport).
    rate_bucket: str


SearchProviderName = Literal["tavily", "duckduckgo"]


class SearchAugmentSettings(BaseModel):
    """When a free answer is weak, also ask the metered provider — within a paced allowance."""

    enabled: bool = True
    # The metered provider asked, and the rate-limit bucket whose monthly quota paces it.
    provider: SearchProviderName
    quota_bucket: str
    # Weak: fewer hits than this, or less than this share of the query's content words in the
    # top `top_k` titles and snippets. Words shorter than `min_term_chars` are not counted.
    min_hits: int = Field(ge=1)
    min_term_coverage: float = Field(ge=0.0, le=1.0)
    top_k: int = Field(ge=1)
    min_term_chars: int = Field(ge=1)
    # The share of the monthly quota augmentation may spend, paced evenly over the month; the
    # rest is kept for pure fallback and for runs that search the web.
    share: float = Field(ge=0.0, le=1.0)


class WebSearchSettings(BaseModel):
    """Which providers answer a search, in order."""

    # General searches: the first provider that answers with hits wins.
    providers: list[SearchProviderName]
    # Searches bounded in time or by topic (news) prefer these, in order.
    time_bounded_providers: list[SearchProviderName]
    max_results: int = Field(ge=1)
    augment: SearchAugmentSettings


class WebFetchSettings(BaseModel):
    """Plain HTTP fetching, and the reader fallback for script-rendered pages."""

    timeout_seconds: float
    # Attempts per page through the fetch's resilience stack: a dead page costs at most
    # `max_attempts` × `timeout_seconds` (plus backoff).
    max_attempts: int = Field(ge=1)
    user_agent: str
    max_chars: int = Field(ge=1)
    # Below this much extracted text a page is treated as script-rendered
    # and re-read through the reader service.
    min_text_chars: int = Field(ge=0)
    js_markers: list[str]
    reader_enabled: bool
    # `{url}` is replaced by the page's URL (a Jina-Reader-style service).
    reader_url_template: str
    # Where a page declares its own publication date: `<meta>` names (in priority order)
    # and JSON-LD keys. Search hits outside a news vertical carry no date.
    date_meta_keys: list[str]
    date_json_keys: list[str]
    # A page's chrome, dropped with everything inside it: menus, footers, sidebars, forms. Its
    # link rows otherwise read as one line of evidence ("Topics Trending Collections").
    chrome_tags: list[str]
    # Tags that end a line, so items of a list are separate lines rather than one run.
    block_tags: list[str]


class WebSettings(YamlBackedSettings):
    """The web gateway: live, recording or replaying (spec §3.3)."""

    _yaml_section = "web"
    mode: WebMode
    fixtures_dir: str
    fixture_set: str
    search: WebSearchSettings
    fetch: WebFetchSettings
    tavily: TavilySettings
    duckduckgo: DuckDuckGoSettings

    model_config = SettingsConfigDict(env_prefix="WEB__")

    @property
    def resolved_fixtures_dir(self) -> Path:
        return resolve_path(self.fixtures_dir)


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
    agent: AgentIdentitySettings = Field(default_factory=lambda: AgentIdentitySettings())  # type: ignore[call-arg]
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
    skills: SkillSettings = Field(default_factory=lambda: SkillSettings())  # type: ignore[call-arg]
    guardrails: GuardrailsSettings = Field(default_factory=lambda: GuardrailsSettings())  # type: ignore[call-arg]
    review: ReviewSettings = Field(default_factory=lambda: ReviewSettings())  # type: ignore[call-arg]
    hallucination_check: HallucinationCheckSettings = Field(
        default_factory=lambda: HallucinationCheckSettings()  # type: ignore[call-arg]
    )
    gatekeeper: GatekeeperSettings = Field(default_factory=lambda: GatekeeperSettings())  # type: ignore[call-arg]
    security: SecuritySettings = Field(default_factory=lambda: SecuritySettings())  # type: ignore[call-arg]
    resilience: ResilienceSettings = Field(default_factory=lambda: ResilienceSettings())  # type: ignore[call-arg]
    durability: DurabilitySettings = Field(default_factory=lambda: DurabilitySettings())  # type: ignore[call-arg]
    loop_control: LoopControlSettings = Field(default_factory=lambda: LoopControlSettings())  # type: ignore[call-arg]
    money: MoneySettings = Field(default_factory=lambda: MoneySettings())  # type: ignore[call-arg]
    judge: JudgeSectionSettings = Field(
        default_factory=lambda: JudgeSectionSettings()  # type: ignore[call-arg]
    )
    evals: EvalsSettings = Field(default_factory=lambda: EvalsSettings())  # type: ignore[call-arg]
    tools: ToolsSettings = Field(default_factory=lambda: ToolsSettings())  # type: ignore[call-arg]
    tui: TuiSettings = Field(default_factory=lambda: TuiSettings())  # type: ignore[call-arg]
    commands: CommandSettings = Field(default_factory=lambda: CommandSettings())  # type: ignore[call-arg]
    presentation: PresentationSettings = Field(
        default_factory=lambda: PresentationSettings()  # type: ignore[call-arg]
    )
    web: WebSettings = Field(default_factory=lambda: WebSettings())  # type: ignore[call-arg]
    analytics: AnalyticsSettings = Field(default_factory=lambda: AnalyticsSettings())  # type: ignore[call-arg]
    mcp: McpSettings = Field(default_factory=lambda: McpSettings())  # type: ignore[call-arg]
    # Which capabilities this deployment was given. Declared by the agent
    # rather than the platform, and merged here by the same overlay chain
    # as everything else — so an episode narrows the posture by declaring
    # a `features:` block, not by calling a setter.
    features: FeatureSet = Field(default_factory=lambda: FeatureSet(tree=load_section("features")))

    # No `env_file`: `.env` is loaded into the process environment
    # (`config/environment.py`), where every section and bare key reads it.
    model_config = SettingsConfigDict()


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
        load_environment()
        _config_instance = AppConfig()
    return _config_instance


def reload_settings() -> AppConfig:
    """Drop the cached singleton and re-read config.yaml from disk."""
    global _config_instance
    _config_instance = None
    return get_settings()
