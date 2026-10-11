"""Configuration for the research agent.

These settings live with the agent, not with the platform: the section is
declared in this package's overlay and read here, so `agent_platform`
never learns what a research run is. Every tunable the agent reads is a
field here — nothing is a literal in the code that uses it.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from agent_platform.config.paths import resolve_path
from agent_platform.config.settings import YamlBackedSettings
from agent_platform.loop_control.types import WatchdogLayer
from research_agent.domain.budget import Budget
from research_agent.domain.scope import Depth


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
    research: str


class MessageSettings(BaseModel):
    """Every agent-authored sentence the user sees (none are model-written)."""

    refuse: str
    # A message one of the input screens (injection, moderation) stopped.
    challenge_prefix: str
    # `{plan}`: the draft plan, one "  - step" line per sub-question.
    propose_done: str
    # One step of a draft plan built from the interview's answers: `{part}`.
    plan_step: str
    # The last line of a turn that waits on the user: what to do next. Each starts with
    # `next_step_marker`, which the console colours.
    next_step_marker: str
    next_step_proposed: str
    next_step_rubric: str
    # The menu offered beside them.
    done_choice: str
    approve_choice: str
    # The rubric as the user reviews it: `{id}`, `{gate}`, `{text}`, `{kind}`, `{check}`,
    # `{threshold}`, `{weight}`, `{origin}`; `{gate}` is `rubric_must` or `rubric_soft`.
    rubric_heading: str
    rubric_line: str
    rubric_must: str
    rubric_soft: str
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
    # `{error}`: why a scoping step failed (the session is left ready for a new question).
    scope_failed: str
    # What working memory keeps of a delivered run: `{run_id}`, `{status}`, `{topic}`.
    run_note: str
    brief_ready: str


class GreetingSettings(BaseModel):
    """What the greeting says (console/greeting.py)."""

    # The whole greeting, one line.
    line: str


class SetupMessages(BaseModel):
    """What start-up says when the live posture is missing a credential."""

    # Fatal: a routed role needs a model key that is not set.
    missing_model_key: str
    # Not fatal: without a Tavily key, search is DuckDuckGo only.
    no_search_key: str


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
    # Units counted on the calendar for a period's start ("the last 20 years" on 2026-10-04
    # starts 2006-10-04, not 7,300 days back): months per unit.
    unit_months: dict[str, int]


class ScopingSettings(BaseModel):
    """The interview: its turn cap, its guard, and how answers are read."""

    max_turns: int = Field(ge=1)
    challenge_every_k_turns: int = Field(ge=1)
    checklists_path: str
    refuse_patterns: list[str]
    archetype_keywords: dict[str, list[str]]
    # The cue rules name a research kind only when at least this many of that kind's cues
    # appear (and no other kind ties it); otherwise the encoder, then the LLM.
    kind_rule_min_cue_hits: int = Field(ge=1)
    time_window: TimeWindowSettings
    answer_split_pattern: str
    vague_answer_max_words: int = Field(ge=0)
    absolute_words: list[str]
    # An answer to a subject item (`subject_fields`) made only of these words names nothing to
    # research: the item stays open; after `vague_answers_before_delegating` such answers the
    # interviewer proposes the values itself, from the topic.
    vague_subject_words: list[str]
    subject_fields: list[str]
    vague_answers_before_delegating: int = Field(ge=1)
    # How each archetype's rubric will check the answers, told to the interviewer so it asks
    # for what can be checked.
    rubric_shapes: dict[str, str]
    waive_all_keyword: str
    any_recency_words: list[str]
    none_answers: list[str]
    # What an optional item the interviewer skipped is recorded as (one of `none_answers`).
    skipped_answer: str
    primary_sources_phrase: str
    # "at least 3 independent domains": how many independent websites the answer must rest on.
    distinct_domains_pattern: str
    # Words that pick a size, and the pattern for "N pages".
    depth_words: dict[Depth, list[str]]
    pages_pattern: str
    # The item whose answer becomes the draft plan when the interviewer proposes none.
    plan_item: str
    # A reply that agrees to what was proposed ("yes", "looks good"): after the draft plan it
    # closes the interview, in rubric review it approves. Matched whole, case-insensitively.
    affirmations: list[str]

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
    # Sub-questions are shown as Q1, Q2, ...; the planner's own ids are replaced by these.
    node_id_prefix: str
    # Rubric checks whose must-pass items some sub-question must serve.
    covered_checks: list[str]


class MatchingSettings(BaseModel):
    """Deterministic term matching (domain/matching.py)."""

    token_pattern: str
    stem_chars: int = Field(ge=1)
    min_token_chars: int = Field(ge=1)
    stopwords: list[str]
    synonyms: dict[str, list[str]]


class SnippetEvidenceSettings(BaseModel):
    """A page that could not be read leaves its search snippet as weaker, labelled evidence."""

    enabled: bool
    # The rejection codes whose snippet is admitted in the page's place (blocked, error, …).
    # Never an injection: a page withheld for attacking the model is not cited at all.
    codes: list[str]
    # A snippet sentence shorter than this many words is not a claim.
    min_words: int = Field(ge=1)
    # The entry's caveat, as the writer and the reader see it.
    caveat: str


class RescueSettings(BaseModel):
    """One more round for a run whose ledger is still empty when it would stop."""

    enabled: bool
    # Granted on top of the size's budget, for this round only.
    searches: int = Field(ge=1)
    fetches: int = Field(ge=0)
    # Caveats on what the relaxed window admits: `{days}`, `{date}`; `{days}`.
    outside_window: str
    undated: str
    # A best-effort report's entries (search results only), and the line atop its report.
    best_effort_caveat: str
    best_effort_note: str


class ResearchSettings(BaseModel):
    """The researcher worker and its defences."""

    max_results_per_search: int = Field(ge=1)
    max_fetches_per_node: int = Field(ge=1)
    max_claims_per_page: int = Field(ge=1)
    # How much of a page the extractor reads; the rest is not sent to the model.
    extractor_page_chars: int = Field(ge=1000)
    # The page is cut into windows of this size and the ones sharing most terms with the
    # sub-question are kept (in page order) up to `extractor_page_chars`.
    extractor_window_chars: int = Field(ge=200)
    # Pages read at once across the run (fetch, screen, extract), on top of node concurrency.
    page_concurrency: int = Field(ge=1)
    primary_source_markers: list[str]
    outbound_allow_patterns: list[str]
    outbound_deny_patterns: list[str]
    authority_rank: str
    # Why a search hit outside the approved recency window was not read.
    recency_rejection: str
    # Host suffixes by kind of source ("press release": [prnewswire.com, ...]). A labelled
    # source is shown with its label and fetched after unlabelled ones; never dropped.
    source_labels: dict[str, list[str]] = Field(default_factory=dict)
    # A hit with no date from the provider: "verify_on_page" reads it and checks the date
    # the page declares; "reject" drops it unread.
    undated_policy: Literal["verify_on_page", "reject"] = "verify_on_page"
    undated_rejection: str = (
        "no readable publication date (got '{date}'), so it cannot be shown to fall "
        "inside the {days}-day window"
    )
    # A question bounded to this many days or fewer is news:
    # search the providers' news vertical (`news_topic`).
    news_window_days: int = Field(ge=1)
    news_topic: str
    # The budget stop recorded when every search provider is out of quota.
    quota_stop: str
    snippet_evidence: SnippetEvidenceSettings
    rescue: RescueSettings


class CallSettings(BaseModel):
    """How model calls are capped, and what happens when one is cut off.

    The roles run on a reasoning model whose hidden reasoning is spent from the same output
    cap as the answer, so the platform default (`llm.max_tokens`) can leave no room for the
    JSON: the call ends `length` with empty content. A role's own cap fixes the common case;
    a cut-off call is retried with a larger one before it is given up on.
    """

    # Output-token cap per role; a role not listed uses the provider's default.
    role_max_tokens: dict[str, int]
    # Roles whose CONTEXT cards are drawn as one per batch rather than one per request.
    context_aggregate_roles: list[str]
    # A cut-off call is retried this many times, each with the cap multiplied by the factor.
    truncation_retries: int = Field(ge=0)
    truncation_retry_factor: float = Field(gt=1)
    # Timeouts, retries and backup models are not here: they are each role's
    # `llm.roles.<role>` chain, applied by the platform (`ModelChainBuilder`).


class DepthProfile(BaseModel):
    """One size of run: what it may spend, and how much report it writes."""

    budget: Budget
    # Ledger entries the synthesizer is shown (spread across sub-questions).
    report_entries: int = Field(ge=1)
    # Output-token cap for the synthesizer call.
    report_max_tokens: int = Field(ge=1)


class ReflectionSettings(BaseModel):
    max_followups_per_round: int = Field(ge=0)
    gap_fill_round_offset: int = Field(ge=1)
    # Checks a failure of which is the report's writing, not missing evidence: the audit sends
    # them to a revision of the report rather than to more research.
    writing_checks: list[str]
    # Revisions of one report, at most; and the least gain in audit score a revision must
    # bring for another to be tried.
    max_revisions: int = Field(ge=0)
    min_score_gain: float = Field(ge=0.0, le=1.0)


class ReflexionSettings(BaseModel):
    """Lessons across runs (`reflexion/lessons.py`): how many, where kept, how recalled."""

    max_lessons: int = Field(ge=1)
    recall_limit: int = Field(ge=1)
    # How many episodic matches to read before filtering to this run's kind.
    search_limit: int = Field(ge=1)
    min_term_chars: int = Field(ge=1)
    # Rejection codes named in one lesson, most frequent first.
    top_codes: int = Field(ge=1)
    category: str
    session: str
    # Lesson templates by kind: failed, rejected, budget, revised.
    wording: dict[str, str]


class SynthesisSettings(BaseModel):
    citation_pattern: str
    entry_id_prefix: str
    templates_path: str
    # The citation guard's rewrites of unsupported sentences before a sentence is removed
    # (0: remove without asking).
    max_guard_rewrites: int = Field(ge=0)
    # A sentence about the research rather than its subject (an evidence id outside a citation,
    # "search snippets only", "the evidence does not cover"): the run's working notes, removed
    # from the report in code. What was not found is Coverage & Limits' to say.
    working_note_patterns: list[str]

    @property
    def resolved_templates_path(self) -> Path:
        return resolve_path(self.templates_path)


class ReaderViewWording(BaseModel):
    """What the reader's view of a report says that no model wrote."""

    # `{status}`, `{count}`, `{noun}`.
    status_line: str
    source_singular: str
    source_plural: str
    no_evidence: str
    sources_heading: str
    # `{number}`, `{host}`, `{label}`, `{url}`, `{date}`; `{label}` is " (press release)" or "".
    source_line: str
    unknown_date: str
    not_established_heading: str
    # `{text}`, `{tried}`; `{note}`.
    not_established_line: str
    nothing_tried: str
    stopped_early: str
    # `{hosts}`, `{more}`; `{count}`.
    not_read: str
    more_hosts: str
    max_not_read: int = Field(ge=1)
    # A rejection code in words; anything unlisted reads `not_used`.
    reasons: dict[str, str]
    not_used: str


class AuditSettings(BaseModel):
    """How independent the Independent Audit's judge must be, and what its checks read."""

    independent_judge: bool
    # `platform`: the platform's configured NLI (local cross-encoder, hosted,
    # or lexical fallback); `lexical`: the deterministic fallback, always.
    entailment_model: Literal["platform", "lexical"]
    # What counts as a named entity in a claim (distinct-subject checks).
    entity_pattern: str
    # The weighted score a report needs for a PASS, and the floor below which it FAILs;
    # between them it is PARTIAL (`audit/verdict.py`). A failed must-pass is never a PASS.
    pass_threshold: float = Field(gt=0, le=1)
    partial_threshold: float = Field(gt=0, le=1)


class InjectionModelSettings(BaseModel):
    """The hosted injection model that backs up the phrase check on fetched pages."""

    # A page the phrase check flags is dropped only if the model agrees (score >= threshold).
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    # Also score pages the phrase check passed (costs model calls; off by default).
    scan_all: bool = False
    # Models read a bounded window; flagged phrases are scored in their own context.
    window_chars: int = Field(default=2000, ge=200)
    max_windows: int = Field(default=4, ge=1)


class KindEncoderSettings(BaseModel):
    """The nearest-example classifier in front of the LLM research-kind call: when it decides.

    A `/research` question is compared with the labelled examples (`exemplars_path`, a file of
    `kind: [messages]`) and the nearest `neighbours` vote. It decides alone only above both bars;
    below either, the LLM decides.
    """

    exemplars_path: str
    neighbours: int = Field(ge=1)
    min_confidence: float = Field(ge=0.0, le=1.0)
    min_margin: float = Field(ge=0.0, le=1.0)


class AnswerBudget(BaseModel):
    """What one direct answer may spend; the ReAct loop stops at the first cap reached."""

    max_steps: int = Field(ge=1)
    max_tool_calls: int = Field(ge=0)
    max_total_tokens: int = Field(ge=1)
    max_seconds: float = Field(gt=0)


class AnswerMessages(BaseModel):
    """What the direct-answer path says that no model wrote."""

    # `{counter}` names the budget counter that ran out (steps, tool_calls, tokens, seconds).
    cap_reached: str
    # `{cause}`, `{step}` and `{seconds}`: why the answer could not be given, and when.
    failed: str
    # `{seconds}`: a model call that did not return in time.
    model_timeout: str
    empty: str
    unknown_tool: str
    bad_arguments: str
    tool_failed: str
    # One loop step's description beside its number: `{step}`, `{max}`.
    loop_step_detail: str
    # LOOP CONTROL: the next step's words, by node, and why a budget forced it, by counter.
    loop_next: dict[str, str]
    loop_because: dict[str, str]
    # THOUGHT: `{calls}` the tools it chose, `{tokens}`, `{seconds}`.
    # The pending THOUGHT card while its model call runs: `{role}`.
    thought_waiting: str
    thought_tools: str
    thought_answer: str
    # REVISE: `{claims}` the unsupported claims it was asked to fix, `{tokens}`, `{seconds}`.
    revise_heading: str
    # ACTION: `{tool}` and `{arguments}`.
    tool_call: str
    # OBSERVATION: `{tool}`, `{lines}`; or `{tool}`, `{error}`.
    observation_heading: str
    observation_heading_via: str
    observation_failed: str
    # ANSWER: `{steps}`, `{calls}`, `{tokens}`, `{seconds}`.
    answered: str
    no_results: str
    sources_heading: str
    # `{when}`: the moment a search or a page was retrieved; `{date}`: a hit's own publish date.
    retrieved: str
    published: str
    # REFLECT: `{tally}` the claim verdicts; `{removed}` the details taken out.
    grounded_heading: str
    ungrounded_heading: str
    # What happened to the answer, in one line.
    outcome_unchanged: str
    outcome_repaired: str
    outcome_filtered: str
    outcome_withheld: str
    # Sent with `{count}` sentence(s) marked as unverified (nothing in the sources decided them).
    outcome_marked: str
    marked_heading: str
    # Appended when details were removed; and the whole reply when nothing could be verified.
    filtered_note: str
    withheld: str
    # The context handed to the model, as system-prompt blocks.
    session_state_heading: str
    state_line: str
    earlier_heading: str
    # `{run_id}`, `{status}`, `{topic}`.
    earlier_line: str
    # The THOUGHT card's context lines: `{turns}`; `{runs}`.
    context_turns: str
    context_runs: str
    # Tool messages: `{timezone}`; `{location}`; `{status}`; `{why}`.
    unknown_timezone: str
    no_place: str
    page_unreadable: str
    page_withheld: str
    # The weather report: `{where}`, `{url}`, `{time}`; `{conditions}`; `{day}`.
    weather_header: str
    weather_now: str
    weather_day: str


class PointwiseSettings(BaseModel):
    """A prose answer of `min_sentences` or more becomes one point per sentence."""

    enabled: bool
    marker: str
    min_sentences: int = Field(ge=2)


class AnswerContextSettings(BaseModel):
    """What the direct answer knows about the session besides the question."""

    # Recent turns of this session handed to the model (also the working memory's bound).
    history_turns: int = Field(ge=0)
    # Earlier archived runs scanned for ones that bear on the question, and how many are shown.
    recall_scan: int = Field(ge=0)
    recall_limit: int = Field(ge=0)
    # Words shorter than this do not count when matching a question to an earlier run.
    min_word_chars: int = Field(ge=1)
    # Words a question must share with an earlier run's topic for the run to be offered.
    min_overlap: int = Field(ge=1)


class WeatherSettings(BaseModel):
    """Open-Meteo: a keyless, structured forecast API (geocoding, then the forecast)."""

    geocode_url: str
    forecast_url: str
    timeout_seconds: float = Field(gt=0)
    # Place-name matches to read when geocoding (the first is used).
    geocode_results: int = Field(ge=1)
    forecast_days: int = Field(ge=1)
    current_fields: list[str]
    daily_fields: list[str]
    # WMO weather interpretation codes, in words.
    codes: dict[int, str]
    unknown_code: str
    # Each API field as the report words it ("temperature_2m_max" -> "high").
    field_labels: dict[str, str] = Field(default_factory=dict)


class AnswerGroundingSettings(BaseModel):
    """Checking the answer against what the tools returned, before the user sees it."""

    # A name: two or more capitalised words in a row. One the tools never returned (and the
    # question did not contain) is invented, so its sentence is removed.
    entity_pattern: str
    # A name is invented when more than this share of its words appears in no source.
    name_unknown_share: float = Field(ge=0.0, lt=1.0)
    # The answer's own "Sources:" block: cited URLs, not claims, so it is not checked.
    sources_pattern: str
    # Evidence lines handed to the rewrite for each refuted claim (the closest by shared words).
    repair_evidence_per_claim: int = Field(ge=1)
    # Cited URLs the loop saw only as a search snippet, fetched in full to ground the answer:
    # at most this many, within this many seconds (never more than the turn has left).
    cited_fetch_max: int = Field(ge=0)
    cited_fetch_seconds: float = Field(gt=0)


class RecencyMessages(BaseModel):
    """What the model is told when a search is pinned to the past or finds only old results."""

    pinned: str
    stale: str
    # A planned query naming years before the brief's resolved period: `{years}`, `{start}`, `{end}`.
    outside_period: str


class RecencySettings(BaseModel):
    """An undated question asks for the latest: the check that holds searches to that.

    A model pins a query to the edition it knows from training ("... 2023") although the prompt
    says what today is. A prompt rule alone does not stop it; this check reads the query and the
    results, and adds a note the model reads before it answers.
    """

    enabled: bool = True
    # A four-digit year in a question or a query.
    year_pattern: str
    # An undated question whose newest dated result is older than this gets the stale note.
    stale_after_days: int = Field(ge=1)
    messages: RecencyMessages


class AnswerSettings(BaseModel):
    """The direct-answer path: a budgeted ReAct loop over a few read-only tools."""

    role: str
    budget: AnswerBudget
    # One model call may take this long; below the provider's own timeout, so a stalled call
    # fails here, named, instead of consuming the whole budget in silence.
    call_timeout_seconds: float = Field(gt=0)
    # The grounding rewrite's own time limit: it is never squeezed to what the loop left.
    repair_timeout_seconds: float = Field(gt=0)
    # A step is not started with less time than this left; the loop answers with what it has.
    min_call_seconds: float = Field(gt=0)
    # Tries for a text answer on the last step before answering from the observations alone.
    final_attempts: int = Field(ge=1)
    # Output-token cap for one model response.
    max_tokens: int = Field(ge=1)
    temperature: float = Field(ge=0.0)
    search_max_results: int = Field(ge=1)
    # A search hit's snippet is the page's own text; this much of it is shown to the model.
    snippet_max_chars: int = Field(ge=100)
    # How much of a fetched page the model is shown.
    fetch_max_chars: int = Field(ge=200)
    # A page fetch for a quick answer: shorter and with fewer attempts than a research run's, so
    # one slow site cannot spend half the turn (the model reads another result instead).
    fetch_timeout_seconds: float = Field(gt=0)
    fetch_max_attempts: int = Field(ge=1)
    # The Executor's deadline for one answer tool (a search, a fetch, a forecast).
    tool_timeout_seconds: float = Field(gt=0)
    # strftime format for the date the prompt carries ("Today is ...").
    date_format: str
    # strftime format of the clock tool's answer.
    clock_format: str
    context: AnswerContextSettings
    # The answer as points when the model wrote prose (`graph/points.py`).
    pointwise: PointwiseSettings
    grounding: AnswerGroundingSettings
    weather: WeatherSettings
    messages: AnswerMessages


class ArchiveSettings(BaseModel):
    # Remember each run as a semantic fact too. The episodic index row is the run pointer;
    # a "fact" about a report is not something the agent believes true.
    semantic_pointer: bool = False
    run_id_prefix: str
    history_limit: int = Field(ge=1)
    # `make runs ra`: the archive read straight from the artifact store.
    list_headers: list[str]
    list_empty: str


class CancelSettings(BaseModel):
    key: str


class ActionSpec(BaseModel):
    """One button: its label, the command it runs (a `command_names` key) and any argument."""

    label: str
    command: str
    argument: str = ""
    # May be pressed while a turn is running (cancel); the others wait for the turn to end.
    during_turn: bool = False
    # Textual button variant: how the button is drawn (success, primary, error, ...).
    variant: str = "default"


class PanelSettings(BaseModel):
    """The console's presentation of this agent."""

    stage_labels: dict[str, str]
    command_help: dict[str, str]
    history_headers: list[str]
    # Every sentence a card or the status block says (`research_agent.presentation.narration`).
    cards: dict[str, str]
    # Rejection and gap codes, in words.
    reasons: dict[str, str]
    # How much of a claim or question a list line shows before an ellipsis.
    clip_chars: int = Field(ge=20)
    # Buttons offered beside the slash commands, by session phase.
    actions: dict[str, list[ActionSpec]]


class TurnSummarySettings(BaseModel):
    """How a turn's summary card reads, from the phase the session is left in."""

    # Phase -> the outcome in words.
    outcomes: dict[str, str]
    # Phases that leave the session waiting on the person, and that mean failure.
    waiting: list[str]
    failed: list[str]
    # The outcome when the turn raised.
    crashed: str


class ResearchTaskSettings(BaseModel):
    """This agent's task kinds and the outcomes it closes them with (`analytics.tasks`)."""

    research_run: str
    direct_answer: str
    # A finished run's status (`RunStatus`) → its outcome.
    run_status: dict[str, str]
    scope_refused: str
    answered: str
    answer_failed: str
    refused: str

    def kinds(self) -> list[str]:
        return [self.research_run, self.direct_answer]

    def outcomes(self) -> list[str]:
        """Every outcome this agent closes a task with, for start-up validation."""
        return [
            *self.run_status.values(),
            self.scope_refused,
            self.answered,
            self.answer_failed,
            self.refused,
        ]


class RunLoopWords(BaseModel):
    """LOOP CONTROL for a research run: the next step's words, and why, by key."""

    next: dict[str, str]
    because: dict[str, str]
    # The next steps that end the research loop (its LOOP CONTROL verdict is FINALIZE).
    stops: list[str]


class InstructionSettings(BaseModel):
    """Which roles carry the profile and their skills (ADR 0010, ra-0021)."""

    profile_roles: list[str]
    # How a role is written as a skill signal (`role:{role}`).
    role_signal: str


class ResearchAgentSettings(YamlBackedSettings):
    """The research agent's own wiring values."""

    _yaml_section = "research_agent"
    instructions: InstructionSettings
    # The tasks this agent opens and the outcomes it closes them with (ADR 0009).
    tasks: ResearchTaskSettings

    title: str
    placeholder: str
    greeting: GreetingSettings
    command_names: CommandNames
    messages: MessageSettings
    turn_summary: TurnSummarySettings
    setup: SetupMessages
    scoping: ScopingSettings
    rubric: RubricSettings
    budget: Budget
    budget_layers: dict[str, WatchdogLayer]
    depths: dict[Depth, DepthProfile]
    planning: PlanningSettings
    matching: MatchingSettings
    research: ResearchSettings
    calls: CallSettings
    run_loop: RunLoopWords
    reflection: ReflectionSettings
    reflexion: ReflexionSettings
    synthesis: SynthesisSettings
    audit: AuditSettings
    injection_model: InjectionModelSettings = Field(default_factory=InjectionModelSettings)
    kind_encoder: KindEncoderSettings
    answer: AnswerSettings
    recency: RecencySettings
    # The reply shows numbered sources and one finding per line; the archived report keeps
    # the evidence ids. Off: the raw report text.
    reader_view: bool = True
    reader_view_wording: ReaderViewWording
    archive: ArchiveSettings
    cancel: CancelSettings
    tui: PanelSettings

    model_config = SettingsConfigDict(env_prefix="RESEARCH_AGENT__")


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
