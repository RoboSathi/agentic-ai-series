"""The composition root: the one place features are read and classes chosen.

Every feature is read here, once, from the posture handed in (or the
declared configuration). A withheld feature is a collaborator that is
never built — no challenge guard, no rubric generator, no classifier, no
quarantined extractor, no independent-audit node in the compiled graph — never a
component that runs and reports having done nothing.

Anything passed in is used as-is; anything omitted is built from config.
Tests pass a scripted model, a replaying gateway and in-memory stores and
get the real wiring of everything else.
"""

import asyncio
import re
from collections.abc import Callable
from functools import cache
from typing import Any

import yaml

from agent_platform.audit import AuditLedger, create_audit_ledger
from agent_platform.config import FeatureSet, active_features, get_settings, set_active_features
from agent_platform.config.paths import resolve_path
from agent_platform.config.settings import WebSettings
from agent_platform.context import (
    CandidateSource,
    ContextWindow,
    CrossEncoderReranker,
    FiveStagePipeline,
    WorkingMemoryContextSource,
)
from agent_platform.context.factory import ContextManagement, build_context_management
from agent_platform.core.persistence import SqliteConnectionPool, SqlitePlatformStore
from agent_platform.durability import (
    DurableGraphRunner,
    RunRegistry,
    create_checkpointer,
    model_types,
)
from agent_platform.durability.checkpointer import SqliteCheckpointer
from agent_platform.guardrails.guards import PydanticValidator
from agent_platform.guardrails.response import ResponseGuardrail
from agent_platform.hallucination_check.checker import DetectionLadder
from agent_platform.hallucination_check.confidence import ConfidenceCheck
from agent_platform.hallucination_check.confirmer import create_claim_confirmer
from agent_platform.hallucination_check.evidence import (
    EvidenceLedger,
    NonEmptySpanValidator,
    VerbatimSpanValidator,
)
from agent_platform.intent.exemplar import ExemplarIntentClassifier
from agent_platform.knowledge.web import WebGateway, create_web_gateway
from agent_platform.llm.embeddings import EmbeddingClient, create_embedding_client
from agent_platform.llm.http import create_http_client
from agent_platform.llm.local_embeddings import LocalSentenceEmbedder
from agent_platform.llm.local_reranker import create_reranker
from agent_platform.llm.observer import ModelCallObserver
from agent_platform.llm.roles import RoleModelRouter
from agent_platform.logging import get_logger
from agent_platform.loop_control import BudgetGuard
from agent_platform.memory import (
    EpisodicStore,
    FileDreamInbox,
    MemoryFacade,
    SemanticStore,
    TurnRecorder,
    WorkingMemoryStore,
)
from agent_platform.memory.procedural.skills import build_agent_instructions
from agent_platform.memory.procedural.store import SkillStore
from agent_platform.observability import StageReporter, checks
from agent_platform.observability.facts import create_fact_store, install_fact_store
from agent_platform.observability.langsmith import build_trace_exporter
from agent_platform.observability.model_observer import StageReporterModelObserver
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.observability.types import utc_now
from agent_platform.observability.units import OutcomeCatalog, declare_screens
from agent_platform.resilience import ResilientCaller, ResourcePoolBulkhead
from agent_platform.resilience.dead_letter_queue import DeadLetterQueue
from agent_platform.runtime.executor import ToolExecutorImpl
from agent_platform.runtime.executor.registry_handler import RegistryToolHandler
from agent_platform.scoring import create_scoring_cascade
from agent_platform.scoring.entailment import (
    EntailmentModel,
    LexicalEntailmentModel,
    create_entailment_model,
)
from agent_platform.security import (
    MessageScreener,
    PromptInjectionGuard,
    SecuritySuite,
    TermContentModerator,
)
from agent_platform.security.pii import PIIFilterImpl
from agent_platform.security.pii.boundary import bounded_roles
from agent_platform.security.prompt_injection.detector import InstructionDataSeparator
from agent_platform.security.prompt_injection.hosted import HostedJailbreakGuard
from agent_platform.tools.path import ToolPath
from research_agent.audit.checks import build_registry
from research_agent.audit.context import CheckContext
from research_agent.audit.independent import IndependentAuditor, LlmJudge
from research_agent.config import ResearchAgentSettings, get_agent_settings
from research_agent.context.session import AnswerContext
from research_agent.domain.matching import TermMatcher
from research_agent.domain.report import COVERAGE_AND_LIMITS
from research_agent.domain.rubric import Contract
from research_agent.graph.agent import ResearchAgent
from research_agent.graph.answerer import DirectAnswerer
from research_agent.graph.run_graph import build_run_graph
from research_agent.graph.run_nodes import RunNodes, RunParts
from research_agent.graph.scoping_graph import build_scoping_graph
from research_agent.graph.scoping_nodes import ScopingNodes, ScopingParts
from research_agent.graph.state import StateTypes
from research_agent.guardrails.answer_grounding import AnswerGrounding
from research_agent.guardrails.injection import InjectionScreen
from research_agent.guardrails.input_screen import InputScreen
from research_agent.guardrails.page_evidence import PageEvidence
from research_agent.guardrails.refusal import RefusalRule
from research_agent.llm.structured import RoleClients, StructuredCaller
from research_agent.memory import RunEventWriter
from research_agent.observability.telemetry import RunTelemetry
from research_agent.planning.planner import Planner
from research_agent.prompts import extractor as extractor_prompt
from research_agent.research.defenses import OutboundGuard
from research_agent.research.extractor import QuarantinedExtractor, UnquarantinedReader
from research_agent.research.recency import RecencyGuard
from research_agent.research.reflector import GapAnalyzer, Reflector
from research_agent.research.resources import RunResources
from research_agent.research.snippets import SnippetSources
from research_agent.research.stop_rules import stop_rule_for
from research_agent.rubric.editor import RubricEditor
from research_agent.rubric.generator import RubricGenerator
from research_agent.scoping.brief import BriefBuilder
from research_agent.scoping.challenge_check import ChallengeGuard
from research_agent.scoping.kind import KindClassifier, LlmKindClassifier
from research_agent.scoping.kind_encoder import EncoderKindClassifier, IntentBackend
from research_agent.scoping.kind_rules import KindRules
from research_agent.scoping.termination import TerminationPolicy
from research_agent.scoping.window import TimeWindowParser
from research_agent.synthesis.citation_guard import CitationGuard
from research_agent.synthesis.synthesizer import Synthesizer
from research_agent.tools import build_answer_registry
from research_agent.tools.web import FetchUrl

GatewayFactory = Callable[[], WebGateway]
JUDGE, SYNTHESIZER = "judge", "synthesizer"


def build_agent(
    reporter: StageReporter | None = None,
    features: FeatureSet | None = None,
    clients: RoleClients | None = None,
    gateway: GatewayFactory | None = None,
    entailment: EntailmentModel | None = None,
    checkpointer: SqliteCheckpointer | None = None,
    audit: AuditLedger | None = None,
    pool: SqliteConnectionPool | None = None,
    episodic: EpisodicStore | None = None,
    agent_settings: ResearchAgentSettings | None = None,
) -> ResearchAgent:
    """Wire the research agent from config, honouring anything injected."""
    settings, agent = get_settings(), agent_settings or get_agent_settings()
    posture = features if features is not None else active_features(settings.features)
    set_active_features(posture)
    platform = SqlitePlatformStore(
        pool or SqliteConnectionPool(settings.persistence.resolved_db_path),
        agent_id=settings.agent.id,
    )
    store = DuckDbObservabilityStore()
    ledger = audit or create_audit_ledger(store=store, agent_id=settings.agent.id)
    # Model calls, tool calls and turns, as rows the analytics source aggregates.
    install_fact_store(
        create_fact_store(store if settings.observability.enabled else None, settings.agent.id)
    )
    OutcomeCatalog().require(kinds=agent.tasks.kinds(), outcomes=agent.tasks.outcomes())
    telemetry = RunTelemetry(reporter, ledger, trace_id=settings.agent.id)
    observer = StageReporterModelObserver(
        lambda: telemetry.reporter, aggregate=agent.calls.context_aggregate_roles
    )
    # Every provider call tokenises personal data out and restores it in the answer.
    roles = bounded_roles(clients or role_clients(observer), posture)
    _require_independent_judge(roles, posture, agent)
    # Learned skills only (ra-0023): synced at start-up, chosen at plan by the run's kind.
    procedural = SkillStore() if posture.is_on("memory.procedural") else None
    caller = StructuredCaller(
        roles,
        PydanticValidator(),
        reporter,
        role_max_tokens=agent.calls.role_max_tokens,
        truncation_retries=agent.calls.truncation_retries,
        truncation_retry_factor=agent.calls.truncation_retry_factor,
        instructions=build_agent_instructions(
            settings.skills, store=procedural, audit=ledger, reporter=reporter
        ),
        profile=agent.instructions,
    )
    memory = _memory(agent, posture, platform, ledger, reporter, episodic, procedural)
    backend = kind_backend(agent, posture)
    answerer = _direct_answerer(agent, posture, caller, telemetry, ledger, gateway, memory)
    saver = checkpointer or create_checkpointer(state_types=model_types(StateTypes))
    registry = RunRegistry(platform)
    # Dream events (ADR 0012): every turn, and every finished run, for `make dream ra`.
    inbox = FileDreamInbox() if posture.is_on("memory") else None
    run_events = RunEventWriter(inbox, settings.agent.id, agent.memory) if inbox else None
    run_nodes = RunNodes(
        _run_parts(agent, posture, caller, roles, entailment, observer.flush, run_events)
    )
    return ResearchAgent(
        scoping=DurableGraphRunner(
            build_scoping_graph(
                ScopingNodes(_scoping_parts(agent, posture, caller, telemetry, backend)),
                saver,
                posture.is_on("rubric"),
            ),
            saver,
            registry,
        ),
        runs=DurableGraphRunner(
            build_run_graph(run_nodes, saver, posture.is_on("independent_audit")), saver, registry
        ),
        run_nodes=run_nodes,
        resources=_resources_factory(agent, posture, caller, telemetry, gateway),
        telemetry=telemetry,
        settings=agent,
        memory=memory,
        answerer=answerer,
        screen=_input_screen(posture, telemetry, platform),
        refusal=RefusalRule(agent.scoping.refuse_patterns),
        # Exported traces are always PII-redacted, whatever the `pii` feature says.
        trace_exporter=build_trace_exporter(settings.agent.id, lambda: PIIFilterImpl().redact),
        dream_recorder=(
            TurnRecorder(
                inbox,
                settings.agent.id,
                redact=PIIFilterImpl().redact if posture.is_on("pii") else None,
            )
            if inbox is not None
            else None
        ),
    )


def _input_screen(
    posture: FeatureSet, telemetry: RunTelemetry, platform: SqlitePlatformStore
) -> InputScreen | None:
    """The person's message screened by the declared detectors — or None when none is declared.

    A block is quarantined in the platform's dead-letter queue, as it is for every agent.
    """
    suite = SecuritySuite(
        pii=PIIFilterImpl() if posture.is_on("pii") else None,
        injection=PromptInjectionGuard() if posture.is_on("prompt_injection") else None,
        moderator=TermContentModerator() if posture.is_on("content_moderation") else None,
    )
    if suite.pii is None and not suite.enabled:
        declare_screens(get_settings().agent.id, [])
        return None
    quarantine = DeadLetterQueue(
        platform,
        agent_id=get_settings().agent.id,
        enabled=get_settings().resilience.dead_letter_queue.enabled,
    )
    screener = MessageScreener(suite, quarantine)
    declare_screens(get_settings().agent.id, screener.declared_checks())
    return InputScreen(screener, telemetry)


def role_clients(observer: ModelCallObserver | None = None) -> RoleClients:
    """`llm.roles` through the router: each role on its model, protected and backed up by the
    platform, its backup events told to `observer` (each call is recorded as a fact)."""
    return RoleModelRouter(get_settings().llm.roles, observer=observer)


def _require_independent_judge(
    roles: RoleClients, posture: FeatureSet, agent: ResearchAgentSettings
) -> None:
    """At start-up, not mid-run: an independent judge that is the writer is a config error."""
    if (
        posture.is_on("independent_audit")
        and agent.audit.independent_judge
        and isinstance(roles, RoleModelRouter)
    ):
        roles.require_distinct(JUDGE, SYNTHESIZER)


# -- scoping ---------------------------------------------------------------------------------


def _encoder_wanted(agent: ResearchAgentSettings, posture: FeatureSet) -> bool:
    """Whether the example classifier is part of this posture (never in the offline one)."""
    return posture.is_on("kind_encoder")


def kind_backend(agent: ResearchAgentSettings, posture: FeatureSet) -> IntentBackend | None:
    """The example classifier behind the research kind, or None (`kind_encoder` off).

    Nearest labelled examples (`kind_encoder.exemplars_path`) over an embedding model: no
    training, and the examples are the agent's own data. With the local embedder it is one per
    process, so the examples are embedded once — warmed at start-up — and not again on every
    `/harness` rebuild (1.8 s on e5-large-v2, measured 2026-10-04).
    """
    if not _encoder_wanted(agent, posture):
        return None
    settings = agent.kind_encoder
    if get_settings().llm.local_embedder.enabled:
        return _resident_kind_examples(settings.exemplars_path, settings.neighbours)
    return _kind_examples(create_embedding_client(), settings.exemplars_path, settings.neighbours)


@cache
def _resident_kind_examples(path: str, neighbours: int) -> ExemplarIntentClassifier:
    return _kind_examples(_local_embedder(), path, neighbours)


def warm_kind_examples(agent: ResearchAgentSettings) -> None:
    """Load the local embedder and embed the kind examples now, before an event loop opens.

    Whatever the posture: `/harness` can turn the example classifier on later, and a model
    loaded inside the console's loop fails on macOS (`fds_to_keep`).
    """
    if not get_settings().llm.local_embedder.enabled:
        return
    settings = agent.kind_encoder
    asyncio.run(_resident_kind_examples(settings.exemplars_path, settings.neighbours).warm())


def _kind_examples(
    embedder: EmbeddingClient, path: str, neighbours: int
) -> ExemplarIntentClassifier:
    exemplars = yaml.safe_load(resolve_path(path).read_text())
    get_logger("research_agent").info(
        "Research-kind classifier: nearest labelled example",
        embedder=embedder.model_id or get_settings().llm.embedding.model,
        examples=sum(len(texts) for texts in exemplars.values()),
        labels=sorted(exemplars),
    )
    return ExemplarIntentClassifier(embedder, exemplars, neighbours)


@cache
@checks.recorded_dependency(checks.EMBEDDING, "embed", lambda _model: checks.LOCAL_MODEL)
def _local_embedder() -> LocalSentenceEmbedder:
    """One embedder per process: the console warms it before its event loop starts."""
    return LocalSentenceEmbedder()


def kind_classifier(
    agent: ResearchAgentSettings,
    posture: FeatureSet,
    caller: StructuredCaller,
    backend: IntentBackend | None,
) -> KindClassifier:
    """Rules, then the example classifier, then the LLM — the LLM only as fallback.

    Refusal patterns and plain cue rules answer first, the example classifier answers the clear
    cases (a confident, unambiguous label) at no model cost, and the LLM keeps everything they
    are unsure about. It decides alone whenever the feature is off or the offline posture is on.
    """
    llm = LlmKindClassifier(caller, agent.scoping)
    if not _encoder_wanted(agent, posture):
        return llm
    return EncoderKindClassifier(backend, llm, agent.kind_encoder, KindRules(agent.scoping))


def _direct_answerer(
    agent: ResearchAgentSettings,
    posture: FeatureSet,
    caller: StructuredCaller,
    telemetry: RunTelemetry,
    ledger: AuditLedger,
    gateway: GatewayFactory | None,
    memory: MemoryFacade | None,
) -> DirectAnswerer | None:
    """What answers a plain question — or None, so every question starts a research scope.

    Withheld with the `direct_answer` feature.
    """
    if not posture.is_on("direct_answer"):
        return None
    web = gateway() if gateway is not None else create_web_gateway(_answer_web(agent))
    screen = _injection_screen(agent, posture, InstructionDataSeparator())
    registry = build_answer_registry(
        agent.answer, web, create_http_client(agent.answer.weather.timeout_seconds), screen
    )
    grounding = _answer_grounding(agent, posture, FetchUrl(web, agent.answer, screen))
    management = build_context_management(
        posture, memory.working if memory is not None else None, InstructionDataSeparator()
    )
    if management.fetch_tool is not None:
        registry.register(management.fetch_tool)
    context = AnswerContext(
        memory,
        agent.answer.context,
        agent.answer.messages,
        window=_context_window(posture, memory, management),
        reporter=lambda: telemetry.reporter,
    )
    # The answer tools read; they still take the platform's one tool path, so their
    # proposals are measured and each run is audited (UNGOVERNED: this agent declares
    # no Gatekeeper) and executed under a deadline — with the EXECUTOR card every agent draws.
    executor = ToolExecutorImpl(
        RegistryToolHandler(registry),
        reporter=telemetry.reporter,
        timeout_seconds=agent.answer.tool_timeout_seconds,
    )
    path = ToolPath(registry, executor, ledger)
    return DirectAnswerer(
        caller,
        registry,
        path,
        agent.answer,
        telemetry,
        grounding,
        context,
        recency_guard(agent),
        instructions=caller.instructions,
    )


def _answer_web(agent: ResearchAgentSettings) -> WebSettings:
    """The platform's web settings with the answer path's own, shorter fetch limits."""
    web = get_settings().web
    fetch = web.fetch.model_copy(
        update={
            "timeout_seconds": agent.answer.fetch_timeout_seconds,
            "max_attempts": agent.answer.fetch_max_attempts,
        }
    )
    return web.model_copy(update={"fetch": fetch})


def recency_guard(agent: ResearchAgentSettings) -> RecencyGuard:
    """No date named means the latest: the check on what the answer and the run search for."""
    return RecencyGuard(agent.recency, TimeWindowParser(agent.scoping.time_window))


def _context_window(
    posture: FeatureSet,
    memory: MemoryFacade | None,
    management: ContextManagement,
) -> ContextWindow | None:
    """The platform's context window over this agent's sources — or None with `context` off.

    The same assembly the support agent runs (five-stage pipeline, strategies, health check,
    card); only the source is this agent's: the session's working memory. Nothing from an
    earlier run is offered: every question is researched fresh.
    """
    if not posture.is_on("context"):
        return None
    scoring = create_scoring_cascade()
    sources: list[CandidateSource] = []
    if memory is not None and memory.working is not None:
        sources.append(WorkingMemoryContextSource(memory.working))
    reranker = create_reranker(fallback=scoring) if posture.is_on("context.rerank") else None
    pipeline = FiveStagePipeline(
        sources=sources,
        ranker=scoring,
        reranker=CrossEncoderReranker(reranker) if reranker is not None else None,
    )
    return ContextWindow(pipeline, management.strategies, management.health)


def _answer_grounding(
    agent: ResearchAgentSettings, posture: FeatureSet, fetch: FetchUrl
) -> AnswerGrounding | None:
    """The answer's grounding check, wired as the support agent wires its own — or None.

    Level 1 (token confidence) travels inside its HALLUCINATION CHECK card, first. `fetch`
    reads a cited page the loop saw only as a snippet, so the claim meets the page itself.
    """
    if not posture.is_on("direct_answer.grounding"):
        return None
    ladder = DetectionLadder(
        create_scoring_cascade(),
        entailment=entailment_for(agent),
        confirmer=create_claim_confirmer(),
    )
    guardrail = ResponseGuardrail(ladder)
    pages = PageEvidence(fetch, agent.answer.grounding)
    return AnswerGrounding(guardrail, agent.answer.grounding, agent.answer.messages, pages)


def _scoping_parts(
    agent: ResearchAgentSettings,
    posture: FeatureSet,
    caller: StructuredCaller,
    telemetry: RunTelemetry,
    backend: IntentBackend | None,
) -> ScopingParts:
    rubric_on = posture.is_on("rubric")
    generator = RubricGenerator(agent.rubric.resolved_templates_path) if rubric_on else None
    return ScopingParts(
        clock=utc_now,
        classifier=kind_classifier(agent, posture, caller, backend),
        caller=caller,
        checklists=agent.scoping.resolved_checklists_path,
        termination=TerminationPolicy(agent.scoping.max_turns),
        briefs=BriefBuilder(agent.scoping, agent.budget, agent.depths),
        telemetry=telemetry,
        settings=agent.scoping,
        messages=agent.messages,
        commands=agent.command_names,
        contract_prefix=agent.scoping.contract_id_prefix,
        challenge=ChallengeGuard(agent.scoping.challenge_every_k_turns)
        if posture.is_on("scoping_challenge_check")
        else None,
        generator=generator,
        editor=RubricEditor(generator.rules["user"]) if generator is not None else None,
    )


# -- the run -----------------------------------------------------------------------------------


def _run_parts(
    agent: ResearchAgentSettings,
    posture: FeatureSet,
    caller: StructuredCaller,
    roles: RoleClients,
    entailment: EntailmentModel | None,
    flush_context: Callable[[], None] | None = None,
    run_events: RunEventWriter | None = None,
) -> RunParts:
    registry = build_registry()
    nli = _entailment(agent, posture, entailment)
    judge = LlmJudge(caller)
    context = context_factory(agent, nli, judge)
    auditor = (
        _independent_auditor(registry, context, roles, agent)
        if posture.is_on("independent_audit")
        else None
    )
    reflection = posture.is_on("reflection")
    return RunParts(
        planner=Planner(
            caller,
            agent.planning.max_queries_per_node,
            agent.planning.node_id_prefix,
            agent.planning.covered_checks,
        ),
        analyzer=GapAnalyzer(registry),
        reflector=Reflector(
            caller if reflection else None, stop_rule_for, agent.reflection.max_followups_per_round
        ),
        synthesizer=Synthesizer(
            caller,
            agent.synthesis.resolved_templates_path,
            agent.synthesis.citation_pattern,
            agent.depths,
            agent.synthesis.working_note_patterns,
        ),
        registry=registry,
        context=context,
        features=posture,
        gap_fill_round_offset=agent.reflection.gap_fill_round_offset,
        pass_threshold=agent.audit.pass_threshold,
        partial_threshold=agent.audit.partial_threshold,
        clip_chars=agent.tui.clip_chars,
        node_id_prefix=agent.planning.node_id_prefix,
        reasons=agent.tui.reasons,
        auditor=auditor,
        reflection_enabled=reflection,
        usage=caller.usage,
        writer_model=roles.model_key(SYNTHESIZER),
        nli_model=str(getattr(nli, "model_name", "")) if nli is not None else "",
        judge_role=JUDGE,
        citation_guard=_citation_guard(agent, caller, nli),
        confidence=ConfidenceCheck(),
        flush_context=flush_context,
        writing_checks=frozenset(agent.reflection.writing_checks),
        max_revisions=agent.reflection.max_revisions,
        min_score_gain=agent.reflection.min_score_gain,
        judge_models=judge.models_used,
        # Learned skills only when procedural memory is composed (ra-0023).
        skills=caller.instructions if posture.is_on("memory.procedural") else None,
        skill_scope=agent.memory.scope,
        run_events=run_events,
    )


def _citation_guard(
    agent: ResearchAgentSettings, caller: StructuredCaller, nli: EntailmentModel | None
) -> CitationGuard | None:
    """The report's sentence-by-sentence check against its spans — wherever NLI is."""
    if nli is None:
        return None
    return CitationGuard(
        nli,
        caller,
        re.compile(agent.synthesis.citation_pattern),
        appended_headings(agent),
        agent.synthesis.max_guard_rewrites,
        confirmer=create_claim_confirmer(),
        rewrite_role=extractor_prompt.ROLE,
    )


def _independent_auditor(
    registry: Any, context: Any, roles: RoleClients, agent: ResearchAgentSettings
) -> IndependentAuditor:
    return IndependentAuditor(
        registry, context, roles.model_key(JUDGE), agent.audit.independent_judge
    )


def entailment_for(agent: ResearchAgentSettings) -> EntailmentModel:
    return (
        LexicalEntailmentModel()
        if agent.audit.entailment_model == "lexical"
        else create_entailment_model()
    )


def _entailment(
    agent: ResearchAgentSettings, posture: FeatureSet, injected: EntailmentModel | None
) -> EntailmentModel | None:
    if not posture.is_on("citation_verification.nli"):
        return None
    return injected if injected is not None else entailment_for(agent)


def appended_headings(agent: ResearchAgentSettings) -> frozenset[str]:
    """The sections code writes after the model (Known unknowns, Coverage & Limits)."""
    templates = yaml.safe_load(agent.synthesis.resolved_templates_path.read_text())
    return frozenset({*templates["appended"].values(), COVERAGE_AND_LIMITS})


def context_factory(
    agent: ResearchAgentSettings, entailment: EntailmentModel | None, judge: Any
) -> Callable[[Contract, Any, Any], CheckContext]:
    matcher = TermMatcher(agent.matching)
    citation = re.compile(agent.synthesis.citation_pattern)
    entity = re.compile(agent.audit.entity_pattern)

    appended = appended_headings(agent)

    def build(contract: Contract, ledger: Any, report: Any) -> CheckContext:
        return CheckContext(
            appended_headings=appended,
            contract=contract,
            ledger=ledger,
            report=report,
            matcher=matcher,
            citation_pattern=citation,
            entity_pattern=entity,
            primary_markers=tuple(agent.research.primary_source_markers),
            entailment=entailment,
            judge=judge,
        )

    return build


def _resources_factory(
    agent: ResearchAgentSettings,
    posture: FeatureSet,
    caller: StructuredCaller,
    telemetry: RunTelemetry,
    gateway: GatewayFactory | None,
) -> Callable[[Contract], RunResources]:
    separator = InstructionDataSeparator()

    def build(contract: Contract) -> RunResources:
        web = gateway() if gateway is not None else create_web_gateway()
        budget = BudgetGuard(
            contract.budget.counters(),
            max_seconds=contract.budget.max_seconds,
            layers=agent.budget_layers,
        )
        caller.bind_budget(budget)
        # A page that could not be read may be cited through its snippet, checked verbatim.
        snippets = SnippetSources(web)
        validator = (
            VerbatimSpanValidator(snippets)
            if posture.is_on("citation_verification")
            else NonEmptySpanValidator()
        )
        reader = (
            QuarantinedExtractor(
                caller,
                separator,
                agent.research.max_claims_per_page,
                agent.research.extractor_page_chars,
                agent.research.extractor_window_chars,
                contract.frozen_at.date(),
            )
            if posture.is_on("quarantined_extractor")
            else UnquarantinedReader(
                caller,
                agent.research.max_claims_per_page,
                contract.brief.model_dump(mode="json"),
                agent.research.extractor_page_chars,
                agent.research.extractor_window_chars,
                contract.frozen_at.date(),
            )
        )
        return RunResources(
            contract=contract,
            gateway=web,
            ledger=EvidenceLedger(validator, agent.synthesis.entry_id_prefix),
            budget=budget,
            reader=reader,
            telemetry=telemetry,
            settings=agent.research,
            outbound=OutboundGuard(contract.brief, agent.research)
            if posture.is_on("injection_structural")
            else None,
            screen=_injection_screen(agent, posture, separator),
            page_reads=ResourcePoolBulkhead(agent.research.page_concurrency),
            recency=recency_guard(agent),
            snippets=snippets,
        )

    return build


def _injection_screen(
    agent: ResearchAgentSettings, posture: FeatureSet, separator: InstructionDataSeparator
) -> InjectionScreen | None:
    """The phrase check, plus the hosted injection model when the feature and the key allow it.

    The offline posture (fixture backend) never calls out: it keeps the phrase check alone,
    which is what the recorded fixtures and evals were built against.
    """
    if not posture.is_on("injection_classifier"):
        return None
    guard = None
    guards = get_settings().security.guards
    if posture.is_on("injection_classifier.model") and guards.enabled and guards.jailbreak.enabled:
        guard = HostedJailbreakGuard(
            base_url=guards.jailbreak.base_url,
            endpoint=guards.jailbreak.endpoint,
            timeout_seconds=guards.jailbreak.timeout_seconds,
            model=guards.jailbreak.model,
            resilient=ResilientCaller(guards.jailbreak.resilience_key),
        )
    cfg = agent.injection_model
    return InjectionScreen(
        separator,
        guard,
        threshold=cfg.threshold,
        scan_all=cfg.scan_all,
        window_chars=cfg.window_chars,
        max_windows=cfg.max_windows,
    )


def _memory(
    agent: ResearchAgentSettings,
    posture: FeatureSet,
    platform: SqlitePlatformStore,
    audit: AuditLedger,
    reporter: StageReporter | None,
    episodic: EpisodicStore | None,
    procedural: SkillStore | None,
) -> MemoryFacade | None:
    """Every CoALA class as declared. Procedural holds learned skills only (ra-0021, ra-0023)."""
    if not posture.is_on("memory"):
        return None
    return MemoryFacade(
        working=WorkingMemoryStore(agent.answer.context.history_turns)
        if posture.is_on("memory.working")
        else None,
        episodic=(episodic or EpisodicStore()) if posture.is_on("memory.episodic") else None,
        semantic=SemanticStore() if posture.is_on("memory.semantic") else None,
        procedural=procedural,
        platform_store=platform,
        redactor=PIIFilterImpl() if posture.is_on("pii") else None,
        audit=audit,
        reporter=reporter,
    )
