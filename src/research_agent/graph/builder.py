"""The composition root: the one place features are read and classes chosen.

Every feature is read here, once, from the posture handed in (or the
declared configuration). A withheld feature is a collaborator that is
never built — no challenge guard, no rubric generator, no classifier, no
quarantined extractor, no Goldfish node in the compiled graph — never a
component that runs and reports having done nothing.

Anything passed in is used as-is; anything omitted is built from config.
Tests pass a scripted model, a replaying gateway and in-memory stores and
get the real wiring of everything else.
"""

import re
from collections.abc import Callable
from datetime import datetime
from typing import Any

import yaml

from agentic_commons.artifacts import DirectoryArtifactStore
from agentic_commons.audit import AuditLedger, create_audit_ledger
from agentic_commons.config import FeatureSet, active_features, get_settings, set_active_features
from agentic_commons.config.paths import resolve_path
from agentic_commons.control_loop import BudgetGuard
from agentic_commons.durability import (
    DurableGraphRunner,
    RunRegistry,
    create_checkpointer,
    model_types,
)
from agentic_commons.durability.checkpointer import SqliteCheckpointer
from agentic_commons.evals.entailment import (
    EntailmentModel,
    LexicalEntailmentModel,
    create_entailment_model,
)
from agentic_commons.evals.scoring import create_scoring_cascade
from agentic_commons.evidence import EvidenceLedger, NonEmptySpanValidator, VerbatimSpanValidator
from agentic_commons.guardrails.guards import PydanticValidator
from agentic_commons.guardrails.response import ResponseGuardrail
from agentic_commons.hallucination_protection.checker import DetectionLadder
from agentic_commons.intent.exemplar import ExemplarIntentClassifier
from agentic_commons.llm.embeddings import create_embedding_client
from agentic_commons.llm.http import create_http_client
from agentic_commons.llm.roles import RoleModelRouter
from agentic_commons.logging import get_logger
from agentic_commons.memory import EpisodicStore, MemoryFacade, SemanticStore, WorkingMemoryStore
from agentic_commons.observability import StageReporter
from agentic_commons.observability.store import DuckDbObservabilityStore
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import SqliteConnectionPool, SqlitePlatformStore
from agentic_commons.resilience import ResilientCaller
from agentic_commons.security.pii import PIIFilterImpl
from agentic_commons.security.prompt_injection.detector import InstructionDataSeparator
from agentic_commons.security.prompt_injection.nvidia import NvidiaHostedJailbreakGuard
from agentic_commons.web import WebGateway, create_web_gateway
from research_agent.answer import DirectAnswerer
from research_agent.answer.context import AnswerContext
from research_agent.answer.grounding import AnswerGrounding
from research_agent.answer.tools import build_answer_registry
from research_agent.archive.store import ResearchArchive
from research_agent.audit.checks import build_registry
from research_agent.audit.context import CheckContext
from research_agent.audit.goldfish import GoldfishAuditor, LlmJudge
from research_agent.config import ResearchAgentSettings, get_agent_settings
from research_agent.config.settings import EncoderGateSettings
from research_agent.domain.report import COVERAGE_AND_LIMITS
from research_agent.domain.rubric import Contract
from research_agent.graph.agent import ResearchAgent
from research_agent.graph.run_graph import build_run_graph
from research_agent.graph.run_nodes import RunNodes, RunParts
from research_agent.graph.scoping_graph import build_scoping_graph
from research_agent.graph.scoping_nodes import ScopingNodes, ScopingParts
from research_agent.graph.state import StateTypes
from research_agent.intent import ResearchIntentRouter
from research_agent.planning.planner import Planner
from research_agent.research.cards import cap_reached_card
from research_agent.research.defenses import InjectionScreen, OutboundGuard
from research_agent.research.extractor import QuarantinedExtractor, UnquarantinedReader
from research_agent.research.matching import TermMatcher
from research_agent.research.reflector import GapAnalyzer, Reflector
from research_agent.research.resources import RunResources
from research_agent.research.stop_rules import stop_rule_for
from research_agent.research.telemetry import RunTelemetry
from research_agent.rubric.editor import RubricEditor
from research_agent.rubric.generator import RubricGenerator
from research_agent.scoping.brief import BriefBuilder
from research_agent.scoping.challenge_check import ChallengeGuard
from research_agent.scoping.encoder_gate import EncoderGateClassifier, IntentBackend
from research_agent.scoping.gate import ArchetypeClassifier, LlmArchetypeClassifier
from research_agent.scoping.rule_gate import RuleGate
from research_agent.scoping.termination import TerminationPolicy
from research_agent.structured import RoleClients, StructuredCaller
from research_agent.synthesis.synthesizer import Synthesizer

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
    artifacts: DirectoryArtifactStore | None = None,
    episodic: EpisodicStore | None = None,
    agent_settings: ResearchAgentSettings | None = None,
) -> ResearchAgent:
    """Wire the research agent from config, honouring anything injected."""
    settings, agent = get_settings(), agent_settings or get_agent_settings()
    posture = features if features is not None else active_features(settings.features)
    set_active_features(posture)
    roles = clients or role_clients(agent)
    _require_independent_judge(roles, posture, agent)
    platform = SqlitePlatformStore(
        pool or SqliteConnectionPool(settings.persistence.resolved_db_path),
        agent_id=settings.agent.id,
    )
    ledger = audit or create_audit_ledger(
        store=DuckDbObservabilityStore(), agent_id=settings.agent.id
    )
    telemetry = RunTelemetry(reporter, ledger, trace_id=settings.agent.id)
    caller = StructuredCaller(
        roles,
        PydanticValidator(),
        reporter,
        role_max_tokens=agent.calls.role_max_tokens,
        truncation_retries=agent.calls.truncation_retries,
        truncation_retry_factor=agent.calls.truncation_retry_factor,
        role_timeout_seconds=agent.calls.role_timeout_seconds,
        timeout_retries=agent.calls.timeout_retries,
    )
    memory = _memory(agent, posture, platform, ledger, reporter, episodic)
    backend = _classifier_backend(agent, posture)
    archive = ResearchArchive(
        artifacts or DirectoryArtifactStore(settings.artifacts.resolved_root),
        memory,
        agent.archive.run_id_prefix,
        semantic_pointer=agent.archive.semantic_pointer,
    )
    router, answerer = _front_door(
        agent, posture, caller, telemetry, gateway, backend, memory, archive
    )
    saver = checkpointer or create_checkpointer(state_types=model_types(StateTypes))
    registry = RunRegistry(platform)
    run_nodes = RunNodes(_run_parts(agent, posture, caller, roles, archive, entailment))
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
            build_run_graph(run_nodes, saver, posture.is_on("goldfish_audit")), saver, registry
        ),
        run_nodes=run_nodes,
        resources=_resources_factory(agent, posture, caller, telemetry, gateway),
        archive=archive,
        telemetry=telemetry,
        settings=agent,
        memory=memory,
        router=router,
        answerer=answerer,
    )


def role_clients(agent: ResearchAgentSettings) -> RoleClients:
    """`llm.roles` through the router — real models, or the fixture world's reference model."""
    roles = get_settings().llm.roles
    if agent.model_backend == "fixture":
        from research_agent.fixture_world.model import fixture_client_factory

        return RoleModelRouter(roles, factory=fixture_client_factory(agent))
    return RoleModelRouter(roles)


def _require_independent_judge(
    roles: RoleClients, posture: FeatureSet, agent: ResearchAgentSettings
) -> None:
    """At start-up, not mid-run: an independent judge that is the writer is a config error."""
    if (
        posture.is_on("goldfish_audit")
        and agent.audit.independent_judge
        and isinstance(roles, RoleModelRouter)
    ):
        roles.require_distinct(JUDGE, SYNTHESIZER)


# -- scoping ---------------------------------------------------------------------------------


def _encoder_wanted(agent: ResearchAgentSettings, posture: FeatureSet) -> bool:
    """Whether the BERT-family encoder is part of this posture (never in the offline one)."""
    return posture.is_on("gate_encoder") and agent.model_backend != "fixture"


def _classifier_backend(agent: ResearchAgentSettings, posture: FeatureSet) -> IntentBackend | None:
    """The learned classifier behind the scope gate and the front door.

    A fine-tuned encoder when one is published or trained locally; otherwise nearest labelled
    example over the embedding model, which needs no training. None when the posture withholds
    classifiers (the offline fixture world, or `gate_encoder` off): rules and the LLM decide.
    """
    if not _encoder_wanted(agent, posture):
        return None
    return _gate_backend(agent.gate_encoder) or _exemplar_backend(agent)


def _exemplar_backend(agent: ResearchAgentSettings) -> IntentBackend:
    path = resolve_path(agent.intent_router.exemplars_path)
    exemplars = yaml.safe_load(path.read_text())
    get_logger("research_agent").info(
        "Intent classifier: nearest labelled example over the embedding model",
        examples=sum(len(texts) for texts in exemplars.values()),
        labels=sorted(exemplars),
    )
    return ExemplarIntentClassifier(
        create_embedding_client(), exemplars, agent.intent_router.neighbours
    )


def gate_classifier(
    agent: ResearchAgentSettings,
    posture: FeatureSet,
    caller: StructuredCaller,
    backend: IntentBackend | None,
) -> ArchetypeClassifier:
    """Rules, then a fine-tuned BERT-family classifier, then the LLM — the LLM only as fallback.

    Refusal patterns and plain cue rules answer first, the encoder answers the clear cases (a
    confident, unambiguous label) at no model cost, and the LLM gate keeps everything they are
    unsure about. It is the whole gate whenever the feature is off or the offline posture is on.
    """
    llm = LlmArchetypeClassifier(caller, agent.scoping)
    if not _encoder_wanted(agent, posture):
        return llm
    return EncoderGateClassifier(backend, llm, agent.gate_encoder, RuleGate(agent.scoping))


def _gate_backend(cfg: EncoderGateSettings) -> IntentBackend | None:
    """The fine-tuned encoder, loaded now (before any event loop), or None to skip that layer.

    It is the published `model`, or a model trained locally into `local_model_dir`
    (`make train-gate ra`). With neither, the layer is skipped and the LLM answers what the
    rules do not.
    """
    from agentic_commons.config.paths import resolve_path
    from agentic_commons.intent.classifier import HuggingFaceIntentClassifier

    local = resolve_path(cfg.local_model_dir)
    if not cfg.model and not local.exists():
        get_logger("research_agent").debug(
            "No fine-tuned gate encoder is published or trained (`make train-gate ra` trains one)",
            local_model_dir=str(local),
        )
        return None
    backend = HuggingFaceIntentClassifier(
        model=cfg.model or str(local),
        local_model_dir=local,
        download_if_missing=cfg.download_if_missing and bool(cfg.model),
    )
    try:
        backend.ensure_loaded_sync()
    except Exception as exc:
        get_logger("research_agent").warning("Gate encoder failed to load", error=str(exc))
        return None
    return backend


def _front_door(
    agent: ResearchAgentSettings,
    posture: FeatureSet,
    caller: StructuredCaller,
    telemetry: RunTelemetry,
    gateway: GatewayFactory | None,
    backend: IntentBackend | None,
    memory: MemoryFacade | None,
    archive: ResearchArchive,
) -> tuple[ResearchIntentRouter | None, DirectAnswerer | None]:
    """The intent router and the direct answerer — or neither, so every question starts a scope.

    Withheld with the `direct_answer` feature, and in the offline fixture posture, whose scripted
    model cannot call tools.
    """
    if not posture.is_on("direct_answer") or agent.model_backend == "fixture":
        return None, None
    web = gateway() if gateway is not None else create_web_gateway()
    registry = build_answer_registry(
        agent.answer,
        web,
        create_http_client(agent.answer.weather.timeout_seconds),
        _injection_screen(agent, posture, InstructionDataSeparator()),
    )
    router = ResearchIntentRouter(agent.intent_router, agent.scoping, backend, caller)
    grounding = _answer_grounding(agent, posture)
    context = AnswerContext(memory, archive, agent.answer.context, agent.answer.messages)
    return router, DirectAnswerer(caller, registry, agent.answer, telemetry, grounding, context)


def _answer_grounding(agent: ResearchAgentSettings, posture: FeatureSet) -> AnswerGrounding | None:
    """The answer's grounding check, wired as the support agent wires its own — or None."""
    if not posture.is_on("direct_answer.grounding"):
        return None
    ladder = DetectionLadder(create_scoring_cascade(), entailment=entailment_for(agent))
    return AnswerGrounding(ResponseGuardrail(ladder), agent.answer.grounding, agent.answer.messages)


def contract_clock(agent: ResearchAgentSettings) -> Callable[[], datetime]:
    """Real time, except over the fixture world, whose dates are fixed at its `as_of`."""
    if agent.model_backend != "fixture":
        return utc_now
    from research_agent.fixture_world.world import load_world

    as_of = load_world(agent.resolved_fixture_world_path).as_of
    return lambda: as_of


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
        clock=contract_clock(agent),
        classifier=gate_classifier(agent, posture, caller, backend),
        caller=caller,
        checklists=agent.scoping.resolved_checklists_path,
        termination=TerminationPolicy(agent.scoping.max_turns),
        briefs=BriefBuilder(agent.scoping, agent.budget, agent.qa.budget, agent.depths),
        telemetry=telemetry,
        settings=agent.scoping,
        messages=agent.messages,
        commands=agent.command_names,
        contract_prefix=agent.archive.run_id_prefix,
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
    archive: ResearchArchive,
    entailment: EntailmentModel | None,
) -> RunParts:
    registry = build_registry()
    nli = _entailment(agent, posture, entailment)
    judge = LlmJudge(caller)
    context = context_factory(agent, nli, judge)
    auditor = (
        _goldfish(registry, context, roles, agent) if posture.is_on("goldfish_audit") else None
    )
    reflection = posture.is_on("reflection")
    return RunParts(
        planner=Planner(caller, agent.planning.max_queries_per_node, agent.planning.node_id_prefix),
        analyzer=GapAnalyzer(registry),
        reflector=Reflector(
            caller if reflection else None, stop_rule_for, agent.reflection.max_followups_per_round
        ),
        synthesizer=Synthesizer(
            caller,
            agent.synthesis.resolved_templates_path,
            agent.synthesis.citation_pattern,
            agent.depths,
        ),
        archive=archive,
        registry=registry,
        context=context,
        features=posture,
        gap_fill_round_offset=agent.reflection.gap_fill_round_offset,
        pass_threshold=agent.audit.pass_threshold,
        partial_threshold=agent.audit.partial_threshold,
        clip_chars=agent.tui.clip_chars,
        node_id_prefix=agent.planning.node_id_prefix,
        budget_headline=agent.tui.budget_headline,
        reasons=agent.tui.reasons,
        auditor=auditor,
        reflection_enabled=reflection,
        usage=caller.usage,
    )


def _goldfish(
    registry: Any, context: Any, roles: RoleClients, agent: ResearchAgentSettings
) -> GoldfishAuditor:
    return GoldfishAuditor(registry, context, roles.model_key(JUDGE), agent.audit.independent_judge)


def build_auditor(
    agent: ResearchAgentSettings, roles: RoleClients, entailment: EntailmentModel
) -> GoldfishAuditor:
    """A stand-alone Goldfish Audit, whatever the agent's posture — the evals' instrument."""
    judge = LlmJudge(StructuredCaller(roles, PydanticValidator()))
    return _goldfish(build_registry(), context_factory(agent, entailment, judge), roles, agent)


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


def context_factory(
    agent: ResearchAgentSettings, entailment: EntailmentModel | None, judge: Any
) -> Callable[[Contract, Any, Any], CheckContext]:
    matcher = TermMatcher(agent.matching)
    citation = re.compile(agent.synthesis.citation_pattern)
    entity = re.compile(agent.audit.entity_pattern)

    appended = frozenset(
        {
            *yaml.safe_load(agent.synthesis.resolved_templates_path.read_text())[
                "appended"
            ].values(),
            COVERAGE_AND_LIMITS,
        }
    )

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
            on_exhausted=lambda report: telemetry.card(
                "budget_tick",
                *cap_reached_card(
                    report.reason, report.checks[0].observed if report.checks else {}
                ),
            ),
        )
        caller.bind_budget(budget)
        validator = (
            VerbatimSpanValidator(web)
            if posture.is_on("citation_verification")
            else NonEmptySpanValidator()
        )
        reader = (
            QuarantinedExtractor(
                caller,
                separator,
                agent.research.max_claims_per_page,
                agent.research.extractor_page_chars,
            )
            if posture.is_on("quarantined_extractor")
            else UnquarantinedReader(
                caller,
                agent.research.max_claims_per_page,
                contract.brief.model_dump(mode="json"),
                agent.research.extractor_page_chars,
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
    if (
        posture.is_on("injection_classifier.model")
        and agent.model_backend != "fixture"
        and guards.enabled
        and guards.jailbreak.enabled
    ):
        guard = NvidiaHostedJailbreakGuard(
            base_url=guards.jailbreak.base_url,
            endpoint=guards.jailbreak.endpoint,
            timeout_seconds=guards.jailbreak.timeout_seconds,
            model=guards.jailbreak.model,
            resilient=ResilientCaller("nvidia-jailbreak"),
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
) -> MemoryFacade | None:
    """Working, episodic and semantic as declared; procedural is never built (by design)."""
    if not posture.is_on("memory"):
        return None
    return MemoryFacade(
        working=WorkingMemoryStore(agent.answer.context.history_turns)
        if posture.is_on("memory.working")
        else None,
        episodic=(episodic or EpisodicStore()) if posture.is_on("memory.episodic") else None,
        semantic=SemanticStore(platform) if posture.is_on("memory.semantic") else None,
        procedural=None,
        platform_store=platform,
        redactor=PIIFilterImpl() if posture.is_on("pii") else None,
        audit=audit,
        reporter=reporter,
    )
