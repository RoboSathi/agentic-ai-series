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
from agentic_commons.evidence import EvidenceLedger, NonEmptySpanValidator, VerbatimSpanValidator
from agentic_commons.guardrails.guards import PydanticValidator
from agentic_commons.llm.roles import RoleModelRouter
from agentic_commons.memory import EpisodicStore, MemoryFacade, SemanticStore, WorkingMemoryStore
from agentic_commons.observability import StageReporter
from agentic_commons.observability.store import DuckDbObservabilityStore
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import SqliteConnectionPool, SqlitePlatformStore
from agentic_commons.security.pii import PIIFilterImpl
from agentic_commons.security.prompt_injection.detector import InstructionDataSeparator
from agentic_commons.web import WebGateway, create_web_gateway
from research_agent.archive.store import ResearchArchive
from research_agent.audit.checks import build_registry
from research_agent.audit.context import CheckContext
from research_agent.audit.goldfish import GoldfishAuditor, LlmJudge
from research_agent.config import ResearchAgentSettings, get_agent_settings
from research_agent.domain.report import COVERAGE_AND_LIMITS
from research_agent.domain.rubric import Contract
from research_agent.graph.agent import ResearchAgent
from research_agent.graph.run_graph import build_run_graph
from research_agent.graph.run_nodes import RunNodes, RunParts
from research_agent.graph.scoping_graph import build_scoping_graph
from research_agent.graph.scoping_nodes import ScopingNodes, ScopingParts
from research_agent.graph.state import StateTypes
from research_agent.planning.planner import Planner
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
from research_agent.scoping.gate import LlmArchetypeClassifier
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
    caller = StructuredCaller(roles, PydanticValidator(), reporter)
    memory = _memory(posture, platform, ledger, reporter, episodic)
    archive = ResearchArchive(
        artifacts or DirectoryArtifactStore(settings.artifacts.resolved_root),
        memory,
        agent.archive.run_id_prefix,
    )
    saver = checkpointer or create_checkpointer(state_types=model_types(StateTypes))
    registry = RunRegistry(platform)
    run_nodes = RunNodes(_run_parts(agent, posture, caller, roles, archive, entailment))
    return ResearchAgent(
        scoping=DurableGraphRunner(
            build_scoping_graph(
                ScopingNodes(_scoping_parts(agent, posture, caller, telemetry)),
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
) -> ScopingParts:
    rubric_on = posture.is_on("rubric")
    generator = RubricGenerator(agent.rubric.resolved_templates_path) if rubric_on else None
    return ScopingParts(
        clock=contract_clock(agent),
        classifier=LlmArchetypeClassifier(caller, agent.scoping),
        caller=caller,
        checklists=agent.scoping.resolved_checklists_path,
        termination=TerminationPolicy(agent.scoping.max_turns),
        briefs=BriefBuilder(agent.scoping, agent.budget, agent.qa.budget),
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
        planner=Planner(caller, agent.planning.max_queries_per_node),
        analyzer=GapAnalyzer(registry),
        reflector=Reflector(
            caller if reflection else None, stop_rule_for, agent.reflection.max_followups_per_round
        ),
        synthesizer=Synthesizer(
            caller, agent.synthesis.resolved_templates_path, agent.synthesis.citation_pattern
        ),
        archive=archive,
        registry=registry,
        context=context,
        features=posture,
        gap_fill_round_offset=agent.reflection.gap_fill_round_offset,
        auditor=auditor,
        reflection_enabled=reflection,
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
            on_exhausted=lambda report: telemetry.stage(
                "budget_tick",
                f"Budget exhausted: {report.reason}",
                report.model_dump(),
                status="warning",
            ),
        )
        caller.bind_budget(budget)
        validator = (
            VerbatimSpanValidator(web)
            if posture.is_on("citation_verification")
            else NonEmptySpanValidator()
        )
        reader = (
            QuarantinedExtractor(caller, separator, agent.research.max_claims_per_page)
            if posture.is_on("quarantined_extractor")
            else UnquarantinedReader(
                caller, agent.research.max_claims_per_page, contract.brief.model_dump(mode="json")
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
            screen=InjectionScreen(separator) if posture.is_on("injection_classifier") else None,
        )

    return build


def _memory(
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
        working=WorkingMemoryStore() if posture.is_on("memory.working") else None,
        episodic=(episodic or EpisodicStore()) if posture.is_on("memory.episodic") else None,
        semantic=SemanticStore(platform) if posture.is_on("memory.semantic") else None,
        procedural=None,
        platform_store=platform,
        redactor=PIIFilterImpl() if posture.is_on("pii") else None,
        audit=audit,
        reporter=reporter,
    )
