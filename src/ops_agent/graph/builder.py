"""The ops agent's composition root: the only place its features are read.

Each declared feature builds a stage; a withheld one is never built:

    prompt_injection     the message is screened before the model reads it
    gatekeeper           each MCP call is authorized against `ops_agent.policy`
    hallucination_guard  every figure in the answer is checked against the query results
    memory.procedural    the agent's skills are synced into procedural memory and selected per turn

Everything else is always on: the MCP client (governed by the local envelope),
the Executor (deadline, facts), the audit trail, the fact ledger, tracing.
"""

from agent_platform.audit import AuditLedger, create_audit_ledger
from agent_platform.config import FeatureSet, active_features, get_settings
from agent_platform.executor import RegistryToolHandler, ToolExecutorImpl
from agent_platform.gatekeeper import CumulativeLedger, Gatekeeper, GatekeeperImpl
from agent_platform.hallucination_check.confidence import ConfidenceCheck
from agent_platform.llm.client import LLMClient, create_llm_client
from agent_platform.mcp import (
    ConfiguredGovernancePolicy,
    GovernedMcpToolSource,
    McpToolSource,
    McpTransport,
    SdkMcpTransport,
    server_spec,
)
from agent_platform.memory.procedural.memory import ProceduralStore
from agent_platform.observability import StageReporter
from agent_platform.observability.facts import create_fact_store, install_fact_store
from agent_platform.observability.langsmith import build_trace_exporter
from agent_platform.observability.model_observer import StageReporterModelObserver
from agent_platform.observability.store import DuckDbObservabilityStore
from agent_platform.observability.types import utc_now
from agent_platform.observability.units import OutcomeCatalog, declare_screens
from agent_platform.persistence import SqliteConnectionPool, SqlitePlatformStore
from agent_platform.resilience.dead_letter_queue import DeadLetterQueue
from agent_platform.security import MessageScreener, PromptInjectionGuard, SecuritySuite
from agent_platform.security.auth.auth import Authorizer
from agent_platform.security.auth.policy import AgentPolicy
from agent_platform.security.pii import PIIFilterImpl
from agent_platform.skills import build_agent_instructions
from agent_platform.tools import NamespacedToolRegistry
from agent_platform.tools.path import ToolPath
from ops_agent.config import get_agent_settings
from ops_agent.config.settings import OpsAgentSettings
from ops_agent.graph.agent import OUTCOMES, OpsAgent
from ops_agent.graph.graph import build_graph
from ops_agent.graph.nodes import OpsNodes
from ops_agent.graph.tool_calls import GovernedCalls
from ops_agent.guardrails import FigureCheck


def build_agent(
    reporter: StageReporter | None = None,
    features: FeatureSet | None = None,
    llm: LLMClient | None = None,
    transport: McpTransport | None = None,
    store: DuckDbObservabilityStore | None = None,
    pool: SqliteConnectionPool | None = None,
) -> OpsAgent:
    """Wire the ops agent from config, honouring anything injected."""
    settings, agent = get_settings(), get_agent_settings()
    posture = features if features is not None else active_features(settings.features)
    observability = store or DuckDbObservabilityStore()
    audit = create_audit_ledger(store=observability, agent_id=settings.agent.id)
    install_fact_store(
        create_fact_store(
            observability if settings.observability.enabled else None, settings.agent.id
        )
    )
    OutcomeCatalog().require(outcomes=OUTCOMES)
    platform = SqlitePlatformStore(
        pool or SqliteConnectionPool(settings.persistence.resolved_db_path),
        agent_id=settings.agent.id,
    )
    registry = NamespacedToolRegistry()
    path = ToolPath(
        registry,
        ToolExecutorImpl(RegistryToolHandler(registry), reporter=reporter),
        audit,
        gatekeeper=_gatekeeper(agent, platform, audit) if posture.is_on("gatekeeper") else None,
    )
    calls = GovernedCalls(registry, path, agent, actor=settings.agent.id)
    # Procedural memory is the only memory this agent has: its skills,
    # versioned in the platform store. Withheld, it runs on its profile.
    procedural = ProceduralStore(platform) if posture.is_on("memory.procedural") else None
    screen = _screen(platform) if posture.is_on("prompt_injection") else None
    declare_screens(settings.agent.id, screen.declared_checks() if screen else [])
    nodes = OpsNodes(
        llm or _llm(reporter),
        registry,
        calls,
        agent,
        reporter=reporter,
        figures=FigureCheck(agent) if posture.is_on("hallucination_guard") else None,
        confidence=_confidence() if posture.is_on("hallucination_guard") else None,
    )
    return OpsAgent(
        build_graph(nodes),
        nodes,
        registry,
        _sources(transport or SdkMcpTransport()),
        agent,
        reporter=reporter,
        screen=screen,
        instructions=build_agent_instructions(
            settings.skills, store=procedural, audit=audit, reporter=reporter
        ),
        # Exported traces are always PII-redacted, whatever the `pii` feature says.
        trace_exporter=build_trace_exporter(settings.agent.id, lambda: PIIFilterImpl().redact),
    )


def _llm(reporter: StageReporter | None) -> LLMClient:
    return create_llm_client(observer=StageReporterModelObserver(lambda: reporter))


def _confidence() -> ConfidenceCheck | None:
    """Token confidence, step 1 of the hallucination check — unless withheld in config."""
    check = ConfidenceCheck()
    return check if check.enabled else None


def _sources(transport: McpTransport) -> list[McpToolSource]:
    """One governed tool source per configured MCP server."""
    servers = get_settings().mcp.servers
    policy = ConfiguredGovernancePolicy(servers)
    return [GovernedMcpToolSource(server_spec(server), transport, policy) for server in servers]


def _gatekeeper(
    agent: OpsAgentSettings, platform: SqlitePlatformStore, audit: AuditLedger
) -> Gatekeeper:
    """Authority from this agent's configured policy; no budgeted action, so no ledger spend."""
    policy = AgentPolicy(
        agent_id=platform.agent_id,
        role=agent.policy.role,
        authority_rank=agent.policy.authority_rank,
        permissions=set(agent.policy.permissions),
        updated_at=utc_now().isoformat(),
    )
    ledger = CumulativeLedger(platform, agent_id=platform.agent_id, enabled=False)
    return GatekeeperImpl(authorizer=Authorizer(policy), policy=policy, ledger=ledger, audit=audit)


def _screen(platform: SqlitePlatformStore) -> MessageScreener:
    quarantine = DeadLetterQueue(
        platform,
        agent_id=platform.agent_id,
        enabled=get_settings().resilience.dead_letter_queue.enabled,
    )
    return MessageScreener(SecuritySuite(injection=PromptInjectionGuard()), quarantine)


__all__ = ["build_agent"]
