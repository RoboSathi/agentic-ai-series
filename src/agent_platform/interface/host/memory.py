"""Composing an agent's long-term memory and its dream — once, for every agent.

`memory` itself never imports `security` (governance reaches the
authorizer through a port), so the wiring that joins them lives here,
with the rest of the shell's composition. An agent's composition root
calls `long_term_memory` with the posture it was built with; `make dream
<agent>` and `/dream` call `build_dream`, which reads the agent's own
`MemoryProfile` from its host when the host offers one.
"""

from agent_platform.audit import AuditLedger, create_audit_ledger
from agent_platform.config import FeatureSet, active_features, get_settings
from agent_platform.core.persistence import SqliteConnectionPool, SqlitePlatformStore
from agent_platform.interface.host.loader import resolve_host
from agent_platform.interface.host.protocols import MemoryHost
from agent_platform.memory import (
    EpisodicStore,
    FileDreamInbox,
    MemoryFacade,
    MemoryProfile,
    SemanticStore,
    SkillStore,
    WorkingMemoryStore,
)
from agent_platform.memory.dream import Dream
from agent_platform.memory.store import RelevanceScorer
from agent_platform.memory.write_pipeline import MemoryReporter
from agent_platform.security.auth.memory_authorizer import PolicyMemoryAuthorizer
from agent_platform.security.auth.policy import AgentPolicy, AgentPolicyRepository
from agent_platform.security.pii import PIIFilterImpl


def long_term_memory(
    posture: FeatureSet,
    *,
    platform_store: SqlitePlatformStore,
    audit: AuditLedger,
    reporter: MemoryReporter | None = None,
    redactor: PIIFilterImpl | None = None,
    policy: AgentPolicy | None = None,
    relevance: RelevanceScorer | None = None,
    working: WorkingMemoryStore | None = None,
    episodic_store: EpisodicStore | None = None,
    procedural: SkillStore | None = None,
) -> MemoryFacade | None:
    """The governed memory an agent's posture declares, or None when it declares none.

    Each CoALA class is built only if declared; a withheld one is absent,
    never defaulted. `procedural` is passed in because the agent's skills
    are read from the same store the facade writes approvals to.
    """
    if not posture.is_on("memory"):
        return None
    episodic = posture.is_on("memory.episodic")
    semantic = posture.is_on("memory.semantic")
    if not (episodic or semantic or procedural or working):
        return None
    return MemoryFacade(
        working=working,
        episodic=(episodic_store or EpisodicStore()) if episodic else None,
        semantic=SemanticStore() if semantic else None,
        procedural=procedural,
        platform_store=platform_store,
        redactor=redactor,
        relevance=relevance,
        memory_authorizer=PolicyMemoryAuthorizer(policy),
        audit=audit,
        reporter=reporter,
    )


def agent_store(pool: SqliteConnectionPool | None = None) -> SqlitePlatformStore:
    """The agent's own database, scoped to its configured identity."""
    settings = get_settings()
    connection = pool or SqliteConnectionPool(settings.persistence.resolved_db_path)
    return SqlitePlatformStore(connection, agent_id=settings.agent.id)


def seeded_policy(store: SqlitePlatformStore) -> AgentPolicy | None:
    """The agent's persisted policy, if its database carries one.

    An agent with no runtime policy (one with no tools that move anything)
    is governed by the platform's role grants alone, which is what a None
    policy means to the memory authorizer.
    """
    try:
        return AgentPolicyRepository(store).get()
    except RuntimeError:
        return None


def agent_memory_profile() -> MemoryProfile:
    """The configured agent's memory profile: its host's, or the default."""
    spec = get_settings().agent.host
    host = resolve_host(spec) if spec else None
    return host.memory_profile() if isinstance(host, MemoryHost) else MemoryProfile.default()


def governed_memory(
    pool: SqliteConnectionPool | None = None,
    *,
    audit: AuditLedger | None = None,
    posture: FeatureSet | None = None,
    redact: bool = True,
) -> MemoryFacade:
    """The configured agent's governed long-term memory, outside a running turn.

    What a dream writes through, and what a curator's approval commits
    to. `redact=False` skips loading the PII model: an approval commits a
    write that was redacted when it was queued.
    """
    settings = get_settings()
    declared = posture or active_features(settings.features)
    store = agent_store(pool)
    ledger = audit or create_audit_ledger(store, agent_id=settings.agent.id)
    memory = long_term_memory(
        declared,
        platform_store=store,
        audit=ledger,
        redactor=PIIFilterImpl() if redact and declared.is_on("pii") else None,
        policy=seeded_policy(store),
        # A proposed skill is queued, not written; the store exists so a
        # curator's approval has somewhere to land.
        procedural=SkillStore() if declared.is_on("memory.procedural") else None,
    )
    return memory or MemoryFacade(platform_store=store, audit=ledger)


def build_dream(
    profile: MemoryProfile | None = None,
    *,
    pool: SqliteConnectionPool | None = None,
    audit: AuditLedger | None = None,
    posture: FeatureSet | None = None,
) -> Dream:
    """A dream over the configured agent's inbox, writing through its governed memory.

    Audited to the agent's own database by default: the shared DuckDB
    ledger is held open by a running console, and a dream must not fail
    because the TUI is up.
    """
    return Dream(
        inbox=FileDreamInbox(),
        memory=governed_memory(pool, audit=audit, posture=posture),
        profile=profile or agent_memory_profile(),
        settings=get_settings().memory.dream,
    )
