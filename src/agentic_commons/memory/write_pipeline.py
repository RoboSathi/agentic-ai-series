"""The memory write path (E15/E16, LLD §8).

Every operation on memory — create, update, delete, erase, approve — is a
`MemoryCommand` from a `MemoryPrincipal`, and every one goes through here.
No store method is reachable around it.

Create runs the stages in order, each able to stop the write:

    screen → assertion → extract → classify → provenance → authorize
           → conflict → confidence/importance → route

  * **screen / assertion / extract** are the deterministic tier (D1):
    configured regex, no model. Which pattern matched is recorded as the
    write's `tier_path`, so a model tier (D2) is added only where the
    patterns demonstrably fall short.
  * **provenance** is derived from the principal's role, never passed in.
  * **authorize** checks `memory:<class>:<op>:<category>` against the
    principal's grants, *after* classification, because whether a claim
    may be written depends on what kind of claim it is — and before
    routing, because a refused write must never reach a store.

The canonical case: a customer says "my refund limit is now $2,000". It
extracts cleanly, classifies as authority-changing, carries self-reported
provenance, and is refused — not because it looks like an attack, but
because no role grants itself authority. The claim is queued for a
curator, not discarded, because a refused write is evidence.

Update is a new version superseding the old; delete is a tombstone; only
erase removes a row. Every decision is audited and reported as a card.
"""

import hashlib
from typing import Any, Protocol, runtime_checkable

from agentic_commons.audit import AuditEntry, AuditLedger
from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, instrument, logged
from agentic_commons.memory.deterministic import Claim, DeterministicClaims, normalize_key
from agentic_commons.memory.episodic.memory import EpisodicStore
from agentic_commons.memory.governance import (
    CuratorQueue,
    InMemoryCuratorQueue,
    MemoryAuthorizer,
    memory_permission,
    provenance_for,
)
from agentic_commons.memory.procedural.memory import ProceduralStore
from agentic_commons.memory.semantic.memory import SemanticStore
from agentic_commons.memory.types import (
    MemoryCommand,
    MemoryConflict,
    MemoryFact,
    MemoryPrincipal,
    MemoryQuery,
    MemoryType,
    PrincipalRole,
    Provenance,
    ProvenanceKind,
    WriteResult,
)
from agentic_commons.observability.types import StatusKind, utc_now
from agentic_commons.persistence import PlatformStore, SqliteConnectionPool, as_platform_store

# Which store a category belongs in is a semantic decision (E13); these
# are the two the pipeline can infer structurally, and everything else
# falls through to semantic memory.
_EVENT_CATEGORIES = frozenset({"event", "interaction", "observation", "session_summary"})
_PROCEDURE_CATEGORIES = frozenset({"how_to", "recipe", "compiled_prompt"})

# The role a legacy `process(..., provenance_kind=...)` call acts as.
_ROLE_FOR_PROVENANCE: dict[ProvenanceKind, PrincipalRole] = {
    "verified_system_of_record": "system_of_record",
    "agent_inference": "agent",
    "self_reported": "customer",
}

# The card each outcome is shown as.
_CARD_STATUS: dict[str, StatusKind] = {
    "auto_commit": "completed",
    "curator_queue": "waiting",
    "refused": "blocked",
    "discard": "warning",
}
_AUDIT_DECISION = {"auto_commit": "allow", "curator_queue": "escalate"}


@runtime_checkable
class WriteAuthorizer(Protocol):
    """Legacy per-category gate: may an actor write a category at all?"""

    def may_write(self, actor: str, category: str) -> bool: ...


@runtime_checkable
class MemoryReporter(Protocol):
    """Where memory decisions are shown (a `StageReporter` satisfies this)."""

    def emit(
        self,
        stage: str,
        status: StatusKind,
        tokens_delta: int | None = None,
        detail: str | None = None,
        *,
        event_kind: Any = ...,
        summary: str | None = None,
        fields: Any = None,
    ) -> None: ...


@instrument("memory")
class MemoryWritePipelineImpl:
    """Chain of Responsibility over the write stages, for every operation."""

    def __init__(
        self,
        semantic: SemanticStore | None = None,
        episodic: EpisodicStore | None = None,
        procedural: ProceduralStore | None = None,
        authorizer: WriteAuthorizer | None = None,
        platform_store: PlatformStore | SqliteConnectionPool | None = None,
        *,
        memory_authorizer: MemoryAuthorizer | None = None,
        audit: AuditLedger | None = None,
        reporter: MemoryReporter | None = None,
        claims: DeterministicClaims | None = None,
    ) -> None:
        settings = get_settings().memory
        # Absent, not defaulted: which CoALA classes exist is the
        # composition root's decision, and rebuilding a store that was
        # deliberately withheld would write to disk a class the
        # configuration says this agent does not have.
        self._semantic = semantic
        self._episodic = episodic
        self._procedural = procedural
        self._authorizer = authorizer
        self._rbac = memory_authorizer
        self._audit = audit
        self._reporter = reporter
        self._claims = claims or DeterministicClaims()
        self._enabled = settings.write_pipeline.enabled
        self._auto_commit = settings.write_pipeline.auto_commit_threshold
        self._discard_below = settings.write_pipeline.discard_threshold
        self._authority_categories = set(settings.write_pipeline.authority_changing_categories)
        self._importance = settings.importance
        self._hints = {
            category: [phrase.casefold() for phrase in phrases]
            for category, phrases in settings.write_pipeline.classification_hints.items()
        }
        self._interrogatives = [
            word.casefold() for word in settings.write_pipeline.interrogative_markers
        ]
        # Durable when the composition root gave us somewhere to keep it.
        # The permission the current operation was checked against, for its card.
        self._checked = ""
        self._curator: CuratorQueue | InMemoryCuratorQueue = (
            CuratorQueue(as_platform_store(platform_store))
            if platform_store is not None
            else InMemoryCuratorQueue()
        )
        self._logger = get_logger("memory")

    # ── the one entry point ──────────────────────────────────────────

    @logged("memory")
    def execute(self, command: MemoryCommand) -> WriteResult:
        """Run one operation through governance; audit and report the outcome."""
        handlers = {
            "create": self._create,
            "update": self._update,
            "delete": self._delete,
            "erase": self._erase,
        }
        handler = handlers.get(command.operation)
        if handler is None:
            raise ValueError(f"'{command.operation}' goes through approve_queued, not execute")
        self._checked = ""
        result = handler(command)
        if not result.permission and self._checked:
            result = result.model_copy(update={"permission": self._checked})
        self._record(command, result)
        return result

    def process(
        self,
        raw_input: str,
        source: str,
        provenance_kind: ProvenanceKind = "self_reported",
        session_id: str = "",
    ) -> WriteResult:
        """Legacy entry: a create, as the role that would carry this provenance."""
        principal = MemoryPrincipal(actor=source, role=_ROLE_FOR_PROVENANCE[provenance_kind])
        return self.execute(
            MemoryCommand(
                principal=principal, operation="create", text=raw_input, session_id=session_id
            )
        )

    # ── create ───────────────────────────────────────────────────────

    def _create(self, command: MemoryCommand) -> WriteResult:
        text = command.text or f"{command.key}: {command.value}"
        stopped = self._precheck(text)
        if stopped is not None:
            return stopped
        if command.require_pattern and not command.key and self._claims.extract(text) is None:
            return WriteResult(
                route="discard",
                written=False,
                reason="no recognised claim in conversational text",
                stage="assertion",
                tier_path=["D1:no-pattern"],
            )
        candidate, tier_path = self._candidate(command, text)
        refusal = self._authorize(command.principal, "create", candidate) or self.authorize(
            candidate, command.principal.actor
        )
        if refusal is not None:
            return self._queue(refusal.model_copy(update={"tier_path": tier_path}))
        conflict = self.detect_conflict(candidate)
        if conflict is not None and conflict.resolution == "curator_queue":
            return self._queue(self._conflicted(candidate, conflict, tier_path))
        if conflict is not None and conflict.resolution == "supersede":
            candidate = self._supersede(conflict.existing, candidate)
        return self.route(candidate).model_copy(update={"tier_path": tier_path})

    def _precheck(self, text: str) -> WriteResult | None:
        """D1 gates that stop a write before anything is extracted."""
        if not self._enabled:
            return WriteResult(route="discard", written=False, reason="write pipeline disabled")
        rejected = self._claims.screen(text)
        if rejected is not None:
            return self._queue(
                WriteResult(
                    route="refused",
                    written=False,
                    reason=f"matches a memory-poisoning pattern ({rejected})",
                    stage="screen",
                    tier_path=["D1:screen"],
                )
            )
        if not self.is_assertion(text):
            return WriteResult(
                route="discard",
                written=False,
                reason="a question states no fact to remember",
                stage="assertion",
                tier_path=["D1:question"],
            )
        return None

    def _candidate(self, command: MemoryCommand, text: str) -> tuple[MemoryFact, list[str]]:
        """Extract, classify, and derive provenance, confidence and importance."""
        claim, tier_path = self._claim(command, text)
        category = command.category or self.classify(text)
        provenance = provenance_for(command.principal)
        fact = MemoryFact(
            key=claim.key,
            value=claim.value,
            category=category,
            confidence=self.score_confidence(provenance, category),
            importance=self.score_importance(category),
            provenance=provenance,
            subject_id=command.principal.subject_id,
            metadata={"session_id": command.session_id} if command.session_id else {},
        )
        return fact, tier_path

    def _claim(self, command: MemoryCommand, text: str) -> tuple[Claim, list[str]]:
        if command.key and command.value:
            explicit = Claim(
                key=normalize_key(command.key), value=command.value, extractor="explicit"
            )
            return explicit, ["D0:explicit"]
        claim = self._claims.extract(text)
        if claim is not None:
            return claim, [f"D1:{claim.extractor}"]
        key, value = self.extract(text)
        # No configured pattern recognised the claim. A model extractor is
        # the next tier; until one is injected the literal fallback stands,
        # and this path is exactly the evidence for whether one is needed.
        return Claim(key=key, value=value, extractor="fallback"), ["D1:fallback", "D2:none"]

    def _conflicted(
        self, candidate: MemoryFact, conflict: MemoryConflict, tier_path: list[str]
    ) -> WriteResult:
        return WriteResult(
            route="curator_queue",
            written=False,
            fact=candidate,
            reason=conflict.reason,
            conflicts_with=[conflict.existing.key],
            stage="conflict",
            tier_path=tier_path,
        )

    def _supersede(self, existing: MemoryFact, candidate: MemoryFact) -> MemoryFact:
        """Retire the prior version; the candidate becomes the next one."""
        version = existing.version + 1
        if self._semantic is not None:
            self._semantic.supersede(existing.key, f"v{version}", existing.subject_id)
        return candidate.model_copy(update={"version": version})

    # ── update / delete / erase ──────────────────────────────────────

    def _update(self, command: MemoryCommand) -> WriteResult:
        """A new version of an existing fact — never an in-place change."""
        existing = self._existing(command)
        if existing is None:
            return self._missing(command)
        provenance = provenance_for(command.principal)
        candidate = existing.model_copy(
            update={
                "value": command.value,
                "provenance": provenance,
                "confidence": self.score_confidence(provenance, existing.category),
                "access_count": 0,
                "last_accessed_at": None,
            }
        )
        refusal = self._authorize(command.principal, "update", candidate)
        if refusal is not None:
            return self._queue(refusal)
        if _authority_rank(provenance.kind) < _authority_rank(existing.provenance.kind):
            conflict = MemoryConflict(
                incoming=candidate,
                existing=existing,
                reason=f"'{provenance.kind}' update to a '{existing.provenance.kind}' fact",
            )
            return self._queue(self._conflicted(candidate, conflict, ["D0:explicit"]))
        committed = self._commit(self._supersede(existing, candidate))
        return committed.model_copy(update={"operation": "update"})

    def _delete(self, command: MemoryCommand) -> WriteResult:
        """Tombstone: the fact leaves recall and stays in history."""
        existing = self._existing(command)
        if existing is None:
            return self._missing(command)
        refusal = self._authorize(command.principal, "delete", existing)
        if refusal is not None:
            return refusal
        assert self._semantic is not None
        self._semantic.tombstone(existing.key, existing.subject_id, command.principal.actor)
        return WriteResult(
            route="auto_commit",
            written=True,
            fact=existing,
            operation="delete",
            reason=f"tombstoned v{existing.version}: {command.reason or 'no reason given'}",
            stage="delete",
        )

    def _erase(self, command: MemoryCommand) -> WriteResult:
        """Remove every version — the data-protection path. The audit keeps no content."""
        existing = self._existing(command)
        if existing is None:
            return self._missing(command)
        refusal = self._authorize(command.principal, "erase", existing)
        if refusal is not None:
            return refusal
        assert self._semantic is not None
        removed = self._semantic.erase(existing.key, existing.subject_id)
        return WriteResult(
            route="auto_commit",
            written=True,
            operation="erase",
            reason=f"erased {removed} version(s) of key#{_key_hash(existing.key)}",
            stage="erase",
        )

    def _existing(self, command: MemoryCommand) -> MemoryFact | None:
        if self._semantic is None:
            return None
        return self._semantic.latest(normalize_key(command.key), command.principal.subject_id)

    def _missing(self, command: MemoryCommand) -> WriteResult:
        return WriteResult(
            route="discard",
            written=False,
            operation=command.operation,
            reason=f"no current fact '{normalize_key(command.key)}' to {command.operation}",
            stage=command.operation,
        )

    # ── the stages, individually callable ────────────────────────────

    def is_assertion(self, raw_input: str) -> bool:
        """Whether this text claims something, rather than asks something.

        A question is not a candidate memory. Without this the curator
        queue fills with "where is my order?" — every message becomes a
        fact awaiting adjudication, and a queue that contains everything
        is a queue nobody reads.
        """
        text = raw_input.strip().casefold()
        if not text or text.endswith("?"):
            return False
        return not any(text.startswith(word) for word in self._interrogatives)

    def extract(self, raw_input: str) -> tuple[str, str]:
        """The literal fallback extractor, used when no configured pattern matches."""
        text = raw_input.strip()
        if ":" in text:
            key, _, value = text.partition(":")
            return key.strip().casefold().replace(" ", "_"), value.strip()
        return "_".join(text.casefold().split()[:6]), text

    def classify(self, raw_input: str) -> str:
        """Assign a category from the configured phrase hints."""
        lowered = raw_input.casefold()
        for category, phrases in self._hints.items():
            if any(phrase in lowered for phrase in phrases):
                return category
        return "general"

    def attach_provenance(self, source: str, kind: ProvenanceKind) -> Provenance:
        """Record who said it and how they knew (legacy helper)."""
        return Provenance(source=source, kind=kind, ingestion_time=utc_now())

    def detect_conflict(self, candidate: MemoryFact) -> MemoryConflict | None:
        """Check the candidate against what this subject already believes.

        Last-write-wins is only legitimate when the new fact has at least
        the authority of the old one. A self-reported claim overwriting a
        verified system-of-record fact is not a temporal update; it is the
        poisoning case, and it goes to a human.
        """
        prior = (
            self._semantic.latest(candidate.key, candidate.subject_id) if self._semantic else None
        )
        if prior is None or prior.value == candidate.value:
            return None
        if _authority_rank(candidate.provenance.kind) >= _authority_rank(prior.provenance.kind):
            return MemoryConflict(
                incoming=candidate,
                existing=prior,
                resolution="supersede",
                reason="newer fact of equal or greater authority",
            )
        return MemoryConflict(
            incoming=candidate,
            existing=prior,
            resolution="curator_queue",
            reason=f"'{candidate.provenance.kind}' claim contradicts a '{prior.provenance.kind}' fact",
        )

    def score_confidence(self, provenance: Provenance, category: str) -> float:
        """How much this claim is believed — from provenance, never from tone."""
        base = get_settings().memory.provenance_weights.get(provenance.kind, 0.0)
        if category in self._authority_categories:
            # An authority claim is the one place extra caution is free.
            return base * 0.5
        return base

    def score_importance(self, category: str) -> float:
        """How much this fact matters, decided at write time from its category."""
        return self._importance.by_category.get(category, self._importance.default)

    def authorize(self, candidate: MemoryFact, actor: str) -> WriteResult | None:
        """The unconditional rules, plus the legacy per-category gate.

        The authority-changing check cannot be granted away: no confidence
        score and no role's grants permit it from a conversational claim.
        """
        if (
            candidate.category in self._authority_categories
            and candidate.provenance.kind != "verified_system_of_record"
        ):
            self._logger.warning(
                "Refusing authority-changing write from unverified source",
                actor=actor,
                category=candidate.category,
                key=candidate.key,
            )
            return self._refused(
                candidate,
                "authority-changing claims are never written from conversational "
                "assertion; no role grants itself authority",
            )
        if self._authorizer is not None and not self._authorizer.may_write(
            actor, candidate.category
        ):
            return self._refused(
                candidate, f"actor '{actor}' may not write category '{candidate.category}'"
            )
        return None

    def route(self, candidate: MemoryFact) -> WriteResult:
        """Auto-commit, queue for a curator, or discard."""
        if candidate.confidence < self._discard_below:
            return WriteResult(
                route="discard",
                written=False,
                fact=candidate,
                reason=f"confidence {candidate.confidence:.2f} below discard threshold",
                stage="route",
            )
        if candidate.confidence < self._auto_commit:
            return self._queue(
                WriteResult(
                    route="curator_queue",
                    written=False,
                    fact=candidate,
                    reason=f"confidence {candidate.confidence:.2f} below auto-commit threshold",
                    stage="route",
                )
            )
        return self._commit(candidate)

    # ── curator ──────────────────────────────────────────────────────

    def curator_queue(self) -> list[WriteResult]:
        """Writes awaiting human adjudication — flagged, never silently dropped."""
        return [result for _, result in self._curator.pending()]

    def pending_with_ids(self) -> list[tuple[int, WriteResult]]:
        """The curator queue with each entry's durable id."""
        return self._curator.pending()

    def approve(self, index: int, curator: str) -> WriteResult:
        """Legacy: approve the index-th pending write as a curator."""
        queue_id, _ = self._curator.pending()[index]
        return self.approve_queued(queue_id, MemoryPrincipal(actor=curator, role="curator"))

    def approve_queued(self, queue_id: int, principal: MemoryPrincipal) -> WriteResult:
        """A curator confirms a queued write, committing it as verified."""
        pending = dict(self._curator.pending()).get(queue_id)
        if pending is None or pending.fact is None:
            return WriteResult(route="discard", written=False, reason="nothing to approve")
        self._checked = ""
        refusal = self._authorize(principal, "approve", pending.fact)
        if refusal is not None:
            return refusal
        confirmed = Provenance(
            source=principal.actor,
            kind="verified_system_of_record",
            confirmed=True,
            confirmed_at=utc_now(),
        )
        fact = pending.fact.model_copy(update={"confidence": 1.0, "provenance": confirmed})
        prior = self._semantic.latest(fact.key, fact.subject_id) if self._semantic else None
        if prior is not None:
            fact = self._supersede(prior, fact)
        self._curator.decide(queue_id, "approved", principal.actor)
        result = self._commit(fact).model_copy(
            update={"operation": "approve", "curator_id": queue_id, "permission": self._checked}
        )
        self._record(MemoryCommand(principal=principal, operation="approve", key=fact.key), result)
        return result

    def permits(
        self, principal: MemoryPrincipal, operation: Any, memory_type: MemoryType, category: str
    ) -> str | None:
        """Why a principal may not do this, or None if it may (or no RBAC is wired)."""
        if self._rbac is None:
            return None
        decision = self._rbac.authorize(
            principal, memory_permission(memory_type, operation, category)
        )
        return None if decision.allowed else decision.reason

    def report(
        self,
        stage: str,
        status: StatusKind,
        summary: str,
        fields: dict[str, Any],
        *,
        quiet: bool = False,
    ) -> None:
        """Report one memory operation the facade performs itself: a card, or quietly (log and tally only)."""
        if self._reporter is not None:
            self._reporter.emit(
                stage, status, event_kind="memory", summary=summary, fields=fields, quiet=quiet
            )

    def recent(self, limit: int = 10) -> list[MemoryFact]:
        """Most recently committed semantic facts — an inspection helper."""
        return self._semantic.search(MemoryQuery(limit=limit)) if self._semantic else []

    # ── internals ────────────────────────────────────────────────────

    def _authorize(
        self, principal: MemoryPrincipal, operation: Any, fact: MemoryFact
    ) -> WriteResult | None:
        """RBAC for one operation, when the composition root injected RBAC.

        The permission checked is remembered either way, so an *allowed*
        write's card says which grant allowed it — not only a refusal's.
        """
        if self._rbac is None:
            self._checked = "— (no RBAC wired)"
            return None
        permission = memory_permission(store_type(fact.category), operation, fact.category)
        decision = self._rbac.authorize(principal, permission)
        if decision.allowed:
            self._checked = f"{permission} ✓ via {decision.granted_by or 'grant'}"
            return None
        self._checked = f"{permission} ✗"
        return self._refused(fact, decision.reason, permission=permission, operation=operation)

    def _refused(
        self,
        fact: MemoryFact,
        reason: str,
        *,
        permission: str = "",
        operation: Any = "create",
    ) -> WriteResult:
        return WriteResult(
            route="refused",
            written=False,
            fact=fact,
            reason=reason,
            stage="authorization",
            permission=permission,
            operation=operation,
        )

    def _commit(self, candidate: MemoryFact) -> WriteResult:
        store = self._store_for(store_type(candidate.category))
        if store is None:
            return WriteResult(
                route="discard",
                written=False,
                fact=candidate,
                reason="this agent has no store for that memory type",
                stage="route",
            )
        result = store.write(candidate, candidate.provenance)
        return result.model_copy(update={"stage": "route"})

    def _queue(self, result: WriteResult) -> WriteResult:
        """Hold a write for a human, and keep the record either way."""
        return result.model_copy(update={"curator_id": self._curator.put(result)})

    def _store_for(
        self, memory_type: MemoryType
    ) -> SemanticStore | EpisodicStore | ProceduralStore | None:
        """The store a memory type belongs to, if this agent has it."""
        if memory_type == "episodic":
            return self._episodic
        if memory_type == "procedural":
            return self._procedural
        return self._semantic

    def _record(self, command: MemoryCommand, result: WriteResult) -> None:
        """Audit the decision and show it as a card — except a question's no-op."""
        if result.stage == "assertion":
            return
        if self._audit is not None:
            self._audit.record(_audit_entry(command, result))
        if self._reporter is not None:
            self._reporter.emit(
                f"memory_{'write' if command.operation in ('create', 'approve') else command.operation}",
                _CARD_STATUS[result.route],
                event_kind="memory",
                summary=_headline(command, result),
                fields=_card_fields(command, result),
            )


def store_type(category: str) -> MemoryType:
    """Which CoALA store a category belongs in."""
    if category in _EVENT_CATEGORIES:
        return "episodic"
    if category in _PROCEDURE_CATEGORIES:
        return "procedural"
    return "semantic"


def _authority_rank(kind: ProvenanceKind) -> int:
    """Ordering over provenance kinds, strongest first."""
    return {"verified_system_of_record": 3, "agent_inference": 2, "self_reported": 1}[kind]


def _key_hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _headline(command: MemoryCommand, result: WriteResult) -> str:
    fact = result.fact
    subject = (
        f"{fact.category} {fact.key}" if fact is not None and command.operation != "erase" else ""
    )
    return f"{command.operation} {subject} → {result.route}".replace("  ", " ")


def _card_fields(command: MemoryCommand, result: WriteResult) -> dict[str, Any]:
    fact = result.fact
    fields: dict[str, Any] = {
        "operation": command.operation,
        "principal": f"{command.principal.actor} ({command.principal.role})",
        "recorded_for": (
            f"customer {command.principal.subject_id}" if command.principal.subject_id else "shared"
        ),
        "route": result.route,
        "reason": result.reason,
        "tier_path": " · ".join(result.tier_path) or "—",
        "permission": result.permission or "—",
        "curator_queue_id": result.curator_id,
    }
    if fact is not None and command.operation != "erase":
        fields.update(
            {
                "memory": f"{fact.memory_type} (facts believed true)",
                "fact": f"{fact.key} → {fact.value}",
                "category": fact.category,
                "provenance": fact.provenance.kind,
                "confidence": round(fact.confidence, 2),
                "importance": round(fact.importance, 2),
                "version": fact.version,
                "conflicts_with": ", ".join(result.conflicts_with) or "—",
            }
        )
    return fields


def _audit_entry(command: MemoryCommand, result: WriteResult) -> AuditEntry:
    key = result.fact.key if result.fact is not None else command.key
    # Erasure keeps no content — not even the key — only that it happened.
    shown = f"key#{_key_hash(key)}" if command.operation == "erase" else key
    return AuditEntry(
        trace_id=f"memory-{command.operation}-{_key_hash(shown + command.principal.actor)}",
        actor=command.principal.actor,
        action=f"memory_{command.operation}",
        authority_context={
            "role": command.principal.role,
            "subject": command.principal.subject_id,
            "permission": result.permission,
            "key": shown,
        },
        decision=_AUDIT_DECISION.get(result.route, "deny"),  # type: ignore[arg-type]
        rationale=result.reason,
    )
