"""One entry point across all four CoALA stores (LLD §8, Facade).

`customer_support_agent` asks memory questions; it does not choose
storage engines. Which store answers a given question is a decision this
facade owns, which is what keeps the information-hiding discipline
intact as stores change underneath.

Writes go through the governed pipeline (`submit`). Reads are a cascade,
cheapest tier first, and each tier runs only if the one before it did
not find enough:

    R1 exact    key or identifier in the query  → SQL equality, rg over logs
    R2 lexical  the query's words               → FTS5 BM25, rg over logs
    R3 dense    only if R1 + R2 came up short   → the injected embedding scorer

Every recall says which tier answered and what it cost, and reinforces
the facts it returned.
"""

import re
import uuid
from typing import Any

from agent_platform.audit import AuditLedger
from agent_platform.config import get_settings
from agent_platform.core.persistence import PlatformStore
from agent_platform.logging import get_logger, instrument
from agent_platform.memory.deterministic import DeterministicClaims, normalize_key
from agent_platform.memory.episodic.memory import EpisodicStore
from agent_platform.memory.governance import MemoryAuthorizer
from agent_platform.memory.procedural.store import SkillStore
from agent_platform.memory.semantic.memory import SemanticStore
from agent_platform.memory.store import CompositeRetrievalScorer, RelevanceScorer
from agent_platform.memory.types import (
    MemoryCommand,
    MemoryFact,
    MemoryPrincipal,
    MemoryQuery,
    Provenance,
    ProvenanceKind,
    RecallReport,
    RecallTier,
    ScoredFact,
    TextRedactor,
    WriteResult,
)
from agent_platform.memory.working.memory import WorkingMemoryStore
from agent_platform.memory.write_pipeline import MemoryReporter, MemoryWritePipelineImpl

# One cascade candidate: the fact, its relevance, and the tier that found it.
_Hit = tuple[MemoryFact, float, RecallTier]

_WORD = re.compile(r"[a-z0-9]{3,}")
_TIER_ORDER: tuple[RecallTier, ...] = ("exact", "lexical", "dense")


def id_of(fact: MemoryFact) -> tuple[str, str, str, int]:
    """What makes two retrieved facts the same fact."""
    return (fact.memory_type, fact.subject_id, fact.key, fact.version)


def _words(text: str) -> list[str]:
    return list(dict.fromkeys(_WORD.findall(text.casefold())))


def _overlap(terms: list[str], text: str) -> float:
    """Share of the query's words a text contains — grep's relevance."""
    if not terms:
        return 0.0
    folded = text.casefold()
    return sum(1 for term in terms if term in folded) / len(terms)


def _best_tier(found: list[_Hit]) -> RecallTier | None:
    tiers = {tier for _, _, tier in found}
    return next((tier for tier in _TIER_ORDER if tier in tiers), None)


@instrument("memory")
class MemoryFacade:
    """Read and write across working, episodic, semantic and procedural memory."""

    def __init__(
        self,
        working: WorkingMemoryStore | None = None,
        episodic: EpisodicStore | None = None,
        semantic: SemanticStore | None = None,
        procedural: SkillStore | None = None,
        pipeline: MemoryWritePipelineImpl | None = None,
        relevance: RelevanceScorer | None = None,
        platform_store: PlatformStore | None = None,
        redactor: TextRedactor | None = None,
        memory_authorizer: MemoryAuthorizer | None = None,
        audit: AuditLedger | None = None,
        reporter: MemoryReporter | None = None,
    ) -> None:
        """Compose whichever CoALA stores this deployment was given.

        A store left out is *absent*, not defaulted. Which classes an
        agent has is the composition root's decision, and a facade that
        quietly rebuilt the one it was denied would make that decision
        unenforceable — the store would be off in configuration and
        writing to disk in fact. `complete()` builds all four for the
        callers that want them.

        The redactor is why memory can be trusted with customer text.
        Redaction at the *write* is the only placement that works: a
        fact is written once and read for the rest of the deployment's
        life, so redacting on the way out leaves the original on disk
        and merely hides it from whoever asked politely.
        """
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural
        self.pipeline = pipeline or MemoryWritePipelineImpl(
            semantic=semantic,
            episodic=episodic,
            procedural=procedural,
            platform_store=platform_store,
            memory_authorizer=memory_authorizer,
            audit=audit,
            reporter=reporter,
        )
        self._relevance = relevance
        self._redactor = redactor
        self._scorer = CompositeRetrievalScorer()
        self._claims = DeterministicClaims()
        self._cascade = get_settings().memory.deterministic
        self.last_recall = RecallReport()
        self._logger = get_logger("memory")

    @classmethod
    def complete(
        cls,
        working: WorkingMemoryStore | None = None,
        episodic: EpisodicStore | None = None,
        semantic: SemanticStore | None = None,
        procedural: SkillStore | None = None,
        pipeline: MemoryWritePipelineImpl | None = None,
        relevance: RelevanceScorer | None = None,
        platform_store: PlatformStore | None = None,
        redactor: TextRedactor | None = None,
    ) -> "MemoryFacade":
        """A facade holding all four CoALA classes, building any not given."""
        return cls(
            working=working or WorkingMemoryStore(),
            episodic=episodic or EpisodicStore(),
            semantic=semantic or SemanticStore(),
            procedural=procedural or SkillStore(),
            pipeline=pipeline,
            relevance=relevance,
            platform_store=platform_store,
            redactor=redactor,
        )

    def _redact(self, text: str) -> str:
        """Strip personal data, when this deployment was given a redactor."""
        return self._redactor.redact(text) if self._redactor is not None else text

    def submit(self, command: MemoryCommand) -> WriteResult:
        """Any governed operation. Redacted before the pipeline ever sees it."""
        redacted = command.model_copy(
            update={"text": self._redact(command.text), "value": self._redact(command.value)}
        )
        return self.pipeline.execute(redacted)

    def remember(
        self,
        raw_input: str,
        source: str,
        provenance_kind: ProvenanceKind = "self_reported",
        session_id: str = "",
    ) -> WriteResult:
        """Legacy: offer text to memory as the role that carries this provenance."""
        return self.pipeline.process(self._redact(raw_input), source, provenance_kind, session_id)

    def note(
        self,
        key: str,
        value: str,
        session_id: str,
        source: str,
        category: str = "",
        provenance_kind: ProvenanceKind = "verified_system_of_record",
    ) -> WriteResult:
        """Hold one compact value in the session's working memory."""
        if self.working is None:
            return WriteResult(route="discard", written=False, reason="no working memory")
        value = self._redact(value)
        memory = self.working.get(session_id)
        memory.set(key, value)
        fact = MemoryFact(
            key=key,
            value=value,
            memory_type="working",
            category=category,
            provenance=Provenance(source=source, kind=provenance_kind),
            metadata={"session_id": session_id},
        )
        return WriteResult(route="auto_commit", written=True, fact=fact, reason="in active window")

    def end_session(self, session_id: str) -> None:
        """Drop one conversation's working memory."""
        if self.working is None:
            return
        self.working.clear(session_id)
        self._logger.debug("Working memory cleared", session_id=session_id)

    def record_event(
        self,
        description: str,
        source: str,
        session_id: str,
        principal: MemoryPrincipal | None = None,
        category: str = "event",
        key: str = "",
    ) -> WriteResult:
        """Append to the episodic log — authorized, but never subject to conflict.

        `category` and `key` let a dream record a session summary the same
        way: it too is what happened (built from recorded tool results),
        not a claim, so it takes this path rather than confidence routing.

        An event is what happened, not a claim about what is true, so it
        skips conflict and confidence routing. It does not skip the
        permission check: appending to a customer's history is still an
        operation someone must be allowed to perform.
        """
        if self.episodic is None:
            return WriteResult(route="discard", written=False, reason="no episodic memory")
        actor = principal or MemoryPrincipal(actor=source, role="agent")
        refused = self.pipeline.permits(actor, "create", "episodic", category)
        if refused is not None:
            return WriteResult(
                route="refused", written=False, reason=refused, stage="authorization"
            )
        fact = MemoryFact(
            # One key per event, not per session: a shared key made every
            # event in a conversation the same record to recall and forget.
            key=key or f"event_{session_id}_{uuid.uuid4().hex}",
            value=self._redact(description),
            memory_type="episodic",
            category=category,
            importance=self.pipeline.score_importance(category),
            metadata={"session_id": session_id},
            provenance=Provenance(source=source),
            subject_id=actor.subject_id,
        )
        result = self.episodic.write(fact, fact.provenance)
        self.pipeline.report(
            "memory_write",
            "completed",
            f"episodic event recorded: {fact.value[:60]}",
            _event_fields(fact),
            quiet=True,
        )
        return result

    async def recall(self, query: MemoryQuery) -> list[ScoredFact]:
        """Retrieve across long-term stores, cheapest tier first (see module docstring)."""
        report = RecallReport()
        found = self._exact(query, report)
        if len(found) >= self._cascade.min_exact_hits:
            report.skipped = ["lexical", "dense"]
        else:
            found += self._lexical(query, report, exclude=found)
            if len(found) < self._cascade.dense_below:
                found += await self._dense(query, report, exclude=found)
            else:
                report.skipped.append("dense")
        report.answered_by = _best_tier(found)
        ranked = self._scorer.rank(
            [fact for fact, _, _ in found],
            [relevance for _, relevance, _ in found],
            query.limit,
            [tier for _, _, tier in found],
        )
        self._reinforce(ranked)
        self._report_recall(query, report, ranked)
        self.last_recall = report
        return ranked

    def skill(self, name: str) -> MemoryFact | None:
        """The active version of a how-to artifact."""
        return self.procedural.get(name) if self.procedural else None

    def forget(self, key: str, principal: MemoryPrincipal | None = None) -> int:
        """Erase a fact — governed, audited, reported. Returns versions removed (legacy)."""
        if self.semantic is None:
            return 0
        actor = principal or MemoryPrincipal(actor="forget", role="data_protection")
        result = self.submit(MemoryCommand(principal=actor, operation="erase", key=key))
        return 1 if result.written else 0

    def curator_queue(self) -> list[WriteResult]:
        """Writes held for human adjudication."""
        return self.pipeline.curator_queue()

    # ── the read cascade ─────────────────────────────────────────────

    def _exact(self, query: MemoryQuery, report: RecallReport) -> list[_Hit]:
        """R1: an asked-for key, or an identifier the query names."""
        keys = [normalize_key(query.key)] if query.key else []
        identifiers = self._claims.identifiers(query.text)
        hits: list[_Hit] = []
        if self.semantic is not None:
            facts = self.semantic.exact(keys, identifiers, query.subject_id, query.limit)
            hits += [(fact, 1.0, "exact") for fact in facts]
        if self.episodic is not None and identifiers:
            events = self.episodic.search_terms(identifiers, query.subject_id, query.limit)
            hits += [(fact, 1.0, "exact") for fact in events]
        report.candidates["exact"] = len(hits)
        return hits

    def _lexical(self, query: MemoryQuery, report: RecallReport, exclude: list[_Hit]) -> list[_Hit]:
        """R2: the query's words — FTS5 BM25 over facts, grep over event logs."""
        seen = {id_of(fact) for fact, _, _ in exclude}
        hits: list[_Hit] = []
        if self.semantic is not None and query.text:
            for fact, score in self.semantic.lexical(query.text, query.subject_id, query.limit):
                hits.append((fact, score, "lexical"))
        if self.episodic is not None and query.text:
            terms = _words(query.text)
            for fact in self.episodic.search_terms(terms, query.subject_id, query.limit):
                overlap = _overlap(terms, fact.value)
                if overlap:
                    hits.append((fact, overlap, "lexical"))
        fresh = [hit for hit in hits if id_of(hit[0]) not in seen]
        report.candidates["lexical"] = len(fresh)
        return fresh

    async def _dense(
        self, query: MemoryQuery, report: RecallReport, exclude: list[_Hit]
    ) -> list[_Hit]:
        """R3: embeddings — reached only when the cheap tiers came up short."""
        if self._relevance is None or not query.text:
            report.skipped.append("dense")
            return []
        seen = {id_of(fact) for fact, _, _ in exclude}
        pool = [
            fact
            for fact in (
                self.semantic.read(MemoryQuery(subject_id=query.subject_id, limit=query.limit * 5))
                if self.semantic
                else []
            )
            if id_of(fact) not in seen
        ]
        if not pool:
            report.candidates["dense"] = 0
            return []
        scores = await self._relevance.rank(query.text, [fact.value for fact in pool])
        report.embed_calls += 1
        hits: list[_Hit] = [(fact, score, "dense") for fact, score in zip(pool, scores)]
        report.candidates["dense"] = len(hits)
        return hits

    def _reinforce(self, ranked: list[ScoredFact]) -> None:
        if self.semantic is not None:
            self.semantic.reinforce(
                [item.fact for item in ranked if item.fact.memory_type == "semantic"]
            )

    def _report_recall(
        self, query: MemoryQuery, report: RecallReport, ranked: list[ScoredFact]
    ) -> None:
        answered = report.answered_by or "nothing"
        self.pipeline.report(
            "memory_recall",
            "completed",
            f"answered at {answered} ({report.embed_calls} embed calls)",
            _recall_fields(query, report, ranked),
            quiet=True,
        )


def _event_fields(fact: MemoryFact) -> dict[str, Any]:
    """An episodic event, as the log shows it: what was recorded, and under which id."""
    return {"event": fact.value, "event_id": fact.key}


def _recall_fields(
    query: MemoryQuery, report: RecallReport, ranked: list[ScoredFact]
) -> dict[str, Any]:
    """What a recall found and how, for the log; the turn summary carries the count.

    An event is named by its text, not the id it is stored under. The
    scoring breakdown stays, one row per fact, because "why was this
    recalled?" is the question that makes a false recall diagnosable.
    """
    return {
        "query": query.text or query.key,
        "answered_by": report.answered_by or "nothing",
        "candidates": report.candidates,
        "skipped": ", ".join(report.skipped),
        "embed_calls": report.embed_calls,
        "recalled": [
            f"{_named(item.fact)} [{item.tier}] score={item.score:.2f} "
            f"rel={item.relevance:.2f} rec={item.recency:.2f} imp={item.importance:.2f} "
            f"auth={item.authority:.2f} freq={item.frequency:.2f} "
            f"conf={item.confidence:.2f} valid={item.temporal_validity:.0f}"
            for item in ranked
        ],
    }


def _named(fact: MemoryFact) -> str:
    """A recalled fact as a reader names it: the text of an event, `key=value` for a fact."""
    return fact.value[:60] if fact.memory_type == "episodic" else f"{fact.key}={fact.value[:40]}"
