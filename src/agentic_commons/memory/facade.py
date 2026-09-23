"""One entry point across all four CoALA stores (LLD §8, Facade).

`customer_support_agent` asks memory questions; it does not choose
storage engines. Which store answers a given question is a decision this
facade owns, which is what keeps E08's information-hiding discipline
intact as stores change underneath.
"""

from agentic_commons.logging import get_logger, instrument
from agentic_commons.memory.episodic.memory import EpisodicStore
from agentic_commons.memory.procedural.memory import ProceduralStore
from agentic_commons.memory.semantic.memory import SemanticStore
from agentic_commons.memory.store import CompositeRetrievalScorer, RelevanceScorer
from agentic_commons.memory.types import (
    MemoryFact,
    MemoryQuery,
    Provenance,
    ProvenanceKind,
    ScoredFact,
    TextRedactor,
    WriteResult,
)
from agentic_commons.memory.working.memory import WorkingMemory
from agentic_commons.memory.write_pipeline import MemoryWritePipelineImpl
from agentic_commons.persistence import PlatformStore


@instrument("memory")
class MemoryFacade:
    """Read and write across working, episodic, semantic and procedural memory."""

    def __init__(
        self,
        working: WorkingMemory | None = None,
        episodic: EpisodicStore | None = None,
        semantic: SemanticStore | None = None,
        procedural: ProceduralStore | None = None,
        pipeline: MemoryWritePipelineImpl | None = None,
        relevance: RelevanceScorer | None = None,
        platform_store: PlatformStore | None = None,
        redactor: TextRedactor | None = None,
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
        )
        self._relevance = relevance
        self._redactor = redactor
        self._scorer = CompositeRetrievalScorer()
        self._logger = get_logger("memory")

    @classmethod
    def complete(
        cls,
        working: WorkingMemory | None = None,
        episodic: EpisodicStore | None = None,
        semantic: SemanticStore | None = None,
        procedural: ProceduralStore | None = None,
        pipeline: MemoryWritePipelineImpl | None = None,
        relevance: RelevanceScorer | None = None,
        platform_store: PlatformStore | None = None,
        redactor: TextRedactor | None = None,
    ) -> "MemoryFacade":
        """A facade holding all four CoALA classes, building any not given."""
        return cls(
            working=working or WorkingMemory(),
            episodic=episodic or EpisodicStore(),
            semantic=semantic or SemanticStore(platform_store),
            procedural=procedural or ProceduralStore(platform_store),
            pipeline=pipeline,
            relevance=relevance,
            platform_store=platform_store,
            redactor=redactor,
        )

    def _redact(self, text: str) -> str:
        """Strip personal data, when this deployment was given a redactor."""
        return self._redactor.redact(text) if self._redactor is not None else text

    def remember(
        self,
        raw_input: str,
        source: str,
        provenance_kind: ProvenanceKind = "self_reported",
        session_id: str = "",
    ) -> WriteResult:
        """Offer something to memory. The pipeline decides whether it lands."""
        return self.pipeline.process(
            self._redact(raw_input), source, provenance_kind, session_id
        )

    def note(
        self,
        key: str,
        value: str,
        session_id: str,
        source: str,
        category: str = "",
        provenance_kind: ProvenanceKind = "verified_system_of_record",
    ) -> WriteResult:
        """Hold a fact in this session's active window.

        This is the short-term path, and it is deliberately not the write
        pipeline: what the customer just selected, what a tool just
        returned, who is verified. Those are true for this conversation
        and meaningless outside it, so they belong in the window rather
        than in long-term storage, where they would need conflict
        resolution and a curator.

        Default provenance is the system of record, because that is what
        almost every caller is: a tool result or a database read. A
        caller relaying something the *customer* said passes
        `self_reported` and it is weighted down at retrieval.
        """
        if self.working is None:
            return WriteResult(route="discard", written=False, reason="no working memory")
        fact = MemoryFact(
            key=key,
            value=self._redact(value),
            memory_type="working",
            category=category,
            metadata={"session_id": session_id},
            provenance=Provenance(source=source, kind=provenance_kind),
        )
        return self.working.write(fact, fact.provenance)

    def active_window(self, session_id: str, limit: int = 0) -> list[MemoryFact]:
        """What this session currently holds, newest first."""
        if self.working is None:
            return []
        query = MemoryQuery(session_id=session_id)
        return self.working.read(query.model_copy(update={"limit": limit or query.limit}))

    def window_size(self, session_id: str) -> int:
        """How many facts this session is holding.

        Asked here rather than by reaching through to the store: callers
        ask memory questions, and one that walks into `.working` breaks
        the moment a deployment is composed without that class.
        """
        return self.working.size(session_id) if self.working else 0

    def end_session(self, session_id: str) -> None:
        """Drop one conversation's window. Long-term memory is untouched."""
        if self.working is None:
            return
        self.working.clear(session_id)
        self._logger.debug("Working memory cleared", session_id=session_id)

    def record_event(self, description: str, source: str, session_id: str) -> WriteResult:
        """Append to the episodic log — events are never subject to conflict."""
        if self.episodic is None:
            return WriteResult(route="discard", written=False, reason="no episodic memory")
        fact = MemoryFact(
            key=f"event_{session_id}",
            value=self._redact(description),
            memory_type="episodic",
            category="event",
            metadata={"session_id": session_id},
            provenance=Provenance(source=source),
        )
        return self.episodic.write(fact, fact.provenance)

    async def recall(self, query: MemoryQuery) -> list[ScoredFact]:
        """Retrieve across long-term stores, ranked by the composite score."""
        candidates = (self.semantic.read(query) if self.semantic else []) + (
            self.episodic.read(query) if self.episodic else []
        )
        if not candidates:
            return []
        relevances = await self._relevances(query, candidates)
        return self._scorer.rank(candidates, relevances, query.limit)

    def procedure(self, name: str) -> MemoryFact | None:
        """The active version of a how-to artifact."""
        return self.procedural.get(name) if self.procedural else None

    def forget(self, key: str) -> int:
        """Hard-delete every semantic fact for a key (revocation, E16)."""
        return self.semantic.revoke(key) if self.semantic else 0

    def curator_queue(self) -> list[WriteResult]:
        """Writes held for human adjudication."""
        return self.pipeline.curator_queue()

    async def _relevances(self, query: MemoryQuery, facts: list[MemoryFact]) -> list[float]:
        """Relevance per candidate, from the injected scorer or lexical overlap."""
        text = query.text or query.key
        if not text:
            return [1.0] * len(facts)
        if self._relevance is not None:
            return await self._relevance.rank(text, [fact.value for fact in facts])
        needle = text.casefold()
        return [1.0 if needle in fact.value.casefold() else 0.0 for fact in facts]
