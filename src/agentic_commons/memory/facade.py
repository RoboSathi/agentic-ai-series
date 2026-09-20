"""One entry point across all four CoALA stores (LLD §8, Facade).

`customer_support_agent` asks memory questions; it does not choose
storage engines. Which store answers a given question is a decision this
facade owns, which is what keeps E08's information-hiding discipline
intact as stores change underneath.
"""

from agentic_commons.config import get_settings
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
    ) -> None:
        self.working = working or WorkingMemory()
        self.episodic = episodic or EpisodicStore()
        self.semantic = semantic or SemanticStore(platform_store)
        self.procedural = procedural or ProceduralStore(platform_store)
        self.pipeline = pipeline or MemoryWritePipelineImpl(
            semantic=self.semantic,
            episodic=self.episodic,
            procedural=self.procedural,
            platform_store=platform_store,
        )
        self._relevance = relevance
        self._scorer = CompositeRetrievalScorer()
        self._enabled = get_settings().memory.enabled
        self._logger = get_logger("memory")

    def remember(
        self,
        raw_input: str,
        source: str,
        provenance_kind: ProvenanceKind = "self_reported",
        session_id: str = "",
    ) -> WriteResult:
        """Offer something to memory. The pipeline decides whether it lands."""
        if not self._enabled:
            return WriteResult(route="discard", written=False, reason="memory disabled")
        return self.pipeline.process(raw_input, source, provenance_kind, session_id)

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
        if not self._enabled:
            return WriteResult(route="discard", written=False, reason="memory disabled")
        fact = MemoryFact(
            key=key,
            value=value,
            memory_type="working",
            category=category,
            metadata={"session_id": session_id},
            provenance=Provenance(source=source, kind=provenance_kind),
        )
        return self.working.write(fact, fact.provenance)

    def active_window(self, session_id: str, limit: int = 0) -> list[MemoryFact]:
        """What this session currently holds, newest first."""
        query = MemoryQuery(session_id=session_id)
        return self.working.read(query.model_copy(update={"limit": limit or query.limit}))

    def end_session(self, session_id: str) -> None:
        """Drop one conversation's window. Long-term memory is untouched."""
        self.working.clear(session_id)
        self._logger.debug("Working memory cleared", session_id=session_id)

    def record_event(self, description: str, source: str, session_id: str) -> WriteResult:
        """Append to the episodic log — events are never subject to conflict."""
        fact = MemoryFact(
            key=f"event_{session_id}",
            value=description,
            memory_type="episodic",
            category="event",
            metadata={"session_id": session_id},
            provenance=Provenance(source=source),
        )
        return self.episodic.write(fact, fact.provenance)

    async def recall(self, query: MemoryQuery) -> list[ScoredFact]:
        """Retrieve across long-term stores, ranked by the composite score."""
        if not self._enabled:
            return []
        candidates = self.semantic.read(query) + self.episodic.read(query)
        if not candidates:
            return []
        relevances = await self._relevances(query, candidates)
        return self._scorer.rank(candidates, relevances, query.limit)

    def procedure(self, name: str) -> MemoryFact | None:
        """The active version of a how-to artifact."""
        return self.procedural.get(name)

    def forget(self, key: str) -> int:
        """Hard-delete every semantic fact for a key (revocation, E16)."""
        return self.semantic.revoke(key)

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
