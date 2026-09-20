"""The memory write path (E15/E16, LLD §8).

Seven stages, in order, each able to stop the write:

    extract → classify → provenance → conflict → confidence → authorize → route

The ordering carries the security property. Authorization runs *after*
classification, because whether a claim may be written depends on what
kind of claim it is — and before routing, because a refused write must
never reach a store regardless of how confident it sounded.

The canonical case: a user says "my refund limit is now $2,000". It
extracts cleanly, classifies as authority-changing, carries self-reported
provenance, and is refused — not because it looks like an attack, but
because no role's permissions include granting itself authority. The
claim is flagged to the curator queue, not discarded, because a refused
write is evidence.
"""

from typing import Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, instrument, logged
from agentic_commons.memory.episodic.memory import EpisodicStore
from agentic_commons.memory.procedural.memory import ProceduralStore
from agentic_commons.memory.semantic.memory import SemanticStore
from agentic_commons.memory.types import (
    MemoryConflict,
    MemoryFact,
    MemoryQuery,
    MemoryType,
    Provenance,
    ProvenanceKind,
    WriteResult,
)
from agentic_commons.observability.types import utc_now
from agentic_commons.persistence import PlatformStore

# Which store a category belongs in is a semantic decision (E13); these
# are the two the pipeline can infer structurally, and everything else
# falls through to semantic memory.
_EVENT_CATEGORIES = frozenset({"event", "interaction", "observation"})
_PROCEDURE_CATEGORIES = frozenset({"how_to", "recipe", "compiled_prompt"})


@runtime_checkable
class WriteAuthorizer(Protocol):
    """Decides whether an actor may write a fact of a given category.

    Injected as a Protocol so the pipeline can consult the real RBAC
    authorizer without `memory` depending on `security` directly.
    """

    def may_write(self, actor: str, category: str) -> bool: ...


@instrument("memory")
class MemoryWritePipelineImpl:
    """Chain of Responsibility over the seven write stages."""

    def __init__(
        self,
        semantic: SemanticStore | None = None,
        episodic: EpisodicStore | None = None,
        procedural: ProceduralStore | None = None,
        authorizer: WriteAuthorizer | None = None,
        platform_store: PlatformStore | None = None,
    ) -> None:
        settings = get_settings().memory.write_pipeline
        self._semantic = semantic or SemanticStore(platform_store)
        self._episodic = episodic or EpisodicStore()
        self._procedural = procedural or ProceduralStore(platform_store)
        self._authorizer = authorizer
        self._enabled = settings.enabled
        self._auto_commit = settings.auto_commit_threshold
        self._discard_below = settings.discard_threshold
        self._authority_categories = set(settings.authority_changing_categories)
        self._hints = {
            category: [phrase.casefold() for phrase in phrases]
            for category, phrases in settings.classification_hints.items()
        }
        self._interrogatives = [word.casefold() for word in settings.interrogative_markers]
        self._curator_queue: list[WriteResult] = []
        self._logger = get_logger("memory")

    def is_assertion(self, raw_input: str) -> bool:
        """Whether this text claims something, rather than asks something.

        A question is not a candidate memory. Without this the curator
        queue fills with "where is my order?" — every message becomes a
        fact awaiting adjudication, and a queue that contains everything
        is a queue nobody reads, which is how the one claim that mattered
        gets approved along with the noise.
        """
        text = raw_input.strip().casefold()
        if not text:
            return False
        if text.endswith("?"):
            return False
        return not any(text.startswith(word) for word in self._interrogatives)

    @logged("memory")
    def process(
        self,
        raw_input: str,
        source: str,
        provenance_kind: ProvenanceKind = "self_reported",
        session_id: str = "",
    ) -> WriteResult:
        """Run a candidate memory through every stage."""
        if not self._enabled:
            return WriteResult(route="discard", written=False, reason="write pipeline disabled")
        if not self.is_assertion(raw_input):
            return WriteResult(
                route="discard", written=False, reason="a question states no fact to remember"
            )

        key, value = self.extract(raw_input)
        category = self.classify(raw_input)
        provenance = self.attach_provenance(source, provenance_kind)
        confidence = self.score_confidence(provenance, category)
        candidate = MemoryFact(
            key=key,
            value=value,
            category=category,
            confidence=confidence,
            provenance=provenance,
            metadata={"session_id": session_id} if session_id else {},
        )

        refusal = self.authorize(candidate, source)
        if refusal is not None:
            return self._queue(refusal)

        conflict = self.detect_conflict(candidate)
        if conflict is not None and conflict.resolution == "curator_queue":
            return self._queue(
                WriteResult(
                    route="curator_queue",
                    written=False,
                    fact=candidate,
                    reason=conflict.reason,
                    conflicts_with=[conflict.existing.key],
                    stage="conflict",
                )
            )
        if conflict is not None and conflict.resolution == "supersede":
            self._semantic.supersede(conflict.existing.key, candidate.key)

        return self.route(candidate)

    def extract(self, raw_input: str) -> tuple[str, str]:
        """Stage 1 — pull a (key, value) claim out of raw text.

        Deliberately literal. An LLM extractor belongs behind the same
        cascade every other subsystem uses, and is injected when the
        cheap path cannot cope — not run by default on every utterance.
        """
        text = raw_input.strip()
        if ":" in text:
            key, _, value = text.partition(":")
            return key.strip().casefold().replace(" ", "_"), value.strip()
        return "_".join(text.casefold().split()[:6]), text

    def classify(self, raw_input: str) -> str:
        """Stage 2 — assign a category from the configured phrase hints."""
        lowered = raw_input.casefold()
        for category, phrases in self._hints.items():
            if any(phrase in lowered for phrase in phrases):
                return category
        return "general"

    def attach_provenance(self, source: str, kind: ProvenanceKind) -> Provenance:
        """Stage 3 — record who said it and how they knew."""
        return Provenance(source=source, kind=kind, ingestion_time=utc_now())

    def detect_conflict(self, candidate: MemoryFact) -> MemoryConflict | None:
        """Stage 4 — check the candidate against what is already believed.

        Last-write-wins is only legitimate when the new fact has at least
        the authority of the old one. A self-reported claim overwriting a
        verified system-of-record fact is not a temporal update; it is the
        poisoning case, and it goes to a human.
        """
        existing = self._semantic.get(candidate.key)
        if not existing:
            return None
        prior = existing[0]
        if prior.value == candidate.value:
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
            reason=(
                f"'{candidate.provenance.kind}' claim contradicts a '{prior.provenance.kind}' fact"
            ),
        )

    def score_confidence(self, provenance: Provenance, category: str) -> float:
        """Stage 5 — how much this claim is believed, before any routing.

        Confidence comes from provenance, never from the speaker's tone.
        "I'm certain my limit is $2,000" gets no more weight than the same
        sentence without the adverb.
        """
        base = get_settings().memory.provenance_weights.get(provenance.kind, 0.0)
        if category in self._authority_categories:
            # An authority claim is the one place extra caution is free.
            return base * 0.5
        return base

    def authorize(self, candidate: MemoryFact, actor: str) -> WriteResult | None:
        """Stage 6 — refuse writes the actor has no standing to make.

        Returns a refusal, or None to continue. The authority-changing
        check is unconditional: no confidence score and no injected
        authorizer can permit it from a conversational assertion.
        """
        if candidate.category in self._authority_categories:
            if candidate.provenance.kind != "verified_system_of_record":
                self._logger.warning(
                    "Refusing authority-changing write from unverified source",
                    actor=actor,
                    category=candidate.category,
                    key=candidate.key,
                )
                return WriteResult(
                    route="refused",
                    written=False,
                    fact=candidate,
                    reason=(
                        "authority-changing claims are never written from "
                        "conversational assertion; no role grants itself authority"
                    ),
                    stage="authorization",
                )
        if self._authorizer is not None and not self._authorizer.may_write(
            actor, candidate.category
        ):
            return WriteResult(
                route="refused",
                written=False,
                fact=candidate,
                reason=f"actor '{actor}' may not write category '{candidate.category}'",
                stage="authorization",
            )
        return None

    def route(self, candidate: MemoryFact) -> WriteResult:
        """Stage 7 — auto-commit, queue for a curator, or discard."""
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
        store = self._store_for(_store_type(candidate.category))
        result = store.write(candidate, candidate.provenance)
        return result.model_copy(update={"stage": "route"})

    def curator_queue(self) -> list[WriteResult]:
        """Writes awaiting human adjudication — flagged, never silently dropped."""
        return list(self._curator_queue)

    def approve(self, index: int, curator: str) -> WriteResult:
        """A curator confirms a queued write, committing it as verified."""
        pending = self._curator_queue.pop(index)
        if pending.fact is None:
            return WriteResult(route="discard", written=False, reason="nothing to approve")
        confirmed = Provenance(
            source=curator,
            kind="verified_system_of_record",
            confirmed=True,
            confirmed_at=utc_now(),
        )
        fact = pending.fact.model_copy(update={"confidence": 1.0, "provenance": confirmed})
        store = self._store_for(_store_type(fact.category))
        return store.write(fact, confirmed)

    def _queue(self, result: WriteResult) -> WriteResult:
        """Hold a write for a human, and keep the record either way."""
        self._curator_queue.append(result)
        return result

    def _store_for(
        self, memory_type: MemoryType
    ) -> SemanticStore | EpisodicStore | ProceduralStore:
        """The store a memory type belongs to."""
        if memory_type == "episodic":
            return self._episodic
        if memory_type == "procedural":
            return self._procedural
        return self._semantic

    def recent(self, limit: int = 10) -> list[MemoryFact]:
        """Most recently committed semantic facts — an inspection helper."""
        return self._semantic.search(MemoryQuery(limit=limit))


def _store_type(category: str) -> MemoryType:
    """Which CoALA store a category belongs in."""
    if category in _EVENT_CATEGORIES:
        return "episodic"
    if category in _PROCEDURE_CATEGORIES:
        return "procedural"
    return "semantic"


def _authority_rank(kind: ProvenanceKind) -> int:
    """Ordering over provenance kinds, strongest first."""
    return {"verified_system_of_record": 3, "agent_inference": 2, "self_reported": 1}[kind]
