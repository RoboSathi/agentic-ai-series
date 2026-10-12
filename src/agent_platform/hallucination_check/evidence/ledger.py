"""An append-only evidence ledger, scoped to one run.

Nothing is ever edited or removed: a candidate is either admitted (it
passed the verbatim rule) or recorded as a rejection, and both lists only
grow. The ledger is working state — the caller decides when a run is
over and discards it — so it lives in memory and hands out immutable
snapshots to anyone who needs to reason about it after the fact.
"""

from collections.abc import Callable

from agent_platform.hallucination_check.evidence.types import (
    EvidenceCandidate,
    EvidenceEntry,
    LedgerSnapshot,
    SpanRejection,
)
from agent_platform.hallucination_check.evidence.validator import CandidateValidator
from agent_platform.logging import get_logger

RejectionObserver = Callable[[SpanRejection], None]


class EvidenceLedger:
    """Admits verbatim-checked entries; records every refusal."""

    def __init__(
        self,
        validator: CandidateValidator,
        id_prefix: str,
        on_rejected: RejectionObserver | None = None,
    ) -> None:
        self._validator = validator
        self._prefix = id_prefix
        self._on_rejected = on_rejected
        self._entries: list[EvidenceEntry] = []
        self._rejections: list[SpanRejection] = []
        self._logger = get_logger("ledger")

    def append(self, candidate: EvidenceCandidate) -> EvidenceEntry | SpanRejection:
        """Admit the candidate, or record and return why it was refused."""
        rejection = self._validator.check(candidate)
        if rejection is not None:
            return self._reject(rejection)
        if self._duplicate(candidate):
            return next(entry for entry in self._entries if _same(entry, candidate))
        entry = EvidenceEntry(
            **candidate.model_dump(), entry_id=f"{self._prefix}{len(self._entries) + 1}"
        )
        self._entries.append(entry)
        self._logger.debug("Evidence admitted", entry=entry.entry_id, source=entry.source_url)
        return entry

    def restore(self, snapshot: LedgerSnapshot) -> None:
        """Continue from a checkpointed snapshot (a resumed run)."""
        self._entries = list(snapshot.entries)
        self._rejections = list(snapshot.rejections)

    def snapshot(self) -> LedgerSnapshot:
        return LedgerSnapshot(entries=tuple(self._entries), rejections=tuple(self._rejections))

    @property
    def entries(self) -> list[EvidenceEntry]:
        return list(self._entries)

    @property
    def rejections(self) -> list[SpanRejection]:
        return list(self._rejections)

    def discard(self) -> None:
        """Drop everything; the run that owned this ledger is over."""
        self._entries.clear()
        self._rejections.clear()

    def _reject(self, rejection: SpanRejection) -> SpanRejection:
        self._rejections.append(rejection)
        self._logger.info(
            "Evidence rejected",
            reason_code=rejection.reason_code,
            source=rejection.candidate.source_url,
        )
        if self._on_rejected is not None:
            self._on_rejected(rejection)
        return rejection

    def _duplicate(self, candidate: EvidenceCandidate) -> bool:
        return any(_same(entry, candidate) for entry in self._entries)


def _same(entry: EvidenceEntry, candidate: EvidenceCandidate) -> bool:
    """One span of one page is one piece of evidence, whichever question found it.

    Counting it twice would let a single source look like corroboration.
    """
    return (
        entry.claim == candidate.claim
        and entry.span == candidate.span
        and entry.source_url == candidate.source_url
    )
