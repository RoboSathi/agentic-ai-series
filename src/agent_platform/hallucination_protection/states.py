"""The refund state machine, read the way an NLI model cannot.

`Order 10665 refund status: pending_review.` is a state record, not a
sentence. Handed to a cross-encoder it "contradicted" `has been submitted`
at 0.99, because "pending" reads as "not yet". Every status paraphrase bug
("returned" for "refunded", "received" for "submitted") has that shape: a
model that reasons about sentences was asked to reason about a state
machine.

The record has an explicit one — a refund is `none`, then `pending_review`,
then `approved`, then `completed`, or it is `denied` after it was asked
for — so a claim about where the refund stands is judged against it, in
both directions, and the record line is never sent to NLI.

The vocabulary is configuration (`hallucination_protection.record_states`):
what the statuses are, and which words mean which. The platform owns the
reading; the agent owns the words.
"""

import re
from typing import NamedTuple

from agent_platform.config import get_settings
from agent_platform.config.settings import StatePhraseSettings
from agent_platform.context.types import Chunk

# Polarity is a property of the clause: "No refund has been processed"
# carries the completed wording exactly while denying it.
_NEGATION = re.compile(r"\b(?:no|not|never|n't|cannot|nothing|neither|nor|unable)\b", re.IGNORECASE)
_OBSERVATION_SEQUENCE = "observation_sequence"


class StateFinding(NamedTuple):
    """What the record says about a claim: whether it holds, why, and the line it read."""

    holds: bool
    rationale: str
    chunk: Chunk


class StateModel:
    """Judges a claim about where a record stands against its recorded status (configured)."""

    def __init__(self) -> None:
        settings = get_settings().hallucination_protection
        states = settings.record_states
        self._line = re.compile(settings.record_state_pattern, re.IGNORECASE)
        self._ladder = [status.casefold() for status in states.ladder]
        self._branches = {k.casefold(): v.casefold() for k, v in states.branches.items()}
        self._settled_from = states.settled_from.casefold()
        self._observed = {
            status.casefold(): [marker.casefold() for marker in markers]
            for status, markers in states.observed_as.items()
        }
        self._identifier = re.compile(settings.record_identifier_pattern, re.IGNORECASE)
        self._context = re.compile(states.context_pattern, re.IGNORECASE)
        # "Will be processed once approved" is a promise, not a status: what follows the
        # first promise marker is judged by whether the refund can still get there.
        self._promise = re.compile(settings.promissory_pattern, re.IGNORECASE)
        # What the record's status is the status *of*, as a rationale names it.
        self._subject = states.subject
        self._phrases = [
            (re.compile(phrase.pattern, re.IGNORECASE), phrase) for phrase in states.phrases
        ]

    @property
    def subject(self) -> str:
        """What the recorded status is the status of ("refund")."""
        return self._subject

    def judge(self, claim: str, record: list[Chunk]) -> StateFinding | None:
        """Whether the record's current status makes this claim true, or None to defer.

        Deferred when the claim is not about a refund, names no state the
        vocabulary knows, or the record states no status: silence is not a
        refutation, and a claim the model cannot place is left to the tiers.
        What the claim *promises* ("will be processed once approved") holds
        while the refund can still reach it; only what it *states* must
        already be true.
        """
        if not self._context.search(claim):
            return None
        current = self.current(self._about(claim, record))
        asserted, promised = self._split_promise(claim)
        stated, pledged = self._matched(asserted), self._matched(promised)
        negated = bool(_NEGATION.search(claim))
        if negated:
            # A denied promise ("won't be refunded") is not a status the record can show.
            pledged = []
        if current is None or not (stated or pledged):
            return None
        status, chunk = current
        truths = [self._holds(phrase, status) for _, phrase in stated]
        holds = (not any(truths) if negated else all(truths)) and (
            not pledged or self._reachable(status)
        )
        return StateFinding(holds, self._rationale(status, stated, pledged, holds, negated), chunk)

    def _split_promise(self, claim: str) -> tuple[str, str]:
        """The claim as stated (before its first promise marker), and as promised (after it)."""
        marker = self._promise.search(claim)
        if marker is None:
            return claim, ""
        return claim[: marker.start()], claim[marker.end() :]

    def _reachable(self, status: str) -> bool:
        """Whether a refund at this status can still move up the ladder: a branch has stopped."""
        return status in self._ladder

    def _about(self, claim: str, record: list[Chunk]) -> list[Chunk]:
        """The record about the order the claim names: another order's status is not its status."""
        named = {d for found in self._identifier.findall(claim) for d in re.findall(r"\d+", found)}
        if not named:
            return record
        return [chunk for chunk in record if named & set(re.findall(r"\d{3,}", chunk.content))]

    def asserts_settled(self, claim: str) -> bool:
        """Whether the claim says money moved, or was authorised, and does not deny it."""
        if not self._context.search(claim) or _NEGATION.search(claim):
            return False
        return any(
            phrase.reaches in self._ladder
            and self._ladder.index(phrase.reaches) >= self._ladder.index(self._settled_from)
            for _, phrase in self._matched(claim)
        )

    def current(self, record: list[Chunk]) -> tuple[str, Chunk] | None:
        """The status the latest observation that states one states, with the line it is on.

        The latest, not the union of every status seen: a read that said
        `none` and a referral that then said `pending_review` leave the
        refund at `pending_review`, and both are in the turn's evidence.
        """
        ordered = sorted(enumerate(record), key=lambda item: (self._sequence(item[1]), item[0]))
        latest: tuple[str, Chunk] | None = None
        for _, chunk in ordered:
            stated = self._stated(chunk.content)
            if stated is not None and self._known(stated):
                latest = (stated, chunk)
        return latest

    def _stated(self, text: str) -> str | None:
        """The status one piece of evidence states: on the status line, else in its own words."""
        lines = [str(status).casefold() for status in self._line.findall(text)]
        if lines:
            return lines[-1]
        lowered = text.casefold()
        for status, markers in self._observed.items():
            if any(marker in lowered for marker in markers):
                return status
        return None

    def _matched(self, claim: str) -> list[tuple[re.Pattern[str], StatePhraseSettings]]:
        """The phrases the claim uses; an event ("was queued") outranks a state ("under review")."""
        found = [(pattern, phrase) for pattern, phrase in self._phrases if pattern.search(claim)]
        events = [item for item in found if not item[1].exact]
        return events or found

    def _holds(self, phrase: StatePhraseSettings, status: str) -> bool:
        """Whether one phrase is true of a refund at this status."""
        if phrase.exact or phrase.reaches not in self._ladder:
            return status == phrase.reaches.casefold()
        return self._progress(status) >= self._ladder.index(phrase.reaches.casefold())

    def _progress(self, status: str) -> int:
        """How far up the ladder a refund at this status has been."""
        if status in self._ladder:
            return self._ladder.index(status)
        return self._ladder.index(self._branches[status])

    def _known(self, status: str) -> bool:
        return status in self._ladder or status in self._branches

    @staticmethod
    def _sequence(chunk: Chunk) -> int:
        sequence = chunk.metadata.get(_OBSERVATION_SEQUENCE)
        return sequence if isinstance(sequence, int) else -1

    def _rationale(
        self,
        status: str,
        stated: list[tuple[re.Pattern[str], StatePhraseSettings]],
        pledged: list[tuple[re.Pattern[str], StatePhraseSettings]],
        holds: bool,
        negated: bool,
    ) -> str:
        parts = []
        if stated:
            said = ", ".join(sorted({phrase.reaches for _, phrase in stated}))
            parts.append(f"a claim of {said}{' (denied)' if negated else ''}")
        if pledged:
            parts.append(f"a promise of {', '.join(sorted({p.reaches for _, p in pledged}))}")
        verb = "holds" if holds else "does not hold"
        return f"{self._subject} status is {status}, so {' and '.join(parts)} {verb}"
