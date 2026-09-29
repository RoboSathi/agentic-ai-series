"""The direct check: what a record claim can be settled on without a model.

Some questions are decided by reading a field. The tool result either
states a settled refund or it does not; an amount is either in the record
or it is not. Entailment scoring adds nothing there, and where it was
asked instead it is also what blocked true sentences for want of token
overlap. So a claim about this case is checked here first, and it can only
be *refuted* here: support comes from the tiers, which can say what a
record means, not merely whether it contains a number.
"""

import re
from typing import NamedTuple

from agentic_commons.config import get_settings
from agentic_commons.context.types import Chunk
from agentic_commons.hallucination_protection.claims import MONEY, specifics
from agentic_commons.money import Money, money_or_none

# How a refused or failed tool call is written into evidence. It grounds a
# statement that the call was refused, never one that its effect happened.
NOT_EXECUTED_MARKER = "NOT EXECUTED"

NO_RECORD = "BUSINESS_CLAIM_WITHOUT_TOOL_EVIDENCE"
ORDER_MISMATCH = "TOOL_EVIDENCE_ORDER_MISMATCH"
AMOUNT_MISMATCH = "TOOL_EVIDENCE_AMOUNT_MISMATCH"
REFUND_STATE_MISMATCH = "TOOL_EVIDENCE_REFUND_STATE_MISMATCH"
NOT_EXECUTED = "TOOL_EVIDENCE_NOT_EXECUTED"
DETAIL_UNSOURCED = "CLAIM_DETAIL_UNSOURCED"

# Polarity is a property of the clause: "No refund has been processed"
# carries the settled-claim wording exactly while denying it.
_NEGATION = re.compile(r"\b(?:no|not|never|n't|cannot|nothing|neither|nor|unable)\b", re.IGNORECASE)


class DirectFinding(NamedTuple):
    """Why a claim was refuted without asking a model."""

    reason_code: str
    rationale: str


def order_ids(text: str) -> set[str]:
    """The digits of every order reference a text names."""
    pattern = re.compile(
        get_settings().hallucination_protection.record_identifier_pattern, re.IGNORECASE
    )
    return {digits for found in pattern.findall(text) for digits in re.findall(r"\d+", found)}


def numbers(text: str) -> set[str]:
    """Every number of three or more digits a text carries, however it is written.

    The record states an order as `order 10786`, `order_id=10786` or
    `order:10786` depending on which tool wrote it. A claim's order is
    anchored to the word "order"; the record it is checked against is
    read for the number alone, or a refusal line would name no order.
    """
    return set(re.findall(r"\d{3,}", text))


def amounts(text: str) -> set[Money]:
    """Every amount a text states, exactly."""
    return {amount for value in MONEY.findall(text) if (amount := money_or_none(value))}


class DirectChecker:
    """Refutes a record claim the record itself contradicts, or cannot have grounded."""

    def __init__(self) -> None:
        settings = get_settings().hallucination_protection
        self._settled_claim = re.compile(settings.settled_claim_pattern, re.IGNORECASE)
        self._refusal = re.compile(settings.refusal_pattern, re.IGNORECASE)
        self._refund_state = re.compile(settings.refund_state_pattern, re.IGNORECASE)
        self._settled_states = frozenset(state.casefold() for state in settings.settled_states)
        self._pending = [marker.casefold() for marker in settings.pending_markers]

    def refutation(self, claim: str, record: list[Chunk]) -> DirectFinding | None:
        """The first way the record fails to ground this claim, if it does.

        `record` is what this turn's tools and read-backs established.
        A fact about this case with no record at all is unsourced, which
        is not the same as merely unscored.
        """
        if not record:
            return DirectFinding(
                NO_RECORD,
                "The reply states a fact about this case but no tool result exists for this turn.",
            )
        text = "\n".join(chunk.content for chunk in record)
        return self._unsourced(claim, text) or self._refund_conflict(claim, text)

    def settled_support(self, claim: str, record: list[Chunk]) -> str | None:
        """What the record shows when it settles a claim that the refund went through.

        Called only for a claim the model could not decide and nothing
        refuted, so every order and amount it names is already known to be
        in the record. What remains is whether the record says the money
        moved: a settled `refund_status`. "Returned", "processed" and
        "completed" are one fact in three words, and an NLI model that
        will not equate them is not evidence that the refund did not
        happen. The rationale says what was read, because the other words
        in the sentence were not checked.
        """
        if not self._settled_claim.search(claim) or _NEGATION.search(claim):
            return None
        text = "\n".join(chunk.content for chunk in record)
        states = sorted(
            {state.casefold() for state in self._refund_state.findall(text)} & self._settled_states
        )
        if not states:
            return None
        shown = ", ".join(str(amount) for amount in sorted(amounts(claim)))
        return f"the record shows refund status {states[0]}" + (
            f" and {shown} as a recorded amount" if shown else ""
        )

    def refusal_support(self, claim: str, record: list[Chunk]) -> str | None:
        """Why the record shows the agent did not act, when the claim says it did not.

        "I'm unable to process a refund for order 10786" is true exactly
        when the call for that order was refused, and the record of a
        refused call is a NOT EXECUTED entry carrying the order's number.
        NLI reading a natural sentence against that structured line calls
        it neutral, so the record is read directly. A claim naming an
        order the refusal is not about gets nothing from it.
        """
        if not self._refusal.search(claim):
            return None
        named = order_ids(claim)
        for chunk in record:
            if NOT_EXECUTED_MARKER in chunk.content and (
                not named or named & numbers(chunk.content)
            ):
                return "the tool call for this order was not executed"
        return None

    def unsourced_detail(self, claim: str, evidence: list[Chunk]) -> DirectFinding | None:
        """A date, weekday or number a rule claim brings that no evidence contains.

        A rule needs no record, so a missing record proves nothing against
        it. A detail the claim invents is another matter: "your package
        will arrive on Tuesday" against evidence naming no day is a
        fabrication, not an abstention, however the NLI model scored it.
        """
        known = specifics("\n".join(chunk.content for chunk in evidence))
        missing = sorted(specifics(claim) - known)
        if not missing:
            return None
        return DirectFinding(
            DETAIL_UNSOURCED, f"{', '.join(missing)} appear(s) in the claim but in no evidence."
        )

    def _unsourced(self, claim: str, text: str) -> DirectFinding | None:
        """An order number or an amount the record never mentions."""
        missing_orders = sorted(order_ids(claim) - numbers(text))
        if missing_orders:
            return DirectFinding(
                ORDER_MISMATCH,
                f"Order reference(s) {missing_orders} appear in the reply but in no tool result.",
            )
        missing_amounts = [str(a) for a in sorted(amounts(claim) - amounts(text))]
        if missing_amounts:
            return DirectFinding(
                AMOUNT_MISMATCH,
                f"Amount(s) {missing_amounts} appear in the reply but in no tool result.",
            )
        return None

    def _refund_conflict(self, claim: str, text: str) -> DirectFinding | None:
        """A claim that money moved, when the record says it did not.

        Telling a customer their money is on the way when it is not is the
        most expensive thing this agent can say, and it is decidable by
        reading one field. What the record states no status for, and does
        not say is waiting on a person, is left alone: silence is not a
        refutation.
        """
        if not self._settled_claim.search(claim) or _NEGATION.search(claim):
            return None
        states = {state.casefold() for state in self._refund_state.findall(text)}
        if states & self._settled_states:
            return None
        if states:
            return DirectFinding(
                REFUND_STATE_MISMATCH,
                f"The reply says the refund is settled, but the recorded refund status is "
                f"{sorted(states)}.",
            )
        if NOT_EXECUTED_MARKER in text:
            return DirectFinding(
                NOT_EXECUTED,
                "The reply says a refund happened, but the only refund evidence is a call that "
                "was not executed.",
            )
        if any(marker in text.casefold() for marker in self._pending):
            return DirectFinding(
                REFUND_STATE_MISMATCH,
                "The reply says the refund is settled, but the record says it is waiting on a person.",
            )
        return None
