"""The direct check: what a record claim can be settled on without a model.

Some questions are decided by reading a field. The record either states a
settled refund or it does not; an amount is either in it or it is not; a
call was refused or it ran. Entailment scoring adds nothing there, and
where it was asked instead it is what blocked true sentences: a state
record contradicted "has been submitted", and a refusal line read as
neutral. So a claim about this case is checked here first — against the
ids and amounts the record holds, the refund state machine
(`states.StateModel`), and the record of a refused call — and only what
this cannot place goes on to the tiers.
"""

import re
from typing import NamedTuple

from agent_platform.config import get_settings
from agent_platform.context.types import Chunk
from agent_platform.hallucination_protection.claims import money_pattern, specifics
from agent_platform.hallucination_protection.states import StateModel
from agent_platform.primitives.money import Money, money_or_none

# How a refused or failed tool call is written into evidence. It grounds a
# statement that the call was refused, never one that its effect happened.
NOT_EXECUTED_MARKER = "NOT EXECUTED"

NO_RECORD = "BUSINESS_CLAIM_WITHOUT_TOOL_EVIDENCE"
ORDER_MISMATCH = "TOOL_EVIDENCE_ORDER_MISMATCH"
AMOUNT_MISMATCH = "TOOL_EVIDENCE_AMOUNT_MISMATCH"
REFUND_STATE_MISMATCH = "TOOL_EVIDENCE_REFUND_STATE_MISMATCH"
NOT_EXECUTED = "TOOL_EVIDENCE_NOT_EXECUTED"
DETAIL_UNSOURCED = "CLAIM_DETAIL_UNSOURCED"


class DirectFinding(NamedTuple):
    """Why a claim was refuted without asking a model, and the line of the record it read."""

    reason_code: str
    rationale: str
    chunk: Chunk | None = None


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
    return {amount for value in money_pattern().findall(text) if (amount := money_or_none(value))}


class DirectChecker:
    """Refutes a record claim the record itself contradicts, or cannot have grounded."""

    def __init__(self) -> None:
        settings = get_settings().hallucination_protection
        self._refusal = re.compile(settings.refusal_pattern, re.IGNORECASE)
        self._states = StateModel()

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
        return self._unsourced(claim, text) or self._state_conflict(claim, record, text)

    def state_support(self, claim: str, record: list[Chunk]) -> tuple[str, Chunk] | None:
        """What the record shows when its refund status makes the claim true.

        "Submitted", "returned" and "refunded" are facts about a status, not
        sentences to be entailed, and a model that will not equate them is
        not evidence that the refund did not happen. Every id and amount the
        claim names was already found in the record by `refutation`; what
        remains is whether the status agrees. The rationale says what was
        read, because the other words in the sentence were not checked.
        """
        finding = self._states.judge(claim, record)
        if finding is None or not finding.holds:
            return None
        shown = ", ".join(str(amount) for amount in sorted(amounts(claim)))
        return finding.rationale + (
            f" and {shown} is a recorded amount" if shown else ""
        ), finding.chunk

    def refusal_support(self, claim: str, record: list[Chunk]) -> tuple[str, Chunk] | None:
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
                return "the tool call for this order was not executed", chunk
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

    def _state_conflict(self, claim: str, record: list[Chunk], text: str) -> DirectFinding | None:
        """A claim about where the refund stands, when the record says otherwise.

        Telling a customer their money is on the way when it is not is the
        most expensive thing this agent can say, and it is decidable by
        reading one field. A record that states no status is left alone
        unless the only refund evidence is a call that was refused:
        silence is not a refutation, but a refusal is a denial.
        """
        finding = self._states.judge(claim, record)
        if finding is not None:
            return (
                None
                if finding.holds
                else DirectFinding(REFUND_STATE_MISMATCH, finding.rationale, finding.chunk)
            )
        if NOT_EXECUTED_MARKER in text and self._states.asserts_settled(claim):
            return DirectFinding(
                NOT_EXECUTED,
                "The reply says a refund happened, but the only refund evidence is a call that "
                "was not executed.",
            )
        return None
