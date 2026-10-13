"""The HALLUCINATION CHECK card's format, the one every agent's check is reported in.

Format only, never a decision. Each agent runs its own check (NLI over a reply's
evidence, figures against query rows) and decides what to cut; this module says
how the result is written down, so every agent's card and log line read the same:

    result      what was sent (PASSED / TRIMMED / BLOCKED …), coloured by its first word
    level 1     token confidence: each sentence's mean, its weak tokens marked
    level 2     the agent's own grounding, as a summary line
    check       the order the claims were checked in
    claims      one line per claim: its verdict, and what decided it

The claim tally is the card's title; the card's colour is what the reader received
(`grounding_status`). The payload is a `GroundingCard`; an agent's extras (a draft
check, a review request) ride along as extra fields.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.config import get_settings
from agent_platform.hallucination_check.levels import CheckLevel
from agent_platform.observability import checks
from agent_platform.observability.types import StatusKind
from agent_platform.observability.units import report_grounding


class ClaimCheck(BaseModel):
    """One claim on the card: its verdict word, what decided it, and the steps that did."""

    model_config = ConfigDict(extra="allow")

    claim: str
    # supported, undetermined, unsupported, low_confidence, refuted (`card.verdict_names`).
    verdict: str
    method: str = ""
    rationale: str = ""
    # The sentence of the reply the claim came from, and why a rule cut it ("" when none did).
    sentence: int = -1
    reason_code: str = ""
    # `readings.Reading` dumps: each step that read the claim, in the order it ran.
    readings: list[dict[str, Any]] = Field(default_factory=list)


class GroundingCard(BaseModel):
    """The HALLUCINATION CHECK payload: result, levels (level 1 first), rule, claims."""

    model_config = ConfigDict(extra="allow")

    outcome: str
    tally: str
    levels: list[CheckLevel] = Field(default_factory=list)
    # Level 1 in full (`ConfidenceAssessment` dump): the card colours each sentence by it.
    confidence: dict[str, Any] = Field(default_factory=dict)
    rule: str = ""
    claim_checks: list[ClaimCheck] = Field(default_factory=list)
    corrected_sentences: list[str] = Field(default_factory=list)
    removed_sentences: list[str] = Field(default_factory=list)
    evidence_searched: str = ""

    def fields(self) -> dict[str, Any]:
        """The card's fields, as the event carries them."""
        return self.model_dump()


def claim_tally(claim_checks: list[dict[str, Any]] | list[ClaimCheck]) -> str:
    """`2 supported · 1 unverified of 3 claims`, flagging a lexical fallback."""
    words = get_settings().hallucination_check.card
    rows = [c.model_dump() if isinstance(c, ClaimCheck) else c for c in claim_checks]
    names = [words.verdict_names.get(row["verdict"], row["verdict"]) for row in rows]
    if not names:
        return words.no_claims
    counts = " · ".join(f"{names.count(name)} {name}" for name in dict.fromkeys(names))
    plural = "" if len(names) == 1 else words.plural
    lexical = any(str(row.get("detector") or "").startswith(words.lexical) for row in rows)
    note = words.lexical_note if lexical else ""
    return words.tally.format(counts=counts, total=len(names), plural=plural, note=note)


def outcome_line(
    *,
    escalated: bool,
    withheld: bool,
    removed: int,
    total: int,
    reason_code: str,
    repaired: bool = False,
    attempts: int = 0,
) -> str:
    """What happened to the reply, in the one line an operator reads (`guardrails.outcomes`)."""
    guardrails = get_settings().guardrails
    words = guardrails.outcomes
    if escalated:
        why = guardrails.block_reasons.get(reason_code, reason_code)
        if withheld:
            return words.withheld.format(why=why)
        return words.trimmed.format(removed=removed, total=total, why=why)
    if repaired:
        return words.repaired.format(attempts=attempts)
    return words.passed


def log_grounding(logger: Any, card: GroundingCard) -> None:
    """The check's one log line, in full (`guardrails.log`)."""
    logger.info("Response grounding checked", **card.fields())


def grounding_status(*, withheld: bool, escalated: bool) -> StatusKind:
    """The card's colour, by what was sent: nothing (blocked), less (warning), all (completed)."""
    if withheld:
        return "blocked"
    return "warning" if escalated else "completed"


def record_grounding_check(
    backend: str, model: str, claims: int, supported: int, unsupported: int, latency_ms: float
) -> None:
    """The check, kept: one check row, and the claims on the turn."""
    checks.record_grounding(backend, model, claims, unsupported, latency_ms)
    report_grounding(claims, supported, unsupported)


__all__ = [
    "ClaimCheck",
    "GroundingCard",
    "claim_tally",
    "grounding_status",
    "log_grounding",
    "outcome_line",
    "record_grounding_check",
]
