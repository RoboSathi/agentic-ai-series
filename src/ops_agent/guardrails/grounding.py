"""The ops agent's hallucination check: token confidence, then every figure against the rows.

Level 1 is the platform's (`ConfidenceCheck`): each sentence of the answer scored by the
tokens that wrote it. Level 2 is this agent's (`FigureCheck`): each figure looked for in
the rows the queries returned. Then the platform's gate decides as it does for the support
agent: a sentence the model guessed (degenerate), or one holding a figure written at low
confidence that no row confirms, is cut. A figure a row holds is confirmed, whatever its
probability; one no row holds and the model wrote confidently is kept and named as
unverified in the reply.

What is reported is the platform's card (`hallucination_check/card.py`), so this card
reads like the support agent's: result, level 1, level 2, the rule, then each figure.
"""

import time

from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.guardrails.response import surviving_sentences
from agent_platform.hallucination_check.card import (
    ClaimCheck,
    GroundingCard,
    claim_tally,
    grounding_status,
    outcome_line,
    record_grounding_check,
)
from agent_platform.hallucination_check.claims import sentence_pieces
from agent_platform.hallucination_check.confidence import (
    LOW_CONFIDENCE_DEGENERATE,
    LOW_CONFIDENCE_SPECIFIC,
    ConfidenceAssessment,
    ConfidenceCheck,
    gate,
    report_confidence,
)
from agent_platform.hallucination_check.levels import CheckLevel, grounding_level
from agent_platform.hallucination_check.readings import EvidenceRef, Reading, Tone
from agent_platform.hallucination_check.types import DetectionVerdict, GenerationAssessment
from agent_platform.observability import checks
from agent_platform.observability.types import StatusKind
from ops_agent.config.settings import Cards, OpsAgentSettings
from ops_agent.domain import QueryRecord
from ops_agent.guardrails.figures import FigureCheck, FigureFinding

# The tier a figure found in a query's rows is decided by.
QUERY_TIER = "query_result"
# A claim's verdict word (`hallucination_check.card.verdict_names`).
_SUPPORTED, _UNVERIFIED, _LOW = "supported", "undetermined", "low_confidence"
_GUESSED = frozenset({LOW_CONFIDENCE_DEGENERATE, LOW_CONFIDENCE_SPECIFIC})
_PASS: Tone = "pass"
_WARN: Tone = "warn"
_FAIL: Tone = "fail"


class OpsGroundingResult(BaseModel):
    """What was sent, the figures it still holds unverified, and the card that says so."""

    response: str
    unverified: list[str]
    card: GroundingCard
    status: StatusKind
    escalated: bool


class OpsGrounding:
    """Level 1, level 2, the gate; the result in the platform's card format."""

    def __init__(
        self,
        figures: FigureCheck,
        settings: OpsAgentSettings,
        confidence: ConfidenceCheck | None = None,
    ) -> None:
        self._figures = figures
        self._settings = settings
        self._confidence = confidence if confidence is not None and confidence.enabled else None

    def check(self, answer: str, queries: list[QueryRecord]) -> OpsGroundingResult:
        """Score, look up, gate; then what the reader receives."""
        started = time.perf_counter()
        scored = self._confidence.assess(answer) if self._confidence is not None else None
        findings = self._figures.findings(answer, queries)
        verdicts = _gated([_verdict(finding) for finding in findings], scored)
        cut = {verdict.sentence for verdict in verdicts if _refuted(verdict)}
        response = surviving_sentences(answer, verdicts) if cut else answer
        withheld = bool(cut) and not response
        claims = self._claims(verdicts, findings, queries)
        supported = sum(verdict.supported for verdict in verdicts)
        record_grounding_check(
            checks.RULE, "", len(verdicts), supported, len(verdicts) - supported,
            (time.perf_counter() - started) * 1000,
        )  # fmt: skip
        unverified = [v.claim for v in verdicts if v.undetermined and v.sentence not in cut]
        card = GroundingCard(
            outcome=self._outcome(answer, verdicts, cut, withheld, unverified),
            tally=claim_tally(claims),
            levels=[*_first(scored), self._second(findings)],
            confidence=scored.model_dump() if scored is not None else {},
            rule=self._settings.cards.rule,
            claim_checks=claims,
            removed_sentences=_removed(answer, cut),
            evidence_searched=self._settings.cards.queries_searched.format(count=len(queries)),
        )
        return OpsGroundingResult(
            response=self._settings.words.withheld if withheld else response,
            unverified=unverified,
            card=card,
            status=_status(withheld, bool(cut), bool(unverified)),
            escalated=bool(cut),
        )

    def _outcome(
        self,
        answer: str,
        verdicts: list[DetectionVerdict],
        cut: set[int],
        withheld: bool,
        unverified: list[str],
    ) -> str:
        """The result line: what was sent, as every agent's card says it."""
        if not cut and unverified:
            return self._settings.cards.unverified.format(count=len(unverified))
        reason = next((v.reason_code for v in verdicts if _refuted(v)), "")
        sentences = sum(1 for piece, _ in sentence_pieces(answer) if piece.strip())
        return outcome_line(
            escalated=bool(cut),
            withheld=withheld,
            removed=len(cut),
            total=sentences,
            reason_code=reason,
        )

    def _second(self, findings: list[FigureFinding]) -> CheckLevel:
        """Level 2: how many figures a query's rows hold, flagged when any is not."""
        cards = self._settings.cards
        if not findings:
            return grounding_level(cards.figure_level, cards.no_figures, flagged=False)
        found = sum(finding.supported for finding in findings)
        tally = cards.figure_tally.format(supported=found, figures=len(findings))
        return grounding_level(cards.figure_level, tally, flagged=found < len(findings))

    def _claims(
        self,
        verdicts: list[DetectionVerdict],
        findings: list[FigureFinding],
        queries: list[QueryRecord],
    ) -> list[ClaimCheck]:
        """Each figure, then any sentence the gate cut that held none (a guessed sentence)."""
        figures = [
            self._claim(verdict, [_query_step(finding, queries, self._settings.cards)])
            for verdict, finding in zip(verdicts, findings, strict=False)
        ]
        return figures + [self._claim(verdict, []) for verdict in verdicts[len(findings) :]]

    @staticmethod
    def _claim(verdict: DetectionVerdict, steps: list[Reading]) -> ClaimCheck:
        """One claim on the card: its verdict, then each step that decided it."""
        if verdict.reason_code in _GUESSED:
            steps.append(_logprob_step(verdict))
        word = _SUPPORTED if verdict.supported else _UNVERIFIED if verdict.undetermined else _LOW
        return ClaimCheck(
            claim=verdict.claim,
            verdict=word,
            rationale=verdict.rationale,
            sentence=verdict.sentence,
            reason_code=verdict.reason_code,
            readings=[step.model_dump() for step in steps],
        )


def _verdict(finding: FigureFinding) -> DetectionVerdict:
    """A figure as a claim: supported when a row holds it, otherwise unverified (never refuted)."""
    return DetectionVerdict(
        claim=finding.figure,
        supported=finding.supported,
        undetermined=not finding.supported,
        confidence=1.0,
        tier=QUERY_TIER,
        claim_class="record",
        sentence=finding.sentence,
    )


def _gated(
    verdicts: list[DetectionVerdict], scored: ConfidenceAssessment | None
) -> list[DetectionVerdict]:
    """The platform's gate over the figures: a guessed sentence or an unconfirmed specific is cut.

    The gate may add a verdict of its own for a degenerate sentence with no figure in it;
    the card shows figures, so only the figures' verdicts are kept (its sentence is still cut).
    """
    if scored is None or not scored.available:
        return verdicts
    assessment = GenerationAssessment(
        verdicts=verdicts, undetermined=[v.claim for v in verdicts if v.undetermined]
    )
    gated = gate(assessment, scored).verdicts
    return gated[: len(verdicts)] + [v for v in gated[len(verdicts) :] if _refuted(v)]


def _query_step(finding: FigureFinding, queries: list[QueryRecord], cards: Cards) -> Reading:
    """Step 1: the figure looked for in the query results, and the row that holds it."""
    words = get_settings().hallucination_check.readings
    if finding.query is None or finding.row is None:
        return Reading(
            step=1,
            check=cards.query_check,
            decision=cards.not_found,
            tone=_WARN,
            lead=words.leads["not_found"],
        )
    query = queries[finding.query - 1]
    where = cards.query_where.format(query=finding.query, row=finding.row + 1)
    return Reading(
        step=1,
        check=cards.query_check,
        decision=words.supported,
        tone=_PASS,
        lead=words.leads["supported"],
        evidence=[EvidenceRef(where=where, excerpt=_row_line(query, finding.row))],
    )


def _logprob_step(verdict: DetectionVerdict) -> Reading:
    """Step 2: the gate cut it — guessed, and no row confirms it."""
    hc = get_settings().hallucination_check
    return Reading(
        step=2,
        check=hc.methods.logprobs.format(prob=verdict.confidence),
        decision=hc.readings.removed,
        tone=_FAIL,
        lead=hc.readings.leads["removed"],
        evidence=[EvidenceRef(where=hc.readings.why, excerpt=verdict.rationale)],
    )


def _row_line(query: QueryRecord, row: int) -> str:
    """One row as `column=value` pairs, the way the reader finds it in the table."""
    cells = query.rows[row]
    return ", ".join(f"{name}={value}" for name, value in zip(query.columns, cells, strict=False))


def _first(scored: ConfidenceAssessment | None) -> list[CheckLevel]:
    """Level 1 (none when token confidence is withheld)."""
    return [report_confidence(scored)] if scored is not None else []


def _removed(answer: str, cut: set[int]) -> list[str]:
    return [
        piece.strip() for index, (piece, _) in enumerate(sentence_pieces(answer)) if index in cut
    ]


def _refuted(verdict: DetectionVerdict) -> bool:
    return not verdict.supported and not verdict.undetermined


def _status(withheld: bool, cut: bool, unverified: bool) -> StatusKind:
    """Coloured by what was sent; a figure left unverified is a warning too."""
    status = grounding_status(withheld=withheld, escalated=cut)
    return "warning" if status == "completed" and unverified else status


__all__ = ["QUERY_TIER", "OpsGrounding", "OpsGroundingResult"]
