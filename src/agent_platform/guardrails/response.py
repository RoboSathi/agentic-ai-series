"""End-to-end response guardrails.

Structural validation and factual grounding are separate controls:
Pydantic constrains the shape; the hallucination detector constrains the
claims.  A failed factual check is repaired, and what still fails is
removed: the reply goes out without its refuted sentences (trimmed), or,
when none survive, the configured withheld reply goes instead. Whether a
person then reviews it is the agent's decision, not this module's.
"""

import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field, computed_field

from agent_platform.config import get_settings
from agent_platform.context.types import Chunk
from agent_platform.guardrails.guards import PydanticValidator
from agent_platform.hallucination_check.checker import (
    DetectionLadder,
    GroundingReasoningDiagnoser,
    Resampler,
)
from agent_platform.hallucination_check.claims import sentence_pieces
from agent_platform.hallucination_check.confidence import (
    LOGPROB_TIER,
    LOW_CONFIDENCE_DEGENERATE,
    LOW_CONFIDENCE_SPECIFIC,
    ConfidenceCheck,
    gate,
    report_confidence,
)
from agent_platform.hallucination_check.direct import (
    NOT_EXECUTED_MARKER,
    RECORD_STATE_MISMATCH,
    amounts,
    numbers,
    record_ids,
)
from agent_platform.hallucination_check.evidence import TOOL_RESULT_KIND
from agent_platform.hallucination_check.ledger import FACT_MISMATCH, FACT_TIER, facts_in
from agent_platform.hallucination_check.levels import CheckLevel, grounding_level
from agent_platform.hallucination_check.readings import check_rule, model_label, readings
from agent_platform.hallucination_check.resolver import (
    CLAIM_UNSUPPORTED,
    CONFIRM_TIER,
    DIRECT_TIER,
    RESTATEMENT_MODEL,
)
from agent_platform.hallucination_check.types import DetectionVerdict, GenerationAssessment
from agent_platform.logging import get_logger
from agent_platform.observability import checks
from agent_platform.observability.units import report_grounding

__all__ = [
    "NOT_EXECUTED_MARKER",
    "CustomerResponse",
    "ResponseGuardrail",
    "ResponseGuardrailResult",
]


class CustomerResponse(BaseModel):
    """Provider-neutral structured envelope for a customer-facing reply."""

    response: str = Field(min_length=1, max_length=4000)


class ResponseGuardrailResult(BaseModel):
    """Auditable result of structural + factual response protection."""

    response: str
    # The check's levels, level 1 (token confidence) first: they lead the card and the log.
    levels: list[CheckLevel] = Field(default_factory=list)
    schema_valid: bool = True
    grounded: bool = True
    repaired: bool = False
    escalated: bool = False
    repair_attempts: int = 0
    claims: int = 0
    supported: int = 0
    unsupported: int = 0
    # Claims no evidence spoke to either way. Reported, never blocking.
    undetermined: int = 0
    hallucination_rate: float = 0.0
    nollm_resolved_ratio: float = 1.0
    failure_kinds: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    unsupported_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    tier_counts: dict[str, int] = Field(default_factory=dict)
    reason_code: str = "GUARDRAIL_PASS"
    # none = fully grounded; partial = some claims are supported and some are not;
    # complete = no customer-facing claims were supported.
    hallucination_scope: str = "none"
    # What the customer received when a claim was refuted: the reply less the
    # sentences that carried one (trimmed), or the withheld reply (withheld).
    withheld: bool = False
    removed_sentences: list[str] = Field(default_factory=list)
    total_sentences: int = 0
    # Every claim, supported or not: its verdict, what decided it, and the
    # top-k evidence retrieval surfaced with the NLI result for each item.
    claim_checks: list[dict[str, Any]] = Field(default_factory=list)
    # The deterministic stage: each order number and amount, and where found.
    provenance_checks: list[str] = Field(default_factory=list)
    evidence_searched: str = ""
    # Sentences the record's own statement replaced, because a claim in them was wrong.
    corrected_sentences: list[str] = Field(default_factory=list)
    # Kept sentences holding a claim nothing verified, as sent (marked when the agent marks them).
    unverified_sentences: list[str] = Field(default_factory=list)
    # Level 1 in full: the model, and each sentence with every scored token's span and
    # probability, so a card can colour the words and a log keeps all of it.
    confidence: dict[str, Any] = Field(default_factory=dict)
    # The order the claims were checked in (NLI, then the second opinion), atop the card.
    rule: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def outcome(self) -> str:
        """What happened to the reply, in one line an operator can read.

        `grounded`, `escalated`, `repaired`, `reason_code` and
        `hallucination_scope` each say part of this; the card says it once.
        """
        words = get_settings().guardrails.outcomes
        if self.escalated:
            # Why, in the words a card shows (`guardrails.block_reasons`, by finding code).
            why = get_settings().guardrails.block_reasons.get(self.reason_code, self.reason_code)
            if self.withheld:
                return words.withheld.format(why=why)
            return words.trimmed.format(
                removed=len(self.removed_sentences), total=self.total_sentences, why=why
            )
        if self.repaired:
            return words.repaired.format(attempts=self.repair_attempts)
        return words.passed

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tally(self) -> str:
        """`2 supported · 1 unverified of 3 claims`, flagging a lexical fallback."""
        names = [_VERDICT_NAMES.get(c["verdict"], c["verdict"]) for c in self.claim_checks]
        if not names:
            return "no checkable claims"
        counts = " · ".join(f"{names.count(name)} {name}" for name in dict.fromkeys(names))
        plural = "" if len(names) == 1 else "s"
        lexical = any(str(c.get("detector") or "").startswith(_LEXICAL) for c in self.claim_checks)
        note = " · lexical fallback (NLI model unavailable)" if lexical else ""
        return f"{counts} of {len(names)} claim{plural}{note}"


class ResponseGuardrail:
    """Fail-closed response gate for tool-backed customer answers."""

    def __init__(
        self,
        grounding: DetectionLadder | None,
        validator: PydanticValidator | None = None,
        enforce_grounding: bool | None = None,
        max_repair_attempts: int | None = None,
        confidence: ConfidenceCheck | None = None,
    ) -> None:
        """Structural validation and factual grounding, wired independently.

        `grounding` is the entire switch for the factual half: given a
        detector this gate verifies claims and refuses unsupported ones;
        given None it validates shape and nothing else.

        Schema validation deliberately does not depend on it. The two
        used to share a single flag computed as `guardrails OR
        hallucination_check`, which meant switching either off took
        half of the other with it — a malformed envelope reaching the
        customer because grounding was disabled, or claims going
        unverified because schema checking was. They are different
        guarantees and they now have different switches.
        """
        self._grounding = grounding
        self._validator = validator or PydanticValidator()
        self._diagnoser = GroundingReasoningDiagnoser()
        settings = get_settings().hallucination_check
        self._enforce_grounding = (
            settings.enforce_grounding if enforce_grounding is None else enforce_grounding
        )
        self._max_repairs = (
            settings.max_repair_attempts if max_repair_attempts is None else max_repair_attempts
        )
        self._repair_direct = settings.repair_direct_refutations
        self._marker = settings.undetermined_marker
        # Level 1: the model's confidence in its own tokens, scored before grounding (level 2).
        # Withheld with `hallucination_check.confidence`.
        check = confidence or ConfidenceCheck()
        self._confidence = check if check.enabled else None
        self._logger = get_logger("guardrails")

    def validate_schema(self, raw: str) -> tuple[str | None, dict[str, Any]]:
        """Validate a structured response envelope."""
        result = self._validator.validate(raw, CustomerResponse)
        fields: dict[str, Any] = {
            "schema_valid": result.valid,
            "validation_mode": result.mode,
            "validation_attempts": result.attempts,
            "repaired": result.repaired,
            "errors": result.errors,
        }
        if not result.valid or result.value is None:
            # NIM may ignore guided JSON. Do not silently claim that the
            # schema constrained decoding; preserve the plain-text candidate
            # and let factual grounding remain the blocking control.
            if raw.strip():
                fields["schema_valid"] = False
                fields["schema_fallback"] = "plain_text_provider_output"
                return raw.strip(), fields
            return None, fields
        return str(result.value["response"]), fields

    async def enforce(
        self,
        response: str,
        evidence: list[Chunk],
        revise: Callable[[str, list[str]], Awaitable[str]] | None = None,
        resample: Resampler | None = None,
    ) -> ResponseGuardrailResult:
        """Check, bounded-repair, then fail closed if unsupported claims remain.

        `resample` is handed to the detection ladder, which uses it only
        where the entailment tier could not decide. Passing it does not
        mean it will be called.
        """
        if self._grounding is None or not self._enforce_grounding:
            return ResponseGuardrailResult(response=response)
        started = time.perf_counter()
        scored = self._confidence.assess(response) if self._confidence is not None else None
        first = [report_confidence(scored)] if scored is not None else []
        result = await self._enforce(response, evidence, revise, resample)
        result = result.model_copy(
            update={
                "levels": [*first, _entailment_level(result)],
                "confidence": scored.model_dump() if scored is not None else {},
                "rule": check_rule(),
            }
        )
        self._record(result, self._grounding, (time.perf_counter() - started) * 1000)
        checked = result.model_copy(
            update={
                "provenance_checks": _provenance_report(response, evidence),
                "evidence_searched": _tiers_searched(evidence),
            }
        )
        self._logger.info(
            "Response grounding checked",
            levels=[level.model_dump() for level in checked.levels],
            rule=checked.rule,
            reason_code=checked.reason_code,
            evidence_searched=checked.evidence_searched,
            provenance=checked.provenance_checks,
            claims=checked.claim_checks,
            confidence=checked.confidence,
            corrected=checked.corrected_sentences,
            removed=checked.removed_sentences,
            sent=checked.response,
        )
        return checked

    def _doubted(self, candidate: str, last: GenerationAssessment) -> GenerationAssessment:
        """What the model guessed is cut: a degenerate sentence, an unconfirmed specific."""
        if self._confidence is None:
            return last
        return gate(last, self._confidence.assess(candidate))

    @staticmethod
    def _record(
        result: ResponseGuardrailResult, grounding: DetectionLadder, latency_ms: float
    ) -> None:
        """The grounding check, kept: one check row, and the claims on the turn."""
        backend, model = grounding.identity
        checks.record_grounding(backend, model, result.claims, result.unsupported, latency_ms)
        report_grounding(result.claims, result.supported, result.unsupported)

    async def _enforce(
        self,
        response: str,
        evidence: list[Chunk],
        revise: Callable[[str, list[str]], Awaitable[str]] | None,
        resample: Resampler | None,
    ) -> ResponseGuardrailResult:
        """The checks themselves; `enforce` adds the reporting every outcome shares."""
        if self._grounding is None:
            return ResponseGuardrailResult(response=response)

        candidate = response
        repaired = False
        attempts_made = 0
        last = self._doubted(
            candidate, await self._grounding.assess(candidate, evidence, resample=resample)
        )
        for attempt in range(self._max_repairs + 1):
            attempts_made = attempt
            if attempt:
                last = self._doubted(
                    candidate,
                    await self._grounding.assess(candidate, evidence, resample=resample),
                )
            if last.clean:
                return self._passed(candidate, last, repaired, attempt)
            if attempt >= self._max_repairs or revise is None or self._final(last):
                break
            revision = await revise(candidate, last.unsupported)
            if revision.strip() == candidate.strip():
                # The reviser had nothing to remove and said so by
                # returning the draft. Re-scoring identical text produces
                # an identical verdict, so the second assessment is pure
                # cost.
                break
            repaired = True
            candidate = revision
        return self._blocked(candidate, last, repaired, attempts_made, evidence)

    def _final(self, last: GenerationAssessment) -> bool:
        """Whether a rule refuted a claim and this agent does not rewrite past rules."""
        return not self._repair_direct and _decided_directly(last)

    def _passed(
        self, candidate: str, last: GenerationAssessment, repaired: bool, attempt: int
    ) -> ResponseGuardrailResult:
        unverified = _unverified(last.verdicts)
        marked = self._marker and unverified
        return ResponseGuardrailResult(
            response=_surviving_sentences(candidate, last.verdicts, marker=self._marker)
            if marked
            else candidate,
            unverified_sentences=_marked_sentences(candidate, unverified, self._marker),
            repaired=repaired,
            repair_attempts=attempt,
            claims=len(last.verdicts),
            supported=sum(verdict.supported for verdict in last.verdicts),
            undetermined=len(last.undetermined),
            hallucination_rate=last.hallucination_rate,
            nollm_resolved_ratio=last.nollm_resolved_ratio,
            tier_counts=_tier_counts(last),
            reason_code="GUARDRAIL_PASS_REPAIRED" if repaired else "GUARDRAIL_PASS",
            claim_checks=[_claim_check(verdict) for verdict in last.verdicts],
        )

    def _blocked(
        self,
        candidate: str,
        last: GenerationAssessment,
        repaired: bool,
        attempts: int,
        evidence: list[Chunk],
    ) -> ResponseGuardrailResult:
        """Fail closed: keep every sentence no claim refuted, correct what the record can, escalate."""
        refuted = [verdict for verdict in last.verdicts if _refuted(verdict)]
        diagnostics = [self._diagnostic(verdict) for verdict in refuted]
        failures = sorted({str(item["failure_kind"]) for item in diagnostics})
        supported_count = sum(verdict.supported for verdict in last.verdicts)
        scope = "complete" if supported_count == 0 else "partial"
        self._logger.warning(
            "Response guardrail failed closed",
            unsupported=last.unsupported,
            supported=supported_count,
            claims=len(last.verdicts),
            hallucination_scope=scope,
            failure_kinds=failures,
            attempts=attempts,
        )
        corrections = _corrections(refuted, evidence)
        surviving = _surviving_sentences(candidate, last.verdicts, corrections, self._marker)
        sentences = [
            piece.strip() for piece, _ in sentence_pieces(candidate) if _is_sentence(piece)
        ]
        rejected = {verdict.sentence for verdict in refuted}
        unverified = _unverified(last.verdicts) - rejected
        reason_code = next((v.reason_code for v in refuted if v.reason_code), BLOCK_CODE)
        return ResponseGuardrailResult(
            response=surviving or _withheld_reply(reason_code),
            withheld=not surviving,
            # By index, not by text: a kept sentence may carry the unverified marker.
            removed_sentences=[
                piece.strip()
                for index, (piece, _) in enumerate(sentence_pieces(candidate))
                if index in rejected and _is_sentence(piece)
            ],
            unverified_sentences=_marked_sentences(candidate, unverified, self._marker),
            corrected_sentences=list(dict.fromkeys(corrections.values())),
            total_sentences=len(sentences),
            # Not fully grounded when any claim was rejected, even if
            # sentences no claim refuted were kept.
            grounded=False,
            repaired=repaired,
            escalated=True,
            # What was actually attempted, not what was configured.
            repair_attempts=attempts,
            claims=len(last.verdicts),
            supported=supported_count,
            unsupported=len(last.unsupported),
            undetermined=len(last.undetermined),
            hallucination_rate=last.hallucination_rate,
            nollm_resolved_ratio=last.nollm_resolved_ratio,
            failure_kinds=failures,
            unsupported_claims=last.unsupported,
            unsupported_diagnostics=diagnostics,
            tier_counts=_tier_counts(last),
            reason_code=reason_code,
            hallucination_scope=scope,
            claim_checks=[_claim_check(verdict) for verdict in last.verdicts],
        )

    def _diagnostic(self, verdict: DetectionVerdict) -> dict[str, Any]:
        """Why one refuted claim failed: a rule the record broke, or evidence ignored."""
        retrieved = [Chunk(content=text, kind="tool_result") for text in verdict.retrieved_evidence]
        best_overlap, best_evidence = max(
            ((self._diagnoser.overlap(verdict.claim, c.content), c.content) for c in retrieved),
            key=lambda item: item[0],
            default=(0.0, ""),
        )
        return {
            "claim": verdict.claim,
            "failure_kind": verdict.reason_code
            or self._diagnoser.diagnose(verdict.claim, retrieved),
            "tier": _decided_by(verdict),
            "confidence": round(verdict.confidence, 3),
            "best_evidence_overlap": round(best_overlap, 3),
            "best_evidence": best_evidence[:240],
            "rationale": verdict.rationale,
            "detector_model": verdict.detector_model,
            "retrieval_top_k": verdict.retrieval_top_k,
            "retrieval_method": verdict.retrieval_method,
            "retrieved_evidence": verdict.retrieved_evidence,
        }


BLOCK_CODE = "HALLUCINATION_GUARD_BLOCK"


def _entailment_level(result: ResponseGuardrailResult) -> CheckLevel:
    """Level 2 for a reply grounded by NLI entailment: its claims tally, flagged when any failed."""
    name = get_settings().hallucination_check.levels.entailment
    return grounding_level(name, result.tally, flagged=result.escalated)


def _refuted(verdict: DetectionVerdict) -> bool:
    """A claim the evidence, or its absence, counted against — not one nothing spoke to."""
    return not verdict.supported and not verdict.undetermined


# Findings a rewrite from the evidence can fix: nothing grounded the claim, or the model
# guessed it. A rule the record broke is final; these are not.
_REWRITABLE = frozenset({CLAIM_UNSUPPORTED, LOW_CONFIDENCE_DEGENERATE, LOW_CONFIDENCE_SPECIFIC})


def _decided_directly(assessment: GenerationAssessment) -> bool:
    """Whether a rule, not a model, refuted a claim — which a rewrite cannot talk its way past."""
    return any(
        verdict.reason_code and verdict.reason_code not in _REWRITABLE
        for verdict in assessment.verdicts
    )


def _withheld_reply(reason_code: str) -> str:
    """What the customer is sent when nothing survived, in the words for why."""
    settings = get_settings().hallucination_check
    return settings.withheld_replies.get(reason_code, settings.withheld_reply)


def _corrections(refuted: list[DetectionVerdict], evidence: list[Chunk]) -> dict[int, str]:
    """For each sentence a wrong fact was cut from, the record's own sentence for that fact.

    The customer asked something the record answers; a cut sentence followed by
    nothing leaves them with less than the record holds. Only a contradiction is
    corrected — a claim nothing grounds has no fact to put back.
    """
    corrections: dict[int, str] = {}
    for verdict in refuted:
        statement = _correction(verdict, evidence)
        if statement and verdict.sentence not in corrections:
            corrections[verdict.sentence] = statement
    return corrections


def _correction(verdict: DetectionVerdict, evidence: list[Chunk]) -> str | None:
    """The fact a contradicted claim got wrong, as the record states it."""
    facts = get_settings().hallucination_check.facts
    if verdict.reason_code == FACT_MISMATCH and verdict.evidence_checks:
        excerpt = verdict.evidence_checks[0].excerpt
        held = next((fact for fact, chunk in facts_in(evidence) if chunk.content == excerpt), None)
        correctable = (
            held is None
            or facts.attributes.get(held.attribute) is None
            or (facts.attributes[held.attribute].correctable)
        )
        return excerpt if correctable else None
    if verdict.reason_code != RECORD_STATE_MISMATCH:
        return None
    attribute = facts.state_attribute
    named = record_ids(verdict.claim)
    states = [(fact, chunk) for fact, chunk in facts_in(evidence) if fact.attribute == attribute]
    subjects = {fact.subject for fact, _ in states}
    wanted = named or (subjects if len(subjects) == 1 else set())
    return next((chunk.content for fact, chunk in states if fact.subject in wanted), None)


def _basis(verdict: DetectionVerdict) -> str:
    """The one number a card shows: a model's probability. Nothing else is a score."""
    if (
        verdict.undetermined
        or verdict.reason_code == CLAIM_UNSUPPORTED
        or verdict.tier in (DIRECT_TIER, FACT_TIER, LOGPROB_TIER, CONFIRM_TIER)
        or verdict.detector_model == RESTATEMENT_MODEL
    ):
        return ""
    reading = "Entailment" if verdict.supported else "Contradiction"
    return f"{model_label(verdict.detector_model)} {reading} {verdict.confidence:.2f}"


def _method(verdict: DetectionVerdict) -> str:
    """What decided the claim, in a word or two: the record, a rule, NLI, or logprobs."""
    words = get_settings().hallucination_check.methods
    if verdict.tier in (DIRECT_TIER, FACT_TIER):
        return words.record
    if verdict.tier == LOGPROB_TIER:
        return words.logprobs.format(prob=verdict.confidence)
    if verdict.tier == CONFIRM_TIER:
        return words.confirmed.format(
            model=model_label(verdict.detector_model), prob=verdict.confidence
        )
    if verdict.detector_model == RESTATEMENT_MODEL:
        # A tool result restating the claim is the source saying it, not a rule.
        from_tool = any(c.decisive and c.tier == TOOL_RESULT_KIND for c in verdict.evidence_checks)
        return words.restated_source if from_tool else words.restated
    if verdict.undetermined or verdict.reason_code == CLAIM_UNSUPPORTED:
        # When a model read evidence, its readings say why; "no evidence" would be untrue.
        read = any(check.model for check in verdict.evidence_checks)
        return "" if read else words.nothing
    return words.nli.format(model=model_label(verdict.detector_model), score=verdict.confidence)


def _status(verdict: DetectionVerdict) -> str:
    """The claim's verdict as one word: supported, undetermined, or why it was cut."""
    if verdict.supported:
        return "supported"
    if verdict.undetermined:
        return "undetermined"
    if verdict.reason_code == CLAIM_UNSUPPORTED:
        return "unsupported"
    if verdict.reason_code in (LOW_CONFIDENCE_DEGENERATE, LOW_CONFIDENCE_SPECIFIC):
        return "low_confidence"
    return "refuted"


def _decided_by(verdict: DetectionVerdict) -> str:
    return "deterministic_provenance" if verdict.tier == DIRECT_TIER else verdict.tier


# A claim's verdict in the words a card shows. "undetermined" reads as
# "unverified": no evidence spoke to it either way, and it never blocks.
_VERDICT_NAMES = {
    "supported": "supported",
    "undetermined": "unverified",
    "unsupported": "unsupported",
    "low_confidence": "low confidence",
    "refuted": "contradicted",
}
# The detector name prefix of the word-overlap fallback, which is not NLI.
_LEXICAL = "lexical"

# Evidence tiers, in the words a card shows.
_TIER_NAMES = {
    "tool_result": "tool result",
    "policy": "policy",
    "system": "system prompt",
    "history": "customer message",
    "retrieved": "knowledge",
    "memory": "memory",
}


def _claim_check(verdict: DetectionVerdict) -> dict[str, Any]:
    """One claim's verdict and the evidence trail behind it, for the card and the log."""
    return {
        "claim": verdict.claim,
        "claim_class": verdict.claim_class,
        "verdict": _status(verdict),
        "decided_by": _decided_by(verdict),
        "method": _method(verdict),
        "sentence": verdict.sentence,
        "basis": _basis(verdict),
        "detector": verdict.detector_model,
        "threshold": verdict.threshold,
        "confidence": round(verdict.confidence, 3),
        "rationale": verdict.rationale,
        **({"reason_code": verdict.reason_code} if verdict.reason_code else {}),
        "retrieval": f"top-{verdict.retrieval_top_k} by {verdict.retrieval_method or 'similarity'}",
        "evidence": [
            {
                "rank": check.rank,
                "similarity": check.similarity,
                "tier": _TIER_NAMES.get(check.tier, check.tier),
                "excerpt": check.excerpt,
                "nli_label": check.nli_label,
                "nli_score": check.nli_score,
                "decisive": check.decisive,
                "source": check.source,
                "model": model_label(check.model) if check.model else "",
            }
            for check in verdict.evidence_checks
        ],
        "scores": _scores(verdict),
        # What decided it, step by step: NLI, then (only if NLI did not support it) the
        # second opinion — each with its answer, what that decided, and the line it read.
        "readings": [reading.model_dump() for reading in readings(verdict)],
    }


def _scores(verdict: DetectionVerdict) -> dict[str, str]:
    """Each model's reading of the claim, in one line: its deciding answer, else its surest.

    `{"NLI DeBERTa": "neutral 1.00", "Laya": "supported 0.46"}` — which model said what, so
    a claim confirmed by the second opinion still shows what NLI made of it.
    """
    best: dict[str, Any] = {}
    for check in verdict.evidence_checks:
        if not check.model:
            continue
        held = best.get(check.model)
        if held is None or (check.decisive, check.nli_score) > (held.decisive, held.nli_score):
            best[check.model] = check
    return {
        model_label(model): f"{_NLI_WORDS.get(c.nli_label, c.nli_label)} {c.nli_score:.2f}"
        for model, c in best.items()
    }


# The NLI model's third label, as a reader should meet it.
_NLI_WORDS = {"unknown": "neutral"}


def _provenance_report(response: str, evidence: list[Chunk]) -> list[str]:
    """Each record id and amount the reply states, and which tier (if any) has it."""
    label = get_settings().hallucination_check.record_identifier_label
    orders = [
        _found(f"{label} {record_id}", [c for c in evidence if record_id in numbers(c.content)])
        for record_id in sorted(record_ids(response))
    ]
    stated = [
        _found(str(amount), [c for c in evidence if amount in amounts(c.content)])
        for amount in sorted(amounts(response))
    ]
    return orders + stated


def _found(label: str, holders: list[Chunk]) -> str:
    """`✓ $12.00 — tool result`, or `✗ $75.00 — not in any evidence`."""
    tiers = list(dict.fromkeys(_TIER_NAMES.get(chunk.kind, chunk.kind) for chunk in holders))
    return f"✓ {label} — {', '.join(tiers)}" if tiers else f"✗ {label} — not in any evidence"


def _tiers_searched(evidence: list[Chunk]) -> str:
    """What was looked in, e.g. `3 tool result · 1 policy · 4 knowledge`."""
    counts: dict[str, int] = {}
    for chunk in evidence:
        name = _TIER_NAMES.get(chunk.kind, chunk.kind)
        counts[name] = counts.get(name, 0) + 1
    return " · ".join(f"{count} {name}" for name, count in counts.items()) or "no evidence"


def _tier_counts(assessment: GenerationAssessment) -> dict[str, int]:
    counts: dict[str, int] = {}
    for verdict in assessment.verdicts:
        counts[verdict.tier] = counts.get(verdict.tier, 0) + 1
    return counts


def _is_sentence(piece: str) -> bool:
    """Whether a piece of a reply says anything: a bare list marker ("1.") does not."""
    return any(char.isalpha() for char in piece)


def _surviving_sentences(
    response: str,
    verdicts: list[DetectionVerdict],
    corrections: dict[int, str] | None = None,
    marker: str = "",
) -> str:
    """The reply's own sentences that carry no refuted claim, verbatim.

    Claims are finer than sentences, so gluing the supported *claims*
    back together returned fragments. A sentence either stands as the
    model wrote it or goes; nothing is reassembled. Only a *refuted*
    claim removes its sentence — found by the index the claim came from,
    because a carved rule is restated and its words are not in the reply.
    One nothing could verify is an abstention, and a courteous closing is
    no reason to lose the answer beside it — though with a `marker` it is
    marked as unverified. A bare list marker ("1.") is not a sentence.
    """
    rejected = {verdict.sentence for verdict in verdicts if _refuted(verdict)}
    unverified = _unverified(verdicts) if marker else set()
    fixes = corrections or {}
    kept = [
        _ended(fixes[index], separator)
        if index in rejected
        else (
            _with_marker(sentence.strip(), marker) if index in unverified else sentence.strip(),
            separator,
        )
        for index, (sentence, separator) in enumerate(sentence_pieces(response))
        if (index not in rejected or index in fixes) and _is_sentence(sentence)
    ]
    # Each kept sentence keeps the break that followed it, so a list stays a list.
    return "".join(sentence + separator for sentence, separator in kept).strip()


# A sentence's closing punctuation (and any markdown emphasis after it), kept after a marker.
_CLOSING = re.compile(r"[.!?…:;*_]*$")


def _unverified(verdicts: list[DetectionVerdict]) -> set[int]:
    """The sentences holding a verifiable claim that nothing verified, by index."""
    return {
        verdict.sentence
        for verdict in verdicts
        if verdict.undetermined and verdict.claim_class != "conversational"
    }


def _with_marker(sentence: str, marker: str) -> str:
    """The sentence with the marker before its closing punctuation ("… free (unverified).")."""
    if not marker:
        return sentence
    closing = _CLOSING.search(sentence)
    cut = closing.start() if closing else len(sentence)
    return f"{sentence[:cut]}{marker}{sentence[cut:]}"


def _marked_sentences(response: str, indices: set[int], marker: str) -> list[str]:
    """The unverified sentences of a reply, as it shows them."""
    return [
        _with_marker(piece.strip(), marker)
        for index, (piece, _) in enumerate(sentence_pieces(response))
        if index in indices and _is_sentence(piece)
    ]


def _ended(sentence: str, separator: str) -> tuple[str, str]:
    """A sentence put back from the record ends like one, and is spaced from the next."""
    if sentence and sentence[-1] not in ".!?;:":
        sentence += "."
    return sentence, separator or " "
