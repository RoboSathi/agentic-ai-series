"""End-to-end response guardrails.

Structural validation and factual grounding are separate controls:
Pydantic constrains the shape; the hallucination detector constrains the
claims.  A failed factual check is repaired once and then fails closed into
HITL rather than sending an unsupported customer-facing assertion.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field, computed_field

from agentic_commons.config import get_settings
from agentic_commons.context.types import Chunk
from agentic_commons.guardrails.guards import PydanticValidator
from agentic_commons.hallucination_protection.checker import (
    DetectionLadder,
    GroundingReasoningDiagnoser,
    Resampler,
)
from agentic_commons.hallucination_protection.claims import split_sentences
from agentic_commons.hallucination_protection.direct import (
    AMOUNT_MISMATCH,
    DETAIL_UNSOURCED,
    NO_RECORD,
    NOT_EXECUTED,
    NOT_EXECUTED_MARKER,
    ORDER_MISMATCH,
    REFUND_STATE_MISMATCH,
    amounts,
    order_ids,
)
from agentic_commons.hallucination_protection.resolver import DIRECT_TIER
from agentic_commons.hallucination_protection.types import DetectionVerdict, GenerationAssessment
from agentic_commons.logging import get_logger

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
    # Every claim, supported or not: its verdict, what decided it, and the
    # top-k evidence retrieval surfaced with the NLI result for each item.
    claim_checks: list[dict[str, Any]] = Field(default_factory=list)
    # The deterministic stage: each order number and amount, and where found.
    provenance_checks: list[str] = Field(default_factory=list)
    evidence_searched: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def outcome(self) -> str:
        """What happened to the reply, in one line an operator can read.

        `grounded`, `escalated`, `repaired`, `reason_code` and
        `hallucination_scope` each say part of this; the card says it once.
        """
        if self.escalated:
            why = _BLOCK_REASONS.get(self.reason_code, self.reason_code)
            return f"BLOCKED · {why} · escalated to human review"
        if self.repaired:
            return f"PASSED after {self.repair_attempts} rewrite(s) · rewritten reply sent"
        return "PASSED · reply sent unchanged"

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
    ) -> None:
        """Structural validation and factual grounding, wired independently.

        `grounding` is the entire switch for the factual half: given a
        detector this gate verifies claims and refuses unsupported ones;
        given None it validates shape and nothing else.

        Schema validation deliberately does not depend on it. The two
        used to share a single flag computed as `guardrails OR
        hallucination_protection`, which meant switching either off took
        half of the other with it — a malformed envelope reaching the
        customer because grounding was disabled, or claims going
        unverified because schema checking was. They are different
        guarantees and they now have different switches.
        """
        self._grounding = grounding
        self._validator = validator or PydanticValidator()
        self._diagnoser = GroundingReasoningDiagnoser()
        settings = get_settings().hallucination_protection
        self._enforce_grounding = (
            settings.enforce_grounding if enforce_grounding is None else enforce_grounding
        )
        self._max_repairs = (
            settings.max_repair_attempts if max_repair_attempts is None else max_repair_attempts
        )
        self._logger = get_logger("guardrails")

    def validate_schema(self, raw: str) -> tuple[str | None, dict[str, Any]]:
        """Validate a structured response envelope."""
        result = self._validator.validate(raw, CustomerResponse)
        fields = {
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
        result = await self._enforce(response, evidence, revise, resample)
        checked = result.model_copy(
            update={
                "provenance_checks": _provenance_report(response, evidence),
                "evidence_searched": _tiers_searched(evidence),
            }
        )
        self._logger.info(
            "Response grounding checked",
            reason_code=checked.reason_code,
            evidence_searched=checked.evidence_searched,
            provenance=checked.provenance_checks,
            claims=checked.claim_checks,
        )
        return checked

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
        last = await self._grounding.assess(candidate, evidence, resample=resample)
        for attempt in range(self._max_repairs + 1):
            attempts_made = attempt
            if attempt:
                last = await self._grounding.assess(candidate, evidence, resample=resample)
            if last.clean:
                return self._passed(candidate, last, repaired, attempt)
            if attempt >= self._max_repairs or revise is None or _decided_directly(last):
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
        return self._blocked(candidate, last, repaired, attempts_made)

    def _passed(
        self, candidate: str, last: GenerationAssessment, repaired: bool, attempt: int
    ) -> ResponseGuardrailResult:
        return ResponseGuardrailResult(
            response=candidate,
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
        self, candidate: str, last: GenerationAssessment, repaired: bool, attempts: int
    ) -> ResponseGuardrailResult:
        """Fail closed: keep every sentence no claim refuted, say why, escalate."""
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
        return ResponseGuardrailResult(
            response=_surviving_sentences(candidate, last.verdicts) or _WITHHELD,
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
            reason_code=next((v.reason_code for v in refuted if v.reason_code), BLOCK_CODE),
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
_WITHHELD = "I couldn't verify all of the details in that response. Please try again."


def _refuted(verdict: DetectionVerdict) -> bool:
    """A claim the evidence, or its absence, counted against — not one nothing spoke to."""
    return not verdict.supported and not verdict.undetermined


def _decided_directly(assessment: GenerationAssessment) -> bool:
    """Whether a rule, not a model, refuted a claim — which a rewrite cannot talk its way past."""
    return any(verdict.reason_code for verdict in assessment.verdicts)


def _decided_by(verdict: DetectionVerdict) -> str:
    return "deterministic_provenance" if verdict.tier == DIRECT_TIER else verdict.tier


# A claim's verdict in the words a card shows. "undetermined" reads as
# "unverified": no evidence spoke to it either way, and it never blocks.
_VERDICT_NAMES = {
    "supported": "supported",
    "undetermined": "unverified",
    "refuted": "contradicted",
}
# The detector name prefix of the word-overlap fallback, which is not NLI.
_LEXICAL = "lexical"

# Why a reply was blocked, in the words a card shows.
_BLOCK_REASONS = {
    NO_RECORD: "states an outcome no tool result shows",
    ORDER_MISMATCH: "names an order number found in no evidence",
    AMOUNT_MISMATCH: "states an amount found in no evidence",
    REFUND_STATE_MISMATCH: "says the refund is settled; the record disagrees",
    NOT_EXECUTED: "claims a refund the tool call never executed",
    DETAIL_UNSOURCED: "states a date or number found in no evidence",
    BLOCK_CODE: "a claim is contradicted by the evidence",
}

# Evidence tiers, in the words a card shows.
_TIER_NAMES = {
    "tool_result": "tool result",
    "verification": "verification",
    "policy": "policy",
    "system": "system prompt",
    "history": "customer message",
    "retrieved": "knowledge",
    "memory": "memory",
}


def _claim_check(verdict: DetectionVerdict) -> dict[str, Any]:
    """One claim's verdict and the evidence trail behind it, for the card and the log."""
    status = (
        "supported" if verdict.supported else "undetermined" if verdict.undetermined else "refuted"
    )
    return {
        "claim": verdict.claim,
        "claim_class": verdict.claim_class,
        "verdict": status,
        "decided_by": _decided_by(verdict),
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
            }
            for check in verdict.evidence_checks
        ],
    }


def _provenance_report(response: str, evidence: list[Chunk]) -> list[str]:
    """Each order number and amount the reply states, and which tier (if any) has it."""
    orders = [
        _found(f"order {order_id}", [c for c in evidence if order_id in order_ids(c.content)])
        for order_id in sorted(order_ids(response))
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


def _surviving_sentences(response: str, verdicts: list[DetectionVerdict]) -> str:
    """The reply's own sentences that carry no refuted claim, verbatim.

    Claims are finer than sentences, so gluing the supported *claims*
    back together returned fragments. A sentence either stands as the
    model wrote it or goes; nothing is reassembled. Only a *refuted*
    claim removes its sentence: one nothing could verify is an
    abstention, and a courteous closing is no reason to lose the
    answer beside it. A bare list marker ("1.") is not a sentence.
    """
    rejected = [
        verdict.claim.strip().rstrip(".").casefold()
        for verdict in verdicts
        if _refuted(verdict) and verdict.claim.strip()
    ]
    kept = [
        sentence.strip()
        for sentence in split_sentences(response)
        if any(char.isalpha() for char in sentence)
        and not any(claim in sentence.casefold() for claim in rejected)
    ]
    return " ".join(kept)
