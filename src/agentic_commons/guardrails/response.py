"""End-to-end response guardrails.

Structural validation and factual grounding are separate controls:
Pydantic constrains the shape; the hallucination detector constrains the
claims.  A failed factual check is repaired once and then fails closed into
HITL rather than sending an unsupported customer-facing assertion.
"""

import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from agentic_commons.config import get_settings
from agentic_commons.context.types import Chunk
from agentic_commons.guardrails.guards import PydanticValidator
from agentic_commons.hallucination_protection.checker import (
    DetectionLadder,
    GroundingReasoningDiagnoser,
    Resampler,
)
from agentic_commons.logging import get_logger


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

        # No successful tool result is not the same thing as "nothing to
        # verify". For operational/business claims, absence of authoritative
        # evidence is itself a grounding failure. This closes the 0/0 PASS
        # loophole where an unsupported refund/write claim bypassed the
        # detector simply because the evidence list was empty.
        if not evidence and _requires_authoritative_evidence(response):
            claim = response.strip()
            reason = "BUSINESS_CLAIM_WITHOUT_TOOL_EVIDENCE"
            self._logger.warning(
                "Response guardrail blocked business claim without tool evidence",
                reason_code=reason,
                claim=claim,
            )
            return ResponseGuardrailResult(
                response="I couldn't verify that action from the available system records. Please try again.",
                grounded=False,
                escalated=True,
                claims=1,
                supported=0,
                unsupported=1,
                hallucination_rate=1.0,
                nollm_resolved_ratio=1.0,
                failure_kinds=[reason],
                unsupported_claims=[claim],
                unsupported_diagnostics=[{
                    "claim": claim,
                    "failure_kind": reason,
                    "tier": "deterministic_provenance",
                    "confidence": 1.0,
                    "best_evidence_overlap": 0.0,
                    "best_evidence": "",
                    "rationale": "The response contains an operational business outcome but no successful tool result exists for this turn.",
                    "detector_model": "deterministic",
                    "retrieval_top_k": 0,
                    "retrieval_method": "authoritative_tool_result_only",
                    "retrieved_evidence": [],
                }],
                reason_code=reason,
                hallucination_scope="complete",
            )

        candidate = response
        deterministic_failure = _deterministic_evidence_failure(candidate, evidence)
        if deterministic_failure is not None:
            claim, reason, rationale = deterministic_failure
            self._logger.warning(
                "Response guardrail blocked deterministic evidence mismatch",
                reason_code=reason,
                claim=claim,
            )
            return ResponseGuardrailResult(
                response="I couldn't verify all of the details in that response. Please try again.",
                grounded=False,
                escalated=True,
                claims=1,
                supported=0,
                unsupported=1,
                hallucination_rate=1.0,
                nollm_resolved_ratio=1.0,
                failure_kinds=[reason],
                unsupported_claims=[claim],
                unsupported_diagnostics=[{
                    "claim": claim,
                    "failure_kind": reason,
                    "tier": "deterministic_provenance",
                    "confidence": 1.0,
                    "best_evidence_overlap": 0.0,
                    "best_evidence": "",
                    "rationale": rationale,
                    "detector_model": "deterministic",
                    "retrieval_top_k": 0,
                    "retrieval_method": "authoritative_tool_result_only",
                    "retrieved_evidence": [chunk.content for chunk in evidence],
                }],
                reason_code=reason,
                hallucination_scope="complete",
            )

        if not evidence:
            return ResponseGuardrailResult(response=response)

        repaired = False
        last = None
        attempts_made = 0
        for attempt in range(self._max_repairs + 1):
            attempts_made = attempt
            last = await self._grounding.assess(candidate, evidence, resample=resample)
            if last.clean:
                return ResponseGuardrailResult(
                    response=candidate,
                    repaired=repaired,
                    repair_attempts=attempt,
                    claims=len(last.verdicts),
                    supported=len(last.verdicts),
                    hallucination_rate=last.hallucination_rate,
                    nollm_resolved_ratio=last.nollm_resolved_ratio,
                    tier_counts=_tier_counts(last),
                    reason_code="GUARDRAIL_PASS_REPAIRED" if repaired else "GUARDRAIL_PASS",
                )
            if attempt >= self._max_repairs or revise is None:
                break
            revision = await revise(candidate, last.unsupported)
            if revision.strip() == candidate.strip():
                # The reviser had nothing to remove and said so by
                # returning the draft. Re-scoring identical text produces
                # an identical verdict, so the second assessment is pure
                # cost — one completion and a full retrieval pass to
                # reach a conclusion already in hand.
                break
            repaired = True
            candidate = revision

        assert last is not None
        unsupported_verdicts = [
            verdict for verdict in last.verdicts if not verdict.supported
        ]
        diagnostics = []
        for verdict in unsupported_verdicts:
            retrieved = [
                Chunk(content=text, kind="tool_result")
                for text in verdict.retrieved_evidence
            ]
            ranked_evidence = sorted(
                (
                    (self._diagnoser.overlap(verdict.claim, chunk.content), chunk.content)
                    for chunk in retrieved
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            best_overlap, best_evidence = ranked_evidence[0] if ranked_evidence else (0.0, "")
            failure_kind = self._diagnoser.diagnose(verdict.claim, retrieved)
            diagnostics.append(
                {
                    "claim": verdict.claim,
                    "failure_kind": failure_kind,
                    "tier": verdict.tier,
                    "confidence": round(verdict.confidence, 3),
                    "best_evidence_overlap": round(best_overlap, 3),
                    "best_evidence": best_evidence[:240],
                    "rationale": verdict.rationale,
                    "detector_model": verdict.detector_model,
                    "retrieval_top_k": verdict.retrieval_top_k,
                    "retrieval_method": verdict.retrieval_method,
                    "retrieved_evidence": verdict.retrieved_evidence,
                }
            )
        failures = sorted({item["failure_kind"] for item in diagnostics})
        supported_count = len(last.verdicts) - len(last.unsupported)
        hallucination_scope = (
            "complete"
            if supported_count == 0 and last.unsupported
            else "partial"
            if last.unsupported
            else "none"
        )
        self._logger.warning(
            "Response guardrail failed closed",
            unsupported=last.unsupported,
            supported=supported_count,
            claims=len(last.verdicts),
            hallucination_scope=hallucination_scope,
            failure_kinds=failures,
            attempts=self._max_repairs,
        )
        safe_claims = [
            verdict.claim.strip().rstrip(".")
            for verdict in last.verdicts
            if verdict.supported and verdict.claim.strip()
        ]
        safe_response = (
            ". ".join(safe_claims) + "."
            if safe_claims
            else "I couldn't verify all of the details in that response. Please try again."
        )
        return ResponseGuardrailResult(
            response=safe_response,
            # The customer-facing response is not fully grounded when any
            # claim was rejected, even if we retained one or more safe claims.
            grounded=False,
            repaired=repaired,
            escalated=True,
            # What was actually attempted, not what was configured. The
            # two differ whenever the loop stopped early, and reporting
            # the ceiling made a trace look like work that never happened.
            repair_attempts=attempts_made,
            claims=len(last.verdicts),
            supported=len(last.verdicts) - len(last.unsupported),
            unsupported=len(last.unsupported),
            hallucination_rate=last.hallucination_rate,
            nollm_resolved_ratio=last.nollm_resolved_ratio,
            failure_kinds=failures,
            unsupported_claims=last.unsupported,
            unsupported_diagnostics=diagnostics,
            tier_counts=_tier_counts(last),
            reason_code="HALLUCINATION_GUARD_BLOCK",
            hallucination_scope=hallucination_scope,
        )



_OPERATIONAL_TERMS = re.compile(
    r"\b(?:refunded|queued|queue(?:d|ing)?|approved|"
    r"processed|issued|completed|submitted|denied|rejected|"
    r"cancelled|canceled|delivered|shipped|pending|"
    r"on\s+its\s+way|human\s+review|request\s+created)\b",
    re.IGNORECASE,
)
_ORDER_REFERENCE = re.compile(r"\border\s*#?\s*(\d{3,})\b", re.IGNORECASE)
_MONEY = re.compile(r"(?<!\w)\$?\d+(?:,\d{3})*(?:\.\d{2})(?!\w)")

# A reply claiming money has moved, and the field in the record that
# would have to agree with it. Checked deterministically because it is a
# deterministic question: the tool result either states a settled refund
# or it does not, and no amount of entailment scoring improves on
# reading the field. An NLI tier asked to catch this instead was the
# thing that also blocked true sentences for want of token overlap.
_REFUND_SETTLED_CLAIM = re.compile(
    r"\brefund(?:ed)?\b[^.;]{0,60}?\b(?:has\s+been\s+)?"
    r"(?:completed|processed|issued|approved|sent|paid)\b"
    r"|\b(?:completed|processed|issued|approved|sent|paid)\b[^.;]{0,40}?\brefund",
    re.IGNORECASE,
)
_REFUND_STATE = re.compile(r"refund(?:_|\s+)status\s*[:=]\s*(\w+)", re.IGNORECASE)
# Clause boundaries, and the markers that flip a clause's polarity.
# "No refund has been processed" matches the settled-claim shape word for
# word while asserting the opposite, so polarity is checked per clause
# rather than over the whole reply — one negated clause must not excuse a
# settled claim elsewhere in the same sentence, and vice versa.
_CLAUSE = re.compile(r"[.;]|\bbut\b|\band\b", re.IGNORECASE)
_CLAIM_NEGATIONS = re.compile(
    r"\b(?:no|not|never|n't|cannot|nothing|neither|nor|unable)\b", re.IGNORECASE
)
# The refund_status values that mean money has actually moved, or been
# authorised to. Anything else ("none", "denied") refutes the claim.
_SETTLED_REFUND_STATES = frozenset({"completed", "approved", "processed"})


def _requires_authoritative_evidence(response: str) -> bool:
    """Detect customer-facing operational outcomes that require tool proof."""
    text = response.strip()
    if not text:
        return False
    return bool(_OPERATIONAL_TERMS.search(text))


def _deterministic_evidence_failure(
    response: str,
    evidence: list[Chunk],
) -> tuple[str, str, str] | None:
    """Check identifier/value provenance before spending an NLI call."""
    if not _requires_authoritative_evidence(response):
        return None

    evidence_text = "\n".join(chunk.content for chunk in evidence)
    response_ids = set(_ORDER_REFERENCE.findall(response))
    evidence_ids = set(_ORDER_REFERENCE.findall(evidence_text))
    missing_ids = sorted(response_ids - evidence_ids)
    if missing_ids:
        return (
            response,
            "TOOL_EVIDENCE_ORDER_MISMATCH",
            f"Order reference(s) {missing_ids} appear in the response but not in authoritative tool results.",
        )

    response_amounts = {
        value.replace(",", "")
        for value in _MONEY.findall(response)
    }
    evidence_amounts = {
        value.replace(",", "")
        for value in _MONEY.findall(evidence_text)
    }
    missing_amounts = sorted(response_amounts - evidence_amounts)
    if missing_amounts:
        return (
            response,
            "TOOL_EVIDENCE_AMOUNT_MISMATCH",
            f"Amount(s) {missing_amounts} appear in the response but not in authoritative tool results.",
        )

    refund_conflict = _refund_state_conflict(response, evidence_text)
    if refund_conflict is not None:
        return refund_conflict

    return None


def _asserts_settled_refund(response: str) -> bool:
    """Whether any clause positively claims the refund is settled.

    Per clause, because polarity is a property of the clause and not of
    the reply: "No refund has been processed for order 10979 yet" carries
    the settled-claim wording exactly while denying it, and blocking that
    would silence the truthful answer this guard exists to protect.
    """
    return any(
        _REFUND_SETTLED_CLAIM.search(clause) and not _CLAIM_NEGATIONS.search(clause)
        for clause in _CLAUSE.split(response)
    )


def _refund_state_conflict(response: str, evidence_text: str) -> tuple[str, str, str] | None:
    """Catch a claim that a refund is settled when the record says otherwise.

    Telling a customer their money is on the way when it is not is the
    single most expensive thing this agent can say, and it is decidable
    by reading one field. Anything the evidence does not state a refund
    status for is left alone — silence is not a refutation.
    """
    if not _asserts_settled_refund(response):
        return None
    states = {state.casefold() for state in _REFUND_STATE.findall(evidence_text)}
    if not states or states & _SETTLED_REFUND_STATES:
        return None
    return (
        response,
        "TOOL_EVIDENCE_REFUND_STATE_MISMATCH",
        (
            f"The response says the refund is settled, but the authoritative refund "
            f"status is {sorted(states)}."
        ),
    )

def _tier_counts(assessment: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for verdict in assessment.verdicts:
        counts[verdict.tier] = counts.get(verdict.tier, 0) + 1
    return counts
