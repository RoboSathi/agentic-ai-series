"""End-to-end response guardrails.

Structural validation and factual grounding are separate controls:
Pydantic constrains the shape; the hallucination detector constrains the
claims.  A failed factual check is repaired once and then fails closed into
HITL rather than sending an unsupported customer-facing assertion.
"""

from collections.abc import Awaitable, Callable
import re
from typing import Any

from pydantic import BaseModel, Field

from agentic_commons.config import get_settings
from agentic_commons.context.types import Chunk
from agentic_commons.guardrails.guards import PydanticValidator
from agentic_commons.hallucination_protection.checker import (
    DetectionLadder,
    GroundingReasoningDiagnoser,
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
    ) -> None:
        self._grounding = grounding
        self._validator = validator or PydanticValidator()
        self._diagnoser = GroundingReasoningDiagnoser()
        settings = get_settings()
        self._enabled = settings.guardrails.enabled or settings.hallucination_protection.enabled
        self._enforce_grounding = settings.hallucination_protection.enforce_grounding
        self._max_repairs = settings.hallucination_protection.max_repair_attempts
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
    ) -> ResponseGuardrailResult:
        """Check, bounded-repair, then fail closed if unsupported claims remain."""
        if not self._enabled or not self._enforce_grounding:
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

        # Deterministic provenance is independent of the NLI detector. If
        # the semantic detector is unavailable, do not skip the deterministic
        # authority boundary; only skip the semantic tier after it passes.
        if self._grounding is None:
            return ResponseGuardrailResult(response=response)

        if not evidence:
            return ResponseGuardrailResult(response=response)

        repaired = False
        last = None
        for attempt in range(self._max_repairs + 1):
            last = await self._grounding.assess(candidate, evidence)
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
            repaired = True
            candidate = await revise(candidate, last.unsupported)

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
            repair_attempts=self._max_repairs,
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

    return None

def _tier_counts(assessment: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for verdict in assessment.verdicts:
        counts[verdict.tier] = counts.get(verdict.tier, 0) + 1
    return counts
