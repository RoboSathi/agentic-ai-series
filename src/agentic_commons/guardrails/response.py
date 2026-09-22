"""End-to-end response guardrails.

Structural validation and factual grounding are separate controls:
Pydantic constrains the shape; the hallucination detector constrains the
claims.  A failed factual check is repaired once and then fails closed into
HITL rather than sending an unsupported customer-facing assertion.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from agentic_commons.config import get_settings
from agentic_commons.context.types import Chunk
from agentic_commons.guardrails.guards import PydanticValidator
from agentic_commons.hallucination_protection.checker import DetectionLadder
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
    tier_counts: dict[str, int] = Field(default_factory=dict)
    reason_code: str = "GUARDRAIL_PASS"


class ResponseGuardrail:
    """Fail-closed response gate for tool-backed customer answers."""

    def __init__(
        self,
        grounding: DetectionLadder | None,
        validator: PydanticValidator | None = None,
    ) -> None:
        self._grounding = grounding
        self._validator = validator or PydanticValidator()
        settings = get_settings()
        self._enabled = settings.guardrails.enabled or settings.hallucination_protection.enabled
        self._enforce_grounding = settings.hallucination_protection.enforce_grounding
        self._max_repairs = settings.guardrails.max_repair_attempts
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
        if (
            not self._enabled
            or not self._enforce_grounding
            or self._grounding is None
            or not evidence
        ):
            return ResponseGuardrailResult(response=response)

        candidate = response
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
        failures = sorted(
            {
                self._grounding.diagnose(claim, evidence)
                for claim in last.unsupported
            }
        )
        self._logger.warning(
            "Response guardrail failed closed",
            unsupported=last.unsupported,
            failure_kinds=failures,
            attempts=self._max_repairs,
        )
        return ResponseGuardrailResult(
            response=(
                "I couldn't verify all of the details in that response. "
                "I'll have a colleague review this for you."
            ),
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
            tier_counts=_tier_counts(last),
            reason_code="HALLUCINATION_GUARD_BLOCK",
        )


def _tier_counts(assessment: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for verdict in assessment.verdicts:
        counts[verdict.tier] = counts.get(verdict.tier, 0) + 1
    return counts
