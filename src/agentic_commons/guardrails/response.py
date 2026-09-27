"""End-to-end response guardrails.

Structural validation and factual grounding are separate controls:
Pydantic constrains the shape; the hallucination detector constrains the
claims.  A failed factual check is repaired once and then fails closed into
HITL rather than sending an unsupported customer-facing assertion.
"""

import re
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field, computed_field

from agentic_commons.config import get_settings
from agentic_commons.context.types import Chunk
from agentic_commons.guardrails.guards import PydanticValidator
from agentic_commons.hallucination_protection.checker import (
    COMBINED_TIER,
    COMBINED_TIER_JOINER,
    DetectionLadder,
    GroundingReasoningDiagnoser,
    Resampler,
    split_sentences,
)
from agentic_commons.hallucination_protection.types import DetectionVerdict
from agentic_commons.logging import get_logger
from agentic_commons.money import Money, money_or_none


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

        # No successful tool result is not the same thing as "nothing to
        # verify". For operational/business claims, absence of authoritative
        # evidence is itself a grounding failure. This closes the 0/0 PASS
        # loophole where an unsupported refund/write claim bypassed the
        # detector simply because the evidence list was empty.
        # Only a tool result can show that something *happened*; policy,
        # knowledge and memory can support a statement, never an outcome.
        if not _tool_chunks(evidence) and _requires_authoritative_evidence(response):
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
                unsupported_diagnostics=[
                    {
                        "claim": claim,
                        "failure_kind": reason,
                        "tier": "deterministic_provenance",
                        "confidence": 1.0,
                        "best_evidence_overlap": 0.0,
                        "best_evidence": "",
                        "rationale": "The response contains an operational business outcome but no successful tool result exists for this turn.",
                        "detector_model": "deterministic",
                        "retrieval_top_k": 0,
                        "retrieval_method": _tiers_searched(evidence),
                        "retrieved_evidence": [],
                    }
                ],
                reason_code=reason,
                hallucination_scope="complete",
                claim_checks=[_deterministic_check(claim, reason, "no tool result for this turn")],
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
                unsupported_diagnostics=[
                    {
                        "claim": claim,
                        "failure_kind": reason,
                        "tier": "deterministic_provenance",
                        "confidence": 1.0,
                        "best_evidence_overlap": 0.0,
                        "best_evidence": _closest_excerpt(claim, evidence),
                        "rationale": rationale,
                        "detector_model": "deterministic",
                        "retrieval_top_k": 0,
                        "retrieval_method": _tiers_searched(evidence),
                        "retrieved_evidence": [chunk.content for chunk in evidence],
                    }
                ],
                reason_code=reason,
                hallucination_scope="complete",
                claim_checks=[_deterministic_check(claim, reason, rationale)],
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
                    supported=sum(verdict.supported for verdict in last.verdicts),
                    undetermined=len(last.undetermined),
                    hallucination_rate=last.hallucination_rate,
                    nollm_resolved_ratio=last.nollm_resolved_ratio,
                    tier_counts=_tier_counts(last),
                    reason_code="GUARDRAIL_PASS_REPAIRED" if repaired else "GUARDRAIL_PASS",
                    claim_checks=[_claim_check(verdict) for verdict in last.verdicts],
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
        unsupported_verdicts = [verdict for verdict in last.verdicts if not verdict.supported]
        diagnostics = []
        for verdict in unsupported_verdicts:
            retrieved = [
                Chunk(content=text, kind="tool_result") for text in verdict.retrieved_evidence
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
        failures = sorted({str(item["failure_kind"]) for item in diagnostics})
        supported_count = sum(verdict.supported for verdict in last.verdicts)
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
        safe_response = _surviving_sentences(candidate, last.verdicts) or (
            "I couldn't verify all of the details in that response. Please try again."
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
            supported=supported_count,
            unsupported=len(last.unsupported),
            undetermined=len(last.undetermined),
            hallucination_rate=last.hallucination_rate,
            nollm_resolved_ratio=last.nollm_resolved_ratio,
            failure_kinds=failures,
            unsupported_claims=last.unsupported,
            unsupported_diagnostics=diagnostics,
            tier_counts=_tier_counts(last),
            reason_code="HALLUCINATION_GUARD_BLOCK",
            hallucination_scope=hallucination_scope,
            claim_checks=[_claim_check(verdict) for verdict in last.verdicts],
        )


_OPERATIONAL_TERMS = re.compile(
    r"\b(?:refunded|queued|queue(?:d|ing)?|approved|"
    r"processed|issued|completed|submitted|denied|rejected|"
    r"cancelled|canceled|delivered|shipped|pending|"
    r"on\s+its\s+way|human\s+review|request\s+created)\b",
    re.IGNORECASE,
)
_ORDER_REFERENCE = re.compile(r"\border\s*#?\s*(\d{3,})\b", re.IGNORECASE)
# An amount in prose: "$210", "$1,234.56", or a bare "210.00". Compared as
# integer cents, so "$210" and "$210.00" are the same amount.
_MONEY = re.compile(r"(?<!\w)(?:\$\d+(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:,\d{3})*\.\d{2})(?!\w)")

# A reply claiming money has moved, and the field in the record that
# would have to agree with it. Checked deterministically because it is a
# deterministic question: the tool result either states a settled refund
# or it does not, and no amount of entailment scoring improves on
# reading the field. An NLI tier asked to catch this instead was the
# thing that also blocked true sentences for want of token overlap.
# How a refused or failed tool call is written into evidence. It grounds a
# statement that the call was refused, never one that its effect happened.
NOT_EXECUTED_MARKER = "NOT EXECUTED"

# Within one clause: any character but a clause break, and a decimal point
# is not a clause break — "$146.07" must not split "refund of $146.07 has
# been processed" into two halves that each look innocent.
_IN_CLAUSE = r"(?:[^.;]|\.(?=\d))"

_REFUND_SETTLED_CLAIM = re.compile(
    r"\brefund(?:ed)?\b" + _IN_CLAUSE + r"{0,60}?\b(?:has\s+been\s+)?"
    r"(?:completed|complete|processed|issued|approved|sent|paid|done|"
    r"(?:gone|went)\s+through|on\s+(?:its|the)\s+way)\b"
    # "Sent for review" hands the refund to a person; it has not settled.
    r"(?!\s+(?:for|to)\s+(?:\w+\s+)?(?:review|approval))"
    r"|\b(?:completed|processed|issued|approved|sent|paid)\b" + _IN_CLAUSE + r"{0,40}?\brefund",
    re.IGNORECASE,
)
_REFUND_STATE = re.compile(r"refund(?:_|\s+)status\s*[:=]\s*(\w+)", re.IGNORECASE)
# Clause boundaries, and the markers that flip a clause's polarity.
# "No refund has been processed" matches the settled-claim shape word for
# word while asserting the opposite, so polarity is checked per clause
# rather than over the whole reply — one negated clause must not excuse a
# settled claim elsewhere in the same sentence, and vice versa.
_CLAUSE = re.compile(r"\.(?!\d)|;|\bbut\b|\band\b", re.IGNORECASE)
_CLAIM_NEGATIONS = re.compile(
    r"\b(?:no|not|never|n't|cannot|nothing|neither|nor|unable)\b", re.IGNORECASE
)
# A clause saying what will *not* happen states the agent's intent, not an
# outcome: "your order will not be cancelled" needs no tool to have run.
# Negated past and present stay outcome claims — "your refund has not been
# processed" describes the record, and can be wrong about it.
_NEGATED_FUTURE = re.compile(
    r"\b(?:will\s+not|will\s+no\s+longer|won['’]t|shall\s+not|not\s+going\s+to)\b",
    re.IGNORECASE,
)
# The refund_status values that mean money has actually moved, or been
# authorised to. Anything else ("none", "denied") refutes the claim.
_SETTLED_REFUND_STATES = frozenset({"completed", "approved", "processed"})


def _requires_authoritative_evidence(response: str) -> bool:
    """Whether any clause states an operational outcome, which only a tool result can prove.

    Per clause, like `_asserts_settled_refund`: one clause declining an
    action must not excuse an outcome claimed in the next.
    """
    return any(
        _OPERATIONAL_TERMS.search(clause) and not _NEGATED_FUTURE.search(clause)
        for clause in _CLAUSE.split(response.strip())
    )


def _deterministic_evidence_failure(
    response: str,
    evidence: list[Chunk],
) -> tuple[str, str, str] | None:
    """Check identifier/value provenance before spending an NLI call.

    Ids and amounts may come from any tier the model was given — a tool
    result, the policy, a knowledge passage, a verified memory. Whether a
    refund *happened* may come from tool results alone.

    A settled-refund claim opens the check on its own: "your refund is
    done" names no operational term, and was the one false outcome the
    gate let through.
    """
    if not (_requires_authoritative_evidence(response) or _asserts_settled_refund(response)):
        return None

    evidence_text = "\n".join(chunk.content for chunk in evidence)
    response_ids = set(_ORDER_REFERENCE.findall(response))
    evidence_ids = set(_ORDER_REFERENCE.findall(evidence_text))
    missing_ids = sorted(response_ids - evidence_ids)
    if missing_ids:
        return (
            response,
            "TOOL_EVIDENCE_ORDER_MISMATCH",
            f"Order reference(s) {missing_ids} appear in the response but in no evidence "
            f"({_tiers_searched(evidence)}).",
        )

    response_amounts = _amounts(response)
    evidence_amounts = _amounts(evidence_text)
    missing_amounts = [str(amount) for amount in sorted(response_amounts - evidence_amounts)]
    if missing_amounts:
        return (
            response,
            "TOOL_EVIDENCE_AMOUNT_MISMATCH",
            f"Amount(s) {missing_amounts} appear in the response but in no evidence "
            f"({_tiers_searched(evidence)}).",
        )

    tool_text = "\n".join(chunk.content for chunk in _tool_chunks(evidence))
    refund_conflict = _refund_state_conflict(response, tool_text)
    if refund_conflict is not None:
        return refund_conflict

    return None


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
    "BUSINESS_CLAIM_WITHOUT_TOOL_EVIDENCE": "states an outcome no tool result shows",
    "TOOL_EVIDENCE_ORDER_MISMATCH": "names an order number found in no evidence",
    "TOOL_EVIDENCE_AMOUNT_MISMATCH": "states an amount found in no evidence",
    "TOOL_EVIDENCE_REFUND_STATE_MISMATCH": "says the refund is settled; the record disagrees",
    "TOOL_EVIDENCE_NOT_EXECUTED": "claims a refund the tool call never executed",
    "HALLUCINATION_GUARD_BLOCK": "a claim is contradicted by the evidence",
}

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
    status = (
        "supported" if verdict.supported else "undetermined" if verdict.undetermined else "refuted"
    )
    return {
        "claim": verdict.claim,
        "verdict": status,
        "decided_by": verdict.tier or "nli_entailment",
        "detector": verdict.detector_model,
        "threshold": verdict.threshold,
        "confidence": round(verdict.confidence, 3),
        "rationale": verdict.rationale,
        "retrieval": f"top-{verdict.retrieval_top_k} by {verdict.retrieval_method or 'similarity'}",
        "evidence": [
            {
                "rank": check.rank,
                "similarity": check.similarity,
                "tier": _tier_name(check.tier),
                "excerpt": check.excerpt,
                "nli_label": check.nli_label,
                "nli_score": check.nli_score,
                "decisive": check.decisive,
            }
            for check in verdict.evidence_checks
        ],
    }


def _tier_name(tier: str) -> str:
    """`tool result`, or `combined tool result + tool result` for a joined premise."""
    kind, _, parts = tier.partition(":")
    if kind != COMBINED_TIER or not parts:
        return _TIER_NAMES.get(tier, tier)
    names = [_TIER_NAMES.get(part, part) for part in parts.split(COMBINED_TIER_JOINER)]
    return f"{COMBINED_TIER} {' + '.join(names)}"


def _deterministic_check(claim: str, reason: str, rationale: str) -> dict[str, Any]:
    """A claim settled by the deterministic stage — no retrieval, no NLI."""
    return {
        "claim": claim,
        "verdict": "refuted",
        "decided_by": "deterministic_provenance",
        "detector": "deterministic",
        "threshold": None,
        "confidence": 1.0,
        "rationale": rationale,
        "reason_code": reason,
        "retrieval": "not needed",
        "evidence": [],
    }


def _provenance_report(response: str, evidence: list[Chunk]) -> list[str]:
    """Each order number and amount the reply states, and which tier (if any) has it."""
    orders = [
        _found(
            f"order {order_id}",
            [c for c in evidence if order_id in _ORDER_REFERENCE.findall(c.content)],
        )
        for order_id in dict.fromkeys(_ORDER_REFERENCE.findall(response))
    ]
    amounts = [
        _found(str(amount), [c for c in evidence if amount in _amounts(c.content)])
        for amount in sorted(_amounts(response))
    ]
    return orders + amounts


def _found(label: str, holders: list[Chunk]) -> str:
    """`✓ $12.00 — tool result`, or `✗ $75.00 — not in any evidence`."""
    tiers = list(dict.fromkeys(_TIER_NAMES.get(chunk.kind, chunk.kind) for chunk in holders))
    return f"✓ {label} — {', '.join(tiers)}" if tiers else f"✗ {label} — not in any evidence"


def _tool_chunks(evidence: list[Chunk]) -> list[Chunk]:
    """The only tier that can show an outcome happened: this turn's tool results."""
    return [chunk for chunk in evidence if chunk.kind == "tool_result"]


def _tiers_searched(evidence: list[Chunk]) -> str:
    """What was looked in, e.g. `3 tool result · 1 policy · 4 knowledge`."""
    counts: dict[str, int] = {}
    for chunk in evidence:
        name = _TIER_NAMES.get(chunk.kind, chunk.kind)
        counts[name] = counts.get(name, 0) + 1
    return " · ".join(f"{count} {name}" for name, count in counts.items()) or "no evidence"


def _closest_excerpt(claim: str, evidence: list[Chunk]) -> str:
    """The evidence line sharing most words with the claim — what nearly supported it."""
    words = set(claim.casefold().split())
    lines = [line for chunk in evidence for line in chunk.content.splitlines() if line.strip()]
    if not lines:
        return ""
    best = max(lines, key=lambda line: len(words & set(line.casefold().split())))
    return best[:240]


def _amounts(text: str) -> set[Money]:
    """Every amount a text states, exactly."""
    return {amount for value in _MONEY.findall(text) if (amount := money_or_none(value))}


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
    if states & _SETTLED_REFUND_STATES:
        return None
    if not states and NOT_EXECUTED_MARKER in evidence_text:
        return (
            response,
            "TOOL_EVIDENCE_NOT_EXECUTED",
            "The response says a refund happened, but the only refund evidence is a call "
            "that was not executed.",
        )
    if not states:
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


def _surviving_sentences(response: str, verdicts: list[DetectionVerdict]) -> str:
    """The reply's own sentences that carry no unsupported claim, verbatim.

    Claims are finer than sentences, so gluing the supported *claims*
    back together returned fragments: "10854), both currently on their
    way." reached a customer as the whole answer. A sentence either
    stands as the model wrote it or goes; nothing is reassembled.
    """
    rejected = [
        verdict.claim.strip().rstrip(".").casefold()
        for verdict in verdicts
        if not verdict.supported and verdict.claim.strip()
    ]
    kept = [
        sentence.strip()
        for sentence in split_sentences(response)
        if sentence.strip() and not any(claim in sentence.casefold() for claim in rejected)
    ]
    return " ".join(kept)
