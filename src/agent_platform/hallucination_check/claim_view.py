"""One claim as every card and log line shows it, projected once from the check's verdict.

The resolver decides a claim; the guardrail acts on the decision; this module only says it.
What decided the claim, step by step, is `readings.py`'s — the projection carries those
steps, and a card renders them rather than re-deriving which line settled the claim from the
evidence it lists. The evidence and per-model scores ride along for the log, which keeps
everything.

Every word comes from `hallucination_check.card` and `hallucination_check.methods`, so an
agent names its own evidence tiers ("customer message") in its overlay, never here.
"""

from collections.abc import Iterable
from typing import Any

from agent_platform.config import get_settings
from agent_platform.context.types import Chunk
from agent_platform.hallucination_check.confidence import (
    LOGPROB_TIER,
    LOW_CONFIDENCE_DEGENERATE,
    LOW_CONFIDENCE_SPECIFIC,
)
from agent_platform.hallucination_check.direct import amounts, numbers, record_ids
from agent_platform.hallucination_check.evidence import TOOL_RESULT_KIND
from agent_platform.hallucination_check.ledger import FACT_TIER
from agent_platform.hallucination_check.readings import (
    deciding,
    label_name,
    model_label,
    readings,
)
from agent_platform.hallucination_check.resolver import (
    CLAIM_UNSUPPORTED,
    CONFIRM_TIER,
    DIRECT_TIER,
    RESTATEMENT_MODEL,
)
from agent_platform.hallucination_check.types import DetectionVerdict, EvidenceCheck

# Why a claim decided by the record is named by its provenance rather than its tier.
DETERMINISTIC_PROVENANCE = "deterministic_provenance"


def claim_check(verdict: DetectionVerdict) -> dict[str, Any]:
    """One claim's verdict, what decided it, and the evidence trail behind it."""
    return {
        "claim": verdict.claim,
        "claim_class": verdict.claim_class,
        "verdict": verdict_word(verdict),
        "decided_by": decided_by(verdict),
        "method": method(verdict),
        "sentence": verdict.sentence,
        "basis": basis(verdict),
        "detector": verdict.detector_model,
        "threshold": verdict.threshold,
        "confidence": round(verdict.confidence, 3),
        "rationale": verdict.rationale,
        **({"reason_code": verdict.reason_code} if verdict.reason_code else {}),
        "retrieval": _retrieval(verdict),
        "evidence": [_evidence(check) for check in verdict.evidence_checks],
        "scores": model_scores(verdict),
        # What decided it, step by step: NLI, then (only if NLI did not support it) the
        # second opinion — each with its answer, what that decided, and the line it read.
        "readings": [reading.model_dump() for reading in readings(verdict)],
    }


def verdict_word(verdict: DetectionVerdict) -> str:
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


def decided_by(verdict: DetectionVerdict) -> str:
    return DETERMINISTIC_PROVENANCE if verdict.tier == DIRECT_TIER else verdict.tier


def method(verdict: DetectionVerdict) -> str:
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


def basis(verdict: DetectionVerdict) -> str:
    """The one number a card shows: a model's probability. Nothing else is a score."""
    if (
        verdict.undetermined
        or verdict.reason_code == CLAIM_UNSUPPORTED
        or verdict.tier in (DIRECT_TIER, FACT_TIER, LOGPROB_TIER, CONFIRM_TIER)
        or verdict.detector_model == RESTATEMENT_MODEL
    ):
        return ""
    label = "entailment" if verdict.supported else "contradiction"
    return f"{model_label(verdict.detector_model)} {label_name(label).capitalize()} {verdict.confidence:.2f}"


def model_scores(verdict: DetectionVerdict) -> dict[str, str]:
    """Each model's reading of the claim, in one line: its deciding answer, else its surest.

    `{"NLI DeBERTa": "neutral 1.00", "Laya": "supported 0.46"}` — which model said what, so
    a claim confirmed by the second opinion still shows what NLI made of it.
    """
    by_model: dict[str, list[EvidenceCheck]] = {}
    for check in verdict.evidence_checks:
        if check.model:
            by_model.setdefault(check.model, []).append(check)
    best = {model: deciding(lines) for model, lines in by_model.items()}
    return {
        model_label(model): f"{label_name(c.nli_label)} {c.nli_score:.2f}"
        for model, c in best.items()
        if c is not None
    }


def tier_name(kind: str) -> str:
    """An evidence tier as the card names it (`hallucination_check.card.tier_names`)."""
    return get_settings().hallucination_check.card.tier_names.get(kind, kind)


def provenance_report(response: str, evidence: list[Chunk]) -> list[str]:
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


def tiers_searched(evidence: Iterable[Chunk]) -> str:
    """What was looked in, e.g. `3 tool result · 1 policy · 4 knowledge`."""
    counts: dict[str, int] = {}
    for chunk in evidence:
        name = tier_name(chunk.kind)
        counts[name] = counts.get(name, 0) + 1
    joined = " · ".join(f"{count} {name}" for name, count in counts.items())
    return joined or get_settings().hallucination_check.card.nothing_searched


def _found(label: str, holders: list[Chunk]) -> str:
    """`✓ $12.00 — tool result`, or `✗ $75.00 — not in any evidence`."""
    words = get_settings().hallucination_check.card
    tiers = list(dict.fromkeys(tier_name(chunk.kind) for chunk in holders))
    if not tiers:
        return words.provenance_missing.format(label=label)
    return words.provenance_found.format(label=label, tiers=", ".join(tiers))


def _retrieval(verdict: DetectionVerdict) -> str:
    words = get_settings().hallucination_check.card
    return words.retrieval.format(
        top_k=verdict.retrieval_top_k,
        method=verdict.retrieval_method or words.retrieval_default_method,
    )


def _evidence(check: EvidenceCheck) -> dict[str, Any]:
    return {
        "rank": check.rank,
        "similarity": check.similarity,
        "tier": tier_name(check.tier),
        "excerpt": check.excerpt,
        "nli_label": check.nli_label,
        "nli_score": check.nli_score,
        "decisive": check.decisive,
        "chosen": check.chosen,
        "source": check.source,
        "model": model_label(check.model) if check.model else "",
    }


__all__ = [
    "claim_check",
    "decided_by",
    "model_scores",
    "provenance_report",
    "tier_name",
    "tiers_searched",
    "verdict_word",
]
