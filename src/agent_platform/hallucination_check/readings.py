"""What decided a claim, step by step, in one form every agent's card and log share.

The check runs in a fixed order, and the card says it in that order:

    1  NLI           entailment / contradiction / neutral — decides when it is sure
    2  second opinion  asked only when NLI did not support the claim; confirms or not (a claim
                      that never reached it does not name it)
    3  rewrite        (an agent that repairs) what the sentence became, and what then

A step is a `Reading`: which check read the claim, what it answered, what that decided, and
the tone the decision takes (pass, fail, warn, dim). A card colours by the decision, never by
the claim: supported is pass, contradicted or removed is fail, undecided is warn. Scores are written to three places so a score at a bar never reads as
meeting it ("0.449 < 0.45").
"""

from typing import Literal

from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.hallucination_check.confidence import LOGPROB_TIER
from agent_platform.hallucination_check.evidence import TOOL_RESULT_KIND
from agent_platform.hallucination_check.ledger import FACT_TIER
from agent_platform.hallucination_check.resolver import (
    CONFIRM_TIER,
    DIRECT_TIER,
    NO_TIER,
    RESTATEMENT_MODEL,
)
from agent_platform.hallucination_check.types import DetectionVerdict, EvidenceCheck

Tone = Literal["pass", "fail", "warn", "dim"]
_PASS: Tone = "pass"
_FAIL: Tone = "fail"
_WARN: Tone = "warn"
_DIM: Tone = "dim"
# The NLI model's third label, as a reader should meet it.
_NEUTRAL = {"unknown": "neutral"}


class EvidenceRef(BaseModel):
    """One line of evidence a step read: what kind and where from, and its words.

    `where` names the evidence the way an operator looks it up — "Tool result ·
    get_order_status", "DB record · order 10677", "KB article · refunds.md#1.0", "Web page ·
    https://…" — from `hallucination_check.evidence_sources`, so each agent words its own.
    """

    where: str
    excerpt: str = ""


class Reading(BaseModel):
    """One step of a claim's check: who read it, what it said, what that decided, on what."""

    step: int
    check: str
    result: str = ""
    decision: str
    tone: Tone
    # How its evidence lines open ("Supported by", "Not decided by", "Contradicted by").
    lead: str = ""
    evidence: list[EvidenceRef] = []


def evidence_ref(kind: str, source: str, excerpt: str = "", entry: str = "") -> EvidenceRef:
    """A line of evidence named by its source prefix, else by its kind (`evidence_sources`).

    `entry` is a ledger id ("E3") a report cites the line by.
    """
    names = get_settings().hallucination_check.evidence_sources
    prefix = next(
        (p for p in sorted(names.prefixes, key=len, reverse=True) if source.startswith(p)), ""
    )
    template = names.prefixes[prefix] if prefix else names.kinds.get(kind, names.default)
    where = template.format(source=source, id=source[len(prefix) :], kind=kind)
    return EvidenceRef(where=f"{where} [{entry}]" if entry else where, excerpt=excerpt)


def _refs(lines: list[EvidenceCheck]) -> list[EvidenceRef]:
    """Each line once, named."""
    seen: dict[str, EvidenceRef] = {}
    for line in lines:
        seen.setdefault(line.excerpt, evidence_ref(line.tier, line.source, line.excerpt))
    return list(seen.values())


def model_label(model: str) -> str:
    """A model as the card and the log name it: `NLI DeBERTa`, `Laya` (`model_labels`)."""
    labels = get_settings().hallucination_check.model_labels
    return labels.get(model) or model.rsplit("/", 1)[-1]


def confirmer_id() -> str:
    """The second-opinion model's id as its evidence lines carry it ("" when none is on)."""
    confirmer = get_settings().hallucination_check.confirmer
    return f"{confirmer.model}/{confirmer.subfolder}" if confirmer.enabled else ""


def check_rule() -> str:
    """The order the check runs in, in one line, for the top of the card."""
    settings = get_settings().hallucination_check
    words = settings.readings
    nli = model_label(get_settings().llm.encoders.nli_model)
    if not settings.confirmer.enabled:
        return words.rule_nli_only.format(nli=nli)
    second = model_label(confirmer_id())
    return words.rule.format(nli=nli, confirmer=second, bar=settings.confirmer.min_score)


def readings(verdict: DetectionVerdict) -> list[Reading]:
    """The steps that decided one claim, in the order they ran."""
    first = _rule_reading(verdict)
    if first is not None:
        return [first]
    steps = [_nli_reading(verdict)]
    second = _confirmer_reading(verdict)
    if second is not None:
        steps.append(second.model_copy(update={"step": 2}))
    return steps


def nli_step(
    model: str, label: str, score: float, decided: bool, evidence: list[EvidenceRef] | None = None
) -> Reading:
    """Step 1, for a caller that ran NLI itself (a report's citation guard)."""
    words = get_settings().hallucination_check.readings
    if decided:
        decision, tone, outcome = _decided(label)
    else:
        decision, tone, outcome = words.undecided, _WARN, "undecided"
    return Reading(
        step=1,
        check=model_label(model),
        result=f"{_NEUTRAL.get(label, label)} {score:.3f}",
        decision=decision,
        tone=tone,
        lead=words.leads[outcome],
        evidence=evidence or [],
    )


def confirmer_step(
    step: int,
    model: str,
    label: str,
    score: float,
    evidence: list[EvidenceRef] | None = None,
) -> Reading:
    """The second opinion's answer, with the bar it was held to."""
    settings = get_settings().hallucination_check
    words, bar = settings.readings, settings.confirmer.min_score
    chose_supported = label == settings.confirmer.supported_label
    confirmed = chose_supported and score >= bar
    # The bar applies to a "supported" answer; any other answer confirms nothing whatever its score.
    rule = (words.at_least if score >= bar else words.below).format(bar=bar)
    said = label.replace("_", " ")
    result = (
        f"{said} {score:.3f} {rule}"
        if chose_supported
        else words.its_choice.format(answer=said, score=score)
    )
    return Reading(
        step=step,
        check=model_label(model),
        result=result,
        decision=words.confirmed if confirmed else words.not_confirmed,
        tone=_PASS if confirmed else _WARN,
        lead=words.leads["confirmed" if confirmed else "not_confirmed"],
        evidence=evidence or [],
    )


def _rule_reading(verdict: DetectionVerdict) -> Reading | None:
    """A claim a deterministic check decided: the record, logprobs, or a restatement.

    Every line it read is listed: "pending review" for order 10677 rests on the refund-status
    line as well as on the order's total, and the card names both.
    """
    settings = get_settings().hallucination_check
    methods = settings.methods
    if verdict.tier in (DIRECT_TIER, FACT_TIER):
        name = methods.record
    elif verdict.tier == LOGPROB_TIER:
        name = methods.logprobs.format(prob=verdict.confidence)
    elif verdict.detector_model == RESTATEMENT_MODEL:
        name = methods.restated_source if _from_tool(verdict) else methods.restated
    else:
        return None
    decision, tone, outcome = _decided("entailment" if verdict.supported else "contradiction")
    read = _refs([c for c in verdict.evidence_checks if c.decisive])
    if not read and verdict.rationale:
        read = [EvidenceRef(where=settings.readings.why, excerpt=verdict.rationale)]
    lead = settings.readings.leads[outcome]
    return Reading(step=1, check=name, decision=decision, tone=tone, lead=lead, evidence=read)


def _nli_reading(verdict: DetectionVerdict) -> Reading:
    """Step 1: what NLI made of the claim, and whether that decided it."""
    lines = [c for c in verdict.evidence_checks if c.model and c.model != confirmer_id()]
    line = _deciding(lines)
    nli_decided = verdict.tier not in (NO_TIER, CONFIRM_TIER) and not verdict.undetermined
    if line is None:
        words = get_settings().hallucination_check.readings
        nli = model_label(get_settings().llm.encoders.nli_model)
        return Reading(step=1, check=nli, decision=words.nothing_related, tone=_WARN,
                       lead=words.leads["undecided"])  # fmt: skip
    return nli_step(line.model, line.nli_label, line.nli_score, nli_decided, _refs([line]))


def _confirmer_reading(verdict: DetectionVerdict) -> Reading | None:
    """Step 2: the second opinion's answer — only when it was asked."""
    model = confirmer_id()
    lines = [c for c in verdict.evidence_checks if model and c.model == model]
    line = _deciding(lines)
    # A claim NLI decided never reached the second opinion, so the card does not name it.
    if line is None:
        return None
    return confirmer_step(2, model, line.nli_label, line.nli_score, _refs([line]))


def _deciding(lines: list[EvidenceCheck]) -> EvidenceCheck | None:
    """The line that decided, else the one the model was surest of."""
    return max(lines, key=lambda c: (c.decisive, c.nli_score), default=None)


def _decided(label: str) -> tuple[str, Tone, str]:
    """The decision's word, its tone, and the key its lead is worded under (`readings.leads`)."""
    words = get_settings().hallucination_check.readings
    if label == "entailment":
        return words.supported, _PASS, "supported"
    if label == "contradiction":
        return words.contradicted, _FAIL, "contradicted"
    return words.undecided, _WARN, "undecided"


def _from_tool(verdict: DetectionVerdict) -> bool:
    return any(c.decisive and c.tier == TOOL_RESULT_KIND for c in verdict.evidence_checks)


__all__ = [
    "EvidenceRef",
    "Reading",
    "evidence_ref",
    "check_rule",
    "confirmer_id",
    "confirmer_step",
    "model_label",
    "nli_step",
    "readings",
]
