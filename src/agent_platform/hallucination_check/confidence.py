"""Step 1 of the hallucination check: how sure the model was of the words it wrote.

A model that is guessing tends to spread probability over several next tokens; the
token it emits then carries a low log-probability. Asking for `logprobs` costs nothing
extra (`llm.logprobs`), so every reply can be scored before anything else looks at it:
each sentence by the tokens that wrote it — their mean probability, and how many fell
below a floor (`hallucination_check.confidence`).

Confidence is not truth: a model can be confidently wrong, so every claim still goes
to grounding. But two things logprobs catch on their own, deterministically (`gate`):

- a *degenerate* sentence — its content tokens (grammar words left out, where a low
  probability means another phrasing, not a guess) average below
  `degenerate_mean_prob` — is cut whatever grounding says ("The-something entirely");
- a *shaky specific* — an id, amount, date or number written below `slot_min_prob`
  — cuts its claim unless the record confirmed it: the made-up-number signature.

Each becomes a real verdict with its own reason code, so the sentence it sits in is
removed and the reason says why — never a list entry the trimming does not read. The
result is level 1 of the hallucination check (`levels.py`): it travels first inside the
agent's one `hallucination_check` event, never as an event of its own.

The tokens cover everything the model generated, reasoning included (a reasoning
model's thinking, a channel's analysis), so a sentence is found *within* the generated
text and only the tokens over it are scored. No agent passes logprobs along: the
outermost model client keeps each completion's on the turn (`unit_context`).
"""

from __future__ import annotations

import math
import re
import time
from typing import Any

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.config.settings import ConfidenceSettings
from agent_platform.hallucination_check.claims import sentence_pieces, specific_spans
from agent_platform.hallucination_check.levels import CONFIDENCE_LEVEL, CheckLevel
from agent_platform.hallucination_check.resolver import CLAIM_UNSUPPORTED
from agent_platform.hallucination_check.types import DetectionVerdict, GenerationAssessment
from agent_platform.observability import checks
from agent_platform.observability.unit_context import Generation, current_turn

# Why logprobs alone cut a sentence: it was guessed, or a specific in it was.
LOW_CONFIDENCE_DEGENERATE = "LOW_CONFIDENCE_DEGENERATE"
LOW_CONFIDENCE_SPECIFIC = "LOW_CONFIDENCE_SPECIFIC"
# The tier a logprob verdict names.
LOGPROB_TIER = "logprobs"


class TokenScore(BaseModel):
    """One scored token: where it sits in its sentence, and how sure the model was."""

    start: int
    end: int
    prob: float
    # Part of an id, amount, date or number.
    slot: bool = False


class SentenceConfidence(BaseModel):
    """One sentence of the reply, scored by the tokens that wrote it."""

    index: int
    sentence: str
    mean_prob: float
    min_prob: float
    weakest: str
    low: bool
    # The mean over content tokens only, and whether it marks the sentence as guessed.
    content_mean: float = 1.0
    degenerate: bool = False
    # The least sure token of any id, amount, date or number; None when there is none.
    slot_min: float | None = None
    weakest_slot: str = ""
    tokens: list[TokenScore] = Field(default_factory=list)


class ConfidenceAssessment(BaseModel):
    """A reply's sentences by the model's confidence in them (or why none were scored)."""

    available: bool = False
    model: str = ""
    reason: str = ""
    sentences: list[SentenceConfidence] = Field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def low(self) -> list[SentenceConfidence]:
        return [sentence for sentence in self.sentences if sentence.low]

    @property
    def low_indexes(self) -> set[int]:
        return {sentence.index for sentence in self.low}

    @property
    def degenerate(self) -> list[SentenceConfidence]:
        return [sentence for sentence in self.sentences if sentence.degenerate]


class ConfidenceCheck:
    """Scores a reply's sentences from the turn's token log-probabilities."""

    def __init__(self, settings: ConfidenceSettings | None = None) -> None:
        self._settings = settings or get_settings().hallucination_check.confidence
        self._ignored = re.compile(self._settings.ignore_token_pattern)
        self._function = re.compile(self._settings.function_token_pattern)

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    def assess(self, text: str) -> ConfidenceAssessment:
        """Score every sentence of `text`; `available=False` (with the reason) when it can't."""
        started = time.perf_counter()
        words = self._settings.card
        pieces = [piece for piece, _ in sentence_pieces(text)]
        generation = _generation_for(pieces)
        if generation is None:
            return ConfidenceAssessment(reason=words.no_logprobs)
        generated, offsets = _joined(generation.tokens)
        scored = [
            score
            for index, piece in enumerate(pieces)
            if (score := self._score(index, piece, generation, generated, offsets)) is not None
        ]
        return ConfidenceAssessment(
            available=bool(scored),
            model=generation.model,
            reason="" if scored else words.no_match,
            sentences=scored,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _score(
        self,
        index: int,
        piece: str,
        generation: Generation,
        generated: str,
        offsets: list[int],
    ) -> SentenceConfidence | None:
        sentence = piece.strip()
        if not sentence:
            return None
        start = generated.rfind(sentence)
        if start < 0:
            return None
        end = start + len(sentence)
        tokens = [
            (token, math.exp(logprob), offset - start)
            for (token, logprob), offset in zip(generation.tokens, offsets, strict=True)
            if offset < end and offset + len(token) > start and not self._ignored.match(token)
        ]
        if not tokens:
            return None
        probs = [prob for _, prob, _ in tokens]
        weakest = min(tokens, key=lambda item: item[1])
        mean = sum(probs) / len(probs)
        low_share = sum(prob < self._settings.min_token_prob for prob in probs) / len(probs)
        low = mean < self._settings.sentence_mean_prob or (
            low_share > self._settings.max_low_token_share
        )
        scored = _token_scores(sentence, tokens)
        return SentenceConfidence(
            index=index,
            sentence=sentence,
            mean_prob=mean,
            min_prob=weakest[1],
            weakest=weakest[0],
            low=low,
            tokens=scored,
            **self._content(tokens),
            **_slot(sentence, scored),
        )

    def _content(self, tokens: list[tuple[str, float, int]]) -> dict[str, Any]:
        """The content-token mean, and whether it marks the sentence as guessed."""
        content = [prob for token, prob, _ in tokens if not self._function.match(token)]
        mean = sum(content) / len(content) if content else 1.0
        return {"content_mean": mean, "degenerate": mean < self._settings.degenerate_mean_prob}


def gate(
    assessment: GenerationAssessment,
    confidence: ConfidenceAssessment,
    settings: ConfidenceSettings | None = None,
) -> GenerationAssessment:
    """Cut what the model guessed: a degenerate sentence, or a specific no record confirmed.

    Each becomes a verdict with its own reason code, so the trimming that reads
    verdicts removes its sentence. A degenerate sentence with no claim in it (too
    short to check, like "The-something entirely") gets one of its own.
    """
    floor = (settings or get_settings().hallucination_check.confidence).slot_min_prob
    degenerate = {item.index: item for item in confidence.degenerate}
    shaky = {
        item.index: item
        for item in confidence.sentences
        if item.slot_min is not None and item.slot_min < floor
    }
    if not degenerate and not shaky:
        return assessment
    verdicts = [_gated(v, degenerate, shaky, confidence.model) for v in assessment.verdicts]
    covered = {verdict.sentence for verdict in verdicts}
    verdicts += [
        _guessed(item, confidence.model)
        for index, item in degenerate.items()
        if index not in covered
    ]
    return _reassessed(assessment, verdicts)


def _gated(
    verdict: DetectionVerdict,
    degenerate: dict[int, SentenceConfidence],
    shaky: dict[int, SentenceConfidence],
    model: str,
) -> DetectionVerdict:
    """The verdict, overruled when its sentence was guessed or its specific was not confirmed."""
    if verdict.sentence in degenerate:
        return _low(verdict, LOW_CONFIDENCE_DEGENERATE, degenerate[verdict.sentence], model)
    item = shaky.get(verdict.sentence)
    # Only what grounding could not decide: a contradiction keeps its own, better reason.
    unsettled = verdict.undetermined or verdict.reason_code == CLAIM_UNSUPPORTED
    if item is None or not unsettled:
        return verdict
    return _low(verdict, LOW_CONFIDENCE_SPECIFIC, item, model)


def _low(
    verdict: DetectionVerdict, code: str, item: SentenceConfidence, model: str
) -> DetectionVerdict:
    rationale = (
        f"content tokens averaged {item.content_mean:.0%}"
        if code == LOW_CONFIDENCE_DEGENERATE
        else f"{item.weakest_slot!r} written at {item.slot_min or 0:.0%} and no record confirms it"
    )
    return verdict.model_copy(
        update={
            "supported": False,
            "undetermined": False,
            "reason_code": code,
            "tier": LOGPROB_TIER,
            "detector_model": model,
            "confidence": item.content_mean if code == LOW_CONFIDENCE_DEGENERATE else item.slot_min,
            "rationale": rationale,
        }
    )


def _guessed(item: SentenceConfidence, model: str) -> DetectionVerdict:
    """A verdict for a degenerate sentence no claim came from."""
    blank = DetectionVerdict(claim=item.sentence, supported=False, confidence=0.0)
    return _low(
        blank.model_copy(update={"sentence": item.index, "claim_class": "rule"}),
        LOW_CONFIDENCE_DEGENERATE,
        item,
        model,
    )


def _reassessed(
    assessment: GenerationAssessment, verdicts: list[DetectionVerdict]
) -> GenerationAssessment:
    unsupported = [v.claim for v in verdicts if not v.supported and not v.undetermined]
    return assessment.model_copy(
        update={
            "verdicts": verdicts,
            "unsupported": unsupported,
            "undetermined": [v.claim for v in verdicts if v.undetermined],
            "hallucination_rate": len(unsupported) / len(verdicts) if verdicts else 0.0,
        }
    )


def _token_scores(sentence: str, tokens: list[tuple[str, float, int]]) -> list[TokenScore]:
    """Each token's span in the sentence, marked when it writes part of a specific."""
    slots = specific_spans(sentence)
    scores = []
    for token, prob, start in tokens:
        lead = len(token) - len(token.lstrip())
        begin, end = max(0, start + lead), min(len(sentence), start + len(token))
        in_slot = any(begin < s_end and end > s_start for s_start, s_end in slots)
        scores.append(TokenScore(start=begin, end=end, prob=prob, slot=in_slot))
    return scores


def _slot(sentence: str, scored: list[TokenScore]) -> dict[str, Any]:
    """The least sure token of any specific in the sentence, if it has one."""
    slots = [token for token in scored if token.slot]
    if not slots:
        return {}
    weakest = min(slots, key=lambda token: token.prob)
    return {"slot_min": weakest.prob, "weakest_slot": sentence[weakest.start : weakest.end]}


def report_confidence(confidence: ConfidenceAssessment) -> CheckLevel:
    """Record the step as a check row, and return it as level 1 of the hallucination check."""
    verdict = (
        checks.SKIPPED
        if not confidence.available
        else checks.FLAG
        if confidence.low
        else checks.PASS
    )
    checks.record_check(
        checks.LOGPROB,
        checks.GROUNDING_PURPOSE,
        checks.LLM,
        confidence.model,
        verdict,
        confidence.latency_ms,
        items=len(confidence.sentences) if confidence.available else None,
        flagged=len(confidence.low) if confidence.available else None,
    )
    return CheckLevel(
        level=CONFIDENCE_LEVEL,
        name=get_settings().hallucination_check.levels.confidence,
        verdict=verdict,
        summary=_summary(confidence),
        lines=_lines(confidence),
    )


def _summary(confidence: ConfidenceAssessment) -> str:
    words = get_settings().hallucination_check.confidence.card
    if not confidence.available:
        return words.skipped.format(reason=confidence.reason)
    scored, low = len(confidence.sentences), len(confidence.low)
    if low:
        return words.heading.format(low=low, scored=scored, model=confidence.model)
    return words.heading_clean.format(scored=scored, model=confidence.model)


def _lines(confidence: ConfidenceAssessment) -> list[str]:
    """One line per scored sentence: its mean and lowest token probability, low ones marked."""
    words = get_settings().hallucination_check.confidence.card
    return [
        words.sentence_line.format(
            mark=words.low_mark if item.low else words.ok_mark,
            mean=item.mean_prob,
            min=item.min_prob,
            weakest=item.weakest,
            sentence=item.sentence,
        )
        for item in confidence.sentences
    ]


def _generation_for(pieces: list[str]) -> Generation | None:
    """The newest completion this turn whose text holds the reply's sentences."""
    turn = current_turn()
    if turn is None:
        return None
    sentences = [piece.strip() for piece in pieces if piece.strip()]
    for generation in reversed(turn.generations):
        generated, _ = _joined(generation.tokens)
        if any(sentence in generated for sentence in sentences):
            return generation
    return None


def _joined(tokens: list[tuple[str, float]]) -> tuple[str, list[int]]:
    """The generated text, and where each token starts in it."""
    offsets, position = [], 0
    for token, _ in tokens:
        offsets.append(position)
        position += len(token)
    return "".join(token for token, _ in tokens), offsets


__all__ = [
    "LOGPROB_TIER",
    "LOW_CONFIDENCE_DEGENERATE",
    "LOW_CONFIDENCE_SPECIFIC",
    "ConfidenceAssessment",
    "ConfidenceCheck",
    "SentenceConfidence",
    "TokenScore",
    "gate",
    "report_confidence",
]
