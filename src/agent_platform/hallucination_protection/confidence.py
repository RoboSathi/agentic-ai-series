"""Step 1 of the hallucination check: how sure the model was of the words it wrote.

A model that is guessing tends to spread probability over several next tokens; the
token it emits then carries a low log-probability. Asking for `logprobs` costs nothing
extra (`llm.logprobs`), so every reply can be scored before anything else looks at it:
each sentence by the tokens that wrote it — their mean probability, and how many fell
below a floor (`hallucination_protection.confidence`).

Confidence is not truth: a model can be confidently wrong, so every claim still goes
to grounding. What a low score changes is the benefit of the doubt — a claim grounding
leaves *undetermined* in a low-confidence sentence is treated as unsupported
(`promote`), and the reader sees which sentences the model was unsure of. The result is
level 1 of the hallucination check (`levels.py`): it travels first inside the agent's one
`hallucination_check` event, never as an event of its own.

The tokens cover everything the model generated, reasoning included (a reasoning
model's thinking, a channel's analysis), so a sentence is found *within* the generated
text and only the tokens over it are scored. No agent passes logprobs along: the
outermost model client keeps each completion's on the turn (`unit_context`).
"""

from __future__ import annotations

import math
import re
import time

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.config.settings import ConfidenceSettings
from agent_platform.hallucination_protection.claims import sentence_pieces
from agent_platform.hallucination_protection.levels import CONFIDENCE_LEVEL, CheckLevel
from agent_platform.hallucination_protection.types import GenerationAssessment
from agent_platform.observability import checks
from agent_platform.observability.unit_context import Generation, current_turn


class SentenceConfidence(BaseModel):
    """One sentence of the reply, scored by the tokens that wrote it."""

    index: int
    sentence: str
    mean_prob: float
    min_prob: float
    weakest: str
    low: bool


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


class ConfidenceCheck:
    """Scores a reply's sentences from the turn's token log-probabilities."""

    def __init__(self, settings: ConfidenceSettings | None = None) -> None:
        self._settings = settings or get_settings().hallucination_protection.confidence
        self._ignored = re.compile(self._settings.ignore_token_pattern)

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
            (token, math.exp(logprob))
            for (token, logprob), offset in zip(generation.tokens, offsets, strict=True)
            if offset < end and offset + len(token) > start and not self._ignored.match(token)
        ]
        if not tokens:
            return None
        probs = [prob for _, prob in tokens]
        weakest = min(tokens, key=lambda item: item[1])
        mean = sum(probs) / len(probs)
        low_share = sum(prob < self._settings.min_token_prob for prob in probs) / len(probs)
        low = mean < self._settings.sentence_mean_prob or (
            low_share > self._settings.max_low_token_share
        )
        return SentenceConfidence(
            index=index,
            sentence=sentence,
            mean_prob=mean,
            min_prob=weakest[1],
            weakest=weakest[0],
            low=low,
        )


def promote(
    assessment: GenerationAssessment, confidence: ConfidenceAssessment
) -> GenerationAssessment:
    """Claims grounding left undetermined, in a low-confidence sentence, become unsupported."""
    low = confidence.low_indexes
    if not low:
        return assessment
    doubted = {v.claim for v in assessment.verdicts if v.undetermined and v.sentence in low}
    if not doubted:
        return assessment
    unsupported = [*assessment.unsupported, *sorted(doubted)]
    checked = len(assessment.verdicts)
    return assessment.model_copy(
        update={
            "unsupported": unsupported,
            "undetermined": [c for c in assessment.undetermined if c not in doubted],
            "hallucination_rate": len(unsupported) / checked if checked else 0.0,
        }
    )


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
        name=get_settings().hallucination_protection.levels.confidence,
        verdict=verdict,
        summary=_summary(confidence),
        lines=_lines(confidence),
    )


def _summary(confidence: ConfidenceAssessment) -> str:
    words = get_settings().hallucination_protection.confidence.card
    if not confidence.available:
        return words.skipped.format(reason=confidence.reason)
    scored, low = len(confidence.sentences), len(confidence.low)
    if low:
        return words.heading.format(low=low, scored=scored, model=confidence.model)
    return words.heading_clean.format(scored=scored, model=confidence.model)


def _lines(confidence: ConfidenceAssessment) -> list[str]:
    """One line per scored sentence: its mean and lowest token probability, low ones marked."""
    words = get_settings().hallucination_protection.confidence.card
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
    "ConfidenceAssessment",
    "ConfidenceCheck",
    "SentenceConfidence",
    "promote",
    "report_confidence",
]
