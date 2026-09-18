"""Natural-language inference for the cascade's tier 4 (E10).

Entailment is the tier that separates *sounds alike* from *means the
same*. The lexical model here is deliberately modest — it is a fallback
that works offline and states its own limits, not a claim to have
reimplemented an NLI model in fifty lines.
"""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agentic_commons.config import get_settings
from agentic_commons.llm.embeddings import tokenize


class EntailmentResult(BaseModel):
    """Whether premise entails hypothesis, and how confidently."""

    entailed: bool
    score: float
    rationale: str = ""


@runtime_checkable
class EntailmentModel(Protocol):
    """Judges whether a premise entails a hypothesis."""

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult: ...


class LexicalEntailmentModel:
    """Token-coverage entailment with negation-polarity awareness.

    Coverage answers "is the hypothesis' content present in the premise?";
    polarity then asks "and does it say the same thing about it?". Without
    the second question, "the refund was approved" and "the refund was not
    approved" score identically — the exact failure this tier exists to
    catch.
    """

    def __init__(self) -> None:
        settings = get_settings().evals.nli
        self._min_overlap = settings.min_overlap
        self._negations = {marker.casefold() for marker in settings.negation_markers}

    def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        """Score hypothesis coverage by the premise, then compare polarity."""
        coverage = self._coverage(premise, hypothesis)
        if coverage < self._min_overlap:
            return EntailmentResult(
                entailed=False,
                score=0.0,
                rationale=f"premise covers only {coverage:.0%} of the hypothesis",
            )
        same_polarity = self._is_negated(premise) == self._is_negated(hypothesis)
        return EntailmentResult(
            entailed=same_polarity,
            score=coverage,
            rationale=(
                f"coverage {coverage:.0%}, {'matching' if same_polarity else 'opposing'} polarity"
            ),
        )

    def _coverage(self, premise: str, hypothesis: str) -> float:
        """Fraction of the hypothesis' content tokens present in the premise."""
        hypothesis_tokens = set(tokenize(hypothesis)) - self._negations
        if not hypothesis_tokens:
            return 0.0
        premise_tokens = set(tokenize(premise))
        return len(hypothesis_tokens & premise_tokens) / len(hypothesis_tokens)

    def _is_negated(self, text: str) -> bool:
        """Whether the text carries a negation marker."""
        lowered = text.casefold()
        return any(marker in lowered for marker in self._negations)
