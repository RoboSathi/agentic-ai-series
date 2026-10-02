"""Natural-language inference for the cascade's tier 4 (E10).

Entailment is the tier that separates *sounds alike* from *means the
same*. The lexical model here is deliberately modest — it is the fallback that
works offline and states its own limits, not a claim to have
reimplemented an NLI model in fifty lines. The model that actually
answers the question is a hosted cross-encoder
(`llm.encoders.HostedNliModel`); this one is what the harness uses when
that cannot be reached.
"""

import re
from collections.abc import Iterable
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from agentic_commons.config import get_settings
from agentic_commons.llm.embeddings import tokenize


class EntailmentResult(BaseModel):
    """Three-way NLI result: entailment, contradiction, or unknown.

    `calibrated` says what `score` means, and callers must not compare
    the two kinds. A cross-encoder returns the winning label's
    probability, which a confidence threshold can be set against. The
    lexical fallback returns token coverage, which cannot: a sentence
    restating its evidence exactly still covers only the content words
    the evidence happens to share, so "A refund has already been
    approved for order 10864" scores 0.44 against "Order 10864 refund
    status: approved" — a perfect paraphrase, four tokens out of nine.
    Judging that against a probability threshold rejects true claims.
    """

    label: Literal["entailment", "contradiction", "unknown"]
    entailed: bool
    score: float
    rationale: str = ""
    model: str = ""
    # False when `score` is a heuristic rather than a probability, so a
    # caller knows its confidence threshold does not apply.
    calibrated: bool = True


@runtime_checkable
class EntailmentModel(Protocol):
    """Judges whether a premise entails a hypothesis.

    Async because the good implementations are hosted. A synchronous
    interface would force every caller of a network model to block an
    event loop, or force the model behind a thread pool — both worse
    than one `await` in a method the lexical model answers instantly.
    """

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult: ...


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
        self._negation = negation_pattern(self._negations)

    @property
    def model_name(self) -> str:
        """Name the deterministic fallback used for this decision."""
        return "lexical-entailment-fallback"

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        """Score hypothesis coverage by the premise, then compare polarity."""
        coverage = self._coverage(premise, hypothesis)
        if coverage < self._min_overlap:
            return EntailmentResult(
                label="unknown",
                entailed=False,
                score=0.0,
                rationale=f"premise covers only {coverage:.0%} of the hypothesis",
                model="lexical-entailment-fallback",
                calibrated=False,
            )
        same_polarity = self._is_negated(premise) == self._is_negated(hypothesis)
        label: Literal["entailment", "contradiction"] = (
            "entailment" if same_polarity else "contradiction"
        )
        return EntailmentResult(
            label=label,
            entailed=same_polarity,
            score=coverage,
            rationale=(
                f"coverage {coverage:.0%}, {'matching' if same_polarity else 'opposing'} polarity"
            ),
            model="lexical-entailment-fallback",
            calibrated=False,
        )

    def _coverage(self, premise: str, hypothesis: str) -> float:
        """Fraction of the hypothesis' content tokens present in the premise."""
        hypothesis_tokens = set(tokenize(hypothesis)) - self._negations
        if not hypothesis_tokens:
            return 0.0
        premise_tokens = set(tokenize(premise))
        return len(hypothesis_tokens & premise_tokens) / len(hypothesis_tokens)

    def _is_negated(self, text: str) -> bool:
        """Whether the text carries a negation marker, as a word.

        A substring test found "no" inside "announced", "economy" and "Now:", and read an
        exact restatement as its opposite.
        """
        return bool(self._negation.search(text.casefold()))


def negation_pattern(markers: Iterable[str]) -> re.Pattern[str]:
    """Each marker as a whole word; a contraction ending ("n't") as the end of one."""
    parts = [
        rf"\b{re.escape(m)}\b" if m.isalnum() else rf"{re.escape(m)}\b"
        for m in sorted(markers, key=len, reverse=True)
        if m
    ]
    return re.compile("|".join(parts) or r"(?!)")


# NLI checkpoints spell their labels ENTAILMENT/entailment and so on,
# depending on who trained them; everything is compared case-folded.
ENTAILMENT = "entailment"
CONTRADICTION = "contradiction"
NEUTRAL = "neutral"


def entailment_from_probabilities(probabilities: dict[str, float], model: str) -> EntailmentResult:
    """The winning NLI label as a result, whichever model produced the scores.

    Hosted and local cross-encoders both answer with three labelled
    probabilities; turning those into a verdict is one rule, kept here so
    the two can never disagree about what a score means.
    """
    labels = {label.casefold(): score for label, score in probabilities.items()}
    entailment = labels.get(ENTAILMENT, 0.0)
    contradiction = labels.get(CONTRADICTION, 0.0)
    neutral = labels.get(NEUTRAL, 0.0)
    score, label = max(
        (entailment, "entailment"),
        (contradiction, "contradiction"),
        (neutral, "unknown"),
        key=lambda item: item[0],
    )
    return EntailmentResult(
        label=label,  # type: ignore[arg-type]
        entailed=label == "entailment",
        # The winning label's probability is confidence in the
        # classification, not text similarity.
        score=score,
        model=model,
        rationale=(
            f"{model}: entailment {entailment:.2f}, "
            f"contradiction {contradiction:.2f}, neutral {neutral:.2f}"
        ),
    )


def create_entailment_model() -> EntailmentModel:
    """The best entailment model this deployment can reach (Factory).

    The local cross-encoder when configured, a hosted one when configured
    and credentialed, the lexical model otherwise. Callers never choose:
    the decision is a deployment fact, and a component that picked its own
    model would make "which NLI ran?" un-answerable from config.
    """
    settings = get_settings().llm.encoders
    if not settings.enabled:
        return LexicalEntailmentModel()
    if settings.nli_provider == "huggingface_local":
        from agentic_commons.llm.local_nli import shared_local_nli_model

        return shared_local_nli_model()
    if not settings.usable:
        return LexicalEntailmentModel()
    from agentic_commons.llm.encoders import HostedNliModel

    return HostedNliModel()


def preload_entailment_model() -> None:
    """Load a local NLI model now, in the calling thread, if one is configured.

    For a console that is about to start an event loop: native model
    loading after that point trips macOS descriptor checks. A model that
    cannot load is not a reason not to start — the first claim check
    falls back to the lexical model and says so.
    """
    model = create_entailment_model()
    load = getattr(model, "ensure_loaded_sync", None)
    if callable(load):
        load()
