"""The research kind's fast path: a local classifier in front of the LLM.

Which kind of research a `/research` question asks for is a classification problem, and a large
model is a heavy tool for it (about 1,400 tokens and several seconds a turn). This is the
same shape as the Customer Support agent's front door: a small local encoder (e5 embeddings,
voting among the nearest of this agent's own labelled examples; nothing is fine-tuned) answers
when it is sure, and everything it is unsure about goes to the LLM.

    refusal patterns   deterministic, first — no model is asked whether to help build a weapon
    time window        deterministic ("last 1 month" -> 30); it is not a classification problem
    cue rules          deterministic: a message that plainly names a research kind (kind_rules.py)
    encoder            a label and a softmax score; confident AND unambiguous -> answer here
    LLM                the fallback, for low confidence, a narrow margin, or any failure

The verdict says which path answered (`source`: rule, encoder or llm) and the card shows it.
"""

from typing import Any, Protocol

from pydantic import BaseModel

from agent_platform.intent.types import IntentCandidate
from agent_platform.logging import get_logger, instrument
from research_agent.config.settings import KindEncoderSettings
from research_agent.domain.scope import RESEARCH_ARCHETYPES, KindVerdict
from research_agent.scoping.kind import LlmKindClassifier
from research_agent.scoping.kind_rules import KindRules

# The classifier's labels: one per research kind (`general` included).
LABELS = RESEARCH_ARCHETYPES


class IntentBackend(Protocol):
    """What the research kind needs of an encoder (`HuggingFaceIntentClassifier` satisfies it)."""

    async def classify(self, text: str) -> list[IntentCandidate]: ...


class EncoderRead(BaseModel):
    """The classifier's top label for a message, its share of the vote and the runner-up's."""

    label: str
    share: float
    runner_up: float

    def as_fields(self) -> dict[str, Any]:
        return {
            "encoder_label": self.label,
            "encoder_share": self.share,
            "encoder_runner_up": self.runner_up,
        }


def _read(candidates: list[IntentCandidate]) -> EncoderRead | None:
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    if not ranked or ranked[0].intent not in LABELS:
        return None
    runner_up = ranked[1].score if len(ranked) > 1 else 0.0
    return EncoderRead(label=ranked[0].intent, share=ranked[0].score, runner_up=runner_up)


@instrument("research_agent")
class EncoderKindClassifier:
    """Answers confidently-classified questions itself; defers the rest to the LLM."""

    def __init__(
        self,
        backend: IntentBackend | None,
        fallback: LlmKindClassifier,
        settings: KindEncoderSettings,
        rules: KindRules | None = None,
    ) -> None:
        self._rules = rules
        self._backend = backend
        self._llm = fallback
        self._settings = settings
        self._logger = get_logger("research_agent")

    async def classify(self, message: str) -> KindVerdict:
        if self._llm.refused(message):
            return KindVerdict(decision="refuse", reason="matches a refusal pattern", source="rule")
        if self._rules is not None:
            decided = self._rules.verdict(message)
            if decided is not None:
                days = self._llm.window_days(message)
                return self._llm.normalise(
                    decided.model_copy(update={"recent_days": days}), message
                )
        if self._backend is None:
            return await self._llm.classify(message)
        try:
            candidates = await self._backend.classify(message)
        except Exception as exc:
            self._logger.warning("Kind encoder unavailable; the LLM answers", error=str(exc))
            return await self._llm.classify(message)
        read = _read(candidates)
        verdict = self._verdict(read, message)
        if verdict is not None:
            return verdict
        fallback = await self._llm.classify(message)
        return fallback if read is None else fallback.model_copy(update=read.as_fields())

    def _verdict(self, read: EncoderRead | None, message: str) -> KindVerdict | None:
        if read is None:
            return None
        decided = (
            read.share >= self._settings.min_confidence
            and read.share - read.runner_up >= self._settings.min_margin
        )
        self._logger.info(
            "Kind encoder decided" if decided else "Kind encoder unsure; the LLM answers",
            label=read.label, p=round(read.share, 3), runner_up=round(read.runner_up, 3),
        )  # fmt: skip
        if not decided:
            return None
        verdict = KindVerdict(
            decision="scopable",
            archetype=read.label,  # type: ignore[arg-type]
            topic=message.strip(),
            recent_days=self._llm.window_days(message),
            reason=f"p={read.share:.2f} ({read.label})",
            source="encoder",
            source_model=str(getattr(self._backend, "model_name", "")),
            **read.as_fields(),
        )
        return self._llm.normalise(verdict, message)
