"""The scope gate's fast path: a local intent classifier in front of the LLM gate.

Routing a first message is a classification problem — Q&A, or which kind of research — and a
120B model is a heavy tool for it (about 1,400 tokens and several seconds a turn). This is the
same shape as the Customer Support agent's front door: a small local encoder (e5 embeddings,
voting among the nearest of this agent's own labelled examples; nothing is fine-tuned) answers
when it is sure, and everything it is unsure about goes to the LLM.

    refusal patterns   deterministic, first — no model is asked whether to help build a weapon
    time window        deterministic ("last 1 month" -> 30); it is not a classification problem
    cue rules          deterministic: a message that plainly names a research kind (rule_gate.py)
    encoder            a label and a softmax score; confident AND unambiguous -> answer here
    LLM gate           the fallback, for low confidence, a narrow margin, or any failure

The verdict says which path answered (`source`: rule, encoder or llm) and the card shows it.
"""

from typing import Any, Protocol

from pydantic import BaseModel

from agent_platform.intent.types import IntentCandidate
from agent_platform.logging import get_logger, instrument
from research_agent.config.settings import EncoderGateSettings
from research_agent.domain.scope import QA, RESEARCH_ARCHETYPES, GateVerdict, Mode
from research_agent.scoping.gate import LlmArchetypeClassifier
from research_agent.scoping.rule_gate import RuleGate

# The classifier's labels: `qa`, then one per research archetype (`general` included).
LABELS = (QA, *RESEARCH_ARCHETYPES)


class IntentBackend(Protocol):
    """What the gate needs of an encoder (`HuggingFaceIntentClassifier` satisfies it)."""

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
class EncoderGateClassifier:
    """Answers confidently-classified first messages itself; defers the rest to the LLM gate."""

    def __init__(
        self,
        backend: IntentBackend | None,
        fallback: LlmArchetypeClassifier,
        settings: EncoderGateSettings,
        rules: RuleGate | None = None,
    ) -> None:
        self._rules = rules
        self._backend = backend
        self._llm = fallback
        self._settings = settings
        self._logger = get_logger("research_agent")

    async def classify(self, message: str, forced: Mode | None = None) -> GateVerdict:
        if self._llm.refused(message):
            return GateVerdict(decision="refuse", reason="matches a refusal pattern", source="rule")
        if forced != "research":
            return self._llm.quick(message)  # a plain question: no classifier, no model
        if self._rules is not None:
            decided = self._rules.verdict(message)
            if decided is not None:
                days = self._llm.window_days(message)
                return self._llm.normalise(
                    decided.model_copy(update={"recent_days": days}), message
                )
        if self._backend is None:
            return await self._llm.classify(message, "research")
        try:
            candidates = await self._backend.classify(message)
        except Exception as exc:
            self._logger.warning("Encoder gate unavailable; the LLM gate answers", error=str(exc))
            return await self._llm.classify(message, "research")
        read = _read(_research_kinds(candidates))
        verdict = self._verdict(read, message)
        if verdict is not None:
            return verdict
        fallback = await self._llm.classify(message, "research")
        return fallback if read is None else fallback.model_copy(update=read.as_fields())

    def _verdict(self, read: EncoderRead | None, message: str) -> GateVerdict | None:
        if read is None:
            return None
        decided = (
            read.share >= self._settings.min_confidence
            and read.share - read.runner_up >= self._settings.min_margin
        )
        self._logger.info(
            "Encoder gate decided" if decided else "Encoder gate unsure; the LLM gate answers",
            label=read.label, p=round(read.share, 3), runner_up=round(read.runner_up, 3),
        )  # fmt: skip
        if not decided:
            return None
        mode: Mode = "qa" if read.label == QA else "research"
        verdict = GateVerdict(
            decision="scopable",
            mode=mode,
            archetype=read.label,  # type: ignore[arg-type]
            topic=message.strip(),
            recent_days=self._llm.window_days(message),
            reason=f"p={read.share:.2f} ({read.label})",
            source="encoder",
            source_model=str(getattr(self._backend, "model_name", "")),
            **read.as_fields(),
        )
        return self._llm.normalise(verdict, message)


def _research_kinds(candidates: list[IntentCandidate]) -> list[IntentCandidate]:
    """The candidates for a message already known to be research, rescaled to sum to one.

    `/research` settles *whether* it is research; what is left to decide is which kind, so the
    quick-answer label is out of the running.
    """
    kinds = [candidate for candidate in candidates if candidate.intent != QA]
    total = sum(candidate.score for candidate in kinds)
    if not total:
        return kinds
    return [
        IntentCandidate(intent=candidate.intent, score=candidate.score / total)
        for candidate in kinds
    ]
