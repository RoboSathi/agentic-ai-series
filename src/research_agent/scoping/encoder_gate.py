"""The scope gate's fast path: a fine-tuned BERT-family classifier in front of the LLM gate.

Routing a first message is a classification problem — Q&A, or which kind of research — and a
120B model is a heavy tool for it (about 1,400 tokens and several seconds a turn). This is the
same shape as the Customer Support agent's front door: a small encoder fine-tuned on this
agent's own labels answers when it is sure, and everything it is unsure about goes to the LLM.

    refusal patterns   deterministic, first — no model is asked whether to help build a weapon
    time window        deterministic ("last 1 month" -> 30); it is not a classification problem
    cue rules          deterministic: a message that plainly names a research kind (rule_gate.py)
    encoder            a label and a softmax score; confident AND unambiguous -> answer here
    LLM gate           the fallback, for low confidence, a narrow margin, or any failure

The verdict says which path answered (`source`: rule, encoder or llm) and the card shows it.
"""

from typing import Protocol

from agentic_commons.intent.types import IntentCandidate
from agentic_commons.logging import get_logger, instrument
from research_agent.config.settings import EncoderGateSettings
from research_agent.domain.scope import QA, RESEARCH_ARCHETYPES, GateVerdict, Mode
from research_agent.scoping.gate import LlmArchetypeClassifier
from research_agent.scoping.rule_gate import RuleGate

# The classifier's labels: `qa`, then one per research archetype (`general` included).
LABELS = (QA, *RESEARCH_ARCHETYPES)


class IntentBackend(Protocol):
    """What the gate needs of an encoder (`HuggingFaceIntentClassifier` satisfies it)."""

    async def classify(self, text: str) -> list[IntentCandidate]: ...


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
        verdict = self._verdict(_research_kinds(candidates), message)
        return verdict if verdict is not None else await self._llm.classify(message, "research")

    def _verdict(self, candidates: list[IntentCandidate], message: str) -> GateVerdict | None:
        ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
        if not ranked or ranked[0].intent not in LABELS:
            return None
        top = ranked[0]
        runner_up = ranked[1].score if len(ranked) > 1 else 0.0
        if (
            top.score < self._settings.min_confidence
            or top.score - runner_up < self._settings.min_margin
        ):
            self._logger.info(
                "Encoder gate unsure; the LLM gate answers",
                label=top.intent, p=round(top.score, 3), runner_up=round(runner_up, 3),
            )  # fmt: skip
            return None
        mode: Mode = "qa" if top.intent == QA else "research"
        verdict = GateVerdict(
            decision="scopable",
            mode=mode,
            archetype=top.intent,  # type: ignore[arg-type]
            topic=message.strip(),
            recent_days=self._llm.window_days(message),
            reason=f"p={top.score:.2f} ({top.intent})",
            source="encoder",
            source_model=str(getattr(self._backend, "model_name", "")),
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
