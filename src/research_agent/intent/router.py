"""Route a message: answer it directly, start research, or refuse.

The same layering as the Customer Support router, cheapest first:

    refusal patterns     deterministic — no model is asked whether to help build a weapon
    research cues        deterministic — the message asks for a research run or names a kind
    encoder              a fine-tuned BERT-family label and score; confident AND unambiguous wins
    LLM                  only for what the encoder is unsure of or could not read

Answering directly is the default mode, so there are no cues for it: with no encoder trained, a
message no rule claims is answered directly, which is what a model call to decide would say.
"""

import re
from typing import Literal

from pydantic import BaseModel

from agentic_commons.intent.types import IntentCandidate
from agentic_commons.logging import get_logger, instrument
from research_agent.config.settings import IntentRouterSettings, ScopingSettings
from research_agent.prompts import intent as intent_prompt
from research_agent.scoping.encoder_gate import IntentBackend
from research_agent.scoping.rule_gate import RuleGate
from research_agent.structured import StructuredCaller

Label = Literal["answer", "research", "refuse"]
Source = Literal["rule", "encoder", "llm", "default"]


class IntentRoute(BaseModel):
    """Where a message goes, and which layer decided."""

    label: Label
    source: Source
    reason: str = ""


@instrument("research_agent")
class ResearchIntentRouter:
    """Rules, then the encoder, then the LLM — each only for what the one before left open."""

    def __init__(
        self,
        settings: IntentRouterSettings,
        scoping: ScopingSettings,
        backend: IntentBackend | None,
        caller: StructuredCaller | None,
    ) -> None:
        self._settings = settings
        self._backend = backend
        self._caller = caller
        self._rules = RuleGate(scoping)
        self._refuse = _compile(scoping.refuse_patterns)
        self._research = _compile([rf"\b{re.escape(cue)}\b" for cue in scoping.research_cues])
        self._logger = get_logger("research_agent")

    async def route(self, message: str) -> IntentRoute:
        decided = self._by_rule(message)
        if decided is not None:
            return self._logged(decided, message)
        if self._backend is None:
            return self._logged(self._default("no encoder is trained"), message)
        try:
            candidates = await self._backend.classify(message)
        except Exception as exc:
            self._logger.warning("Intent encoder unavailable", error=str(exc))
            return self._logged(await self._unsure(message, []), message)
        decided = self._by_encoder(candidates)
        return self._logged(decided or await self._unsure(message, candidates), message)

    # -- layers ---------------------------------------------------------------------------

    def _by_rule(self, message: str) -> IntentRoute | None:
        if _hits(self._refuse, message):
            return IntentRoute(label="refuse", source="rule", reason="matches a refusal pattern")
        cues = _hits(self._research, message)
        if len(cues) >= self._settings.research_cue_min_hits:
            return IntentRoute(label="research", source="rule", reason=", ".join(cues))
        kind = self._rules.verdict(message)
        if kind is not None:
            return IntentRoute(label="research", source="rule", reason=kind.reason)
        return None

    def _by_encoder(self, candidates: list[IntentCandidate]) -> IntentRoute | None:
        """Answer against research, each side the sum of its labels' scores; decided when one is sure."""
        answer = sum(
            c.score for c in candidates if c.intent in self._settings.encoder_answer_labels
        )
        research = sum(
            c.score for c in candidates if c.intent not in self._settings.encoder_answer_labels
        )
        total = answer + research
        if not total:
            return None
        label: Label = "answer" if answer >= research else "research"
        share = max(answer, research) / total
        if share < self._settings.min_confidence:
            return None
        return IntentRoute(label=label, source="encoder", reason=f"p={share:.2f}")

    async def _unsure(self, message: str, candidates: list[IntentCandidate]) -> IntentRoute:
        if self._caller is None or not self._settings.llm_fallback:
            return self._default("the encoder was unsure")
        hint = [{"label": item.intent, "score": round(item.score, 3)} for item in candidates]
        try:
            answer = await self._caller.call(
                intent_prompt.build(message, hint), intent_prompt.IntentAnswer
            )
        except Exception as exc:
            self._logger.warning("Intent LLM unavailable; answering directly", error=str(exc))
            return self._default("the LLM was unavailable")
        return IntentRoute(label=answer.label, source="llm", reason=answer.reason)

    @staticmethod
    def _default(reason: str) -> IntentRoute:
        return IntentRoute(label="answer", source="default", reason=reason)

    def _logged(self, route: IntentRoute, message: str) -> IntentRoute:
        self._logger.info(
            "Intent route", label=route.label, source=route.source, reason=route.reason,
            text=message,
        )  # fmt: skip
        return route


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def _hits(patterns: list[re.Pattern[str]], message: str) -> list[str]:
    return [match.group(0) for pattern in patterns if (match := pattern.search(message))]
