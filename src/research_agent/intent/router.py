"""Route a message: answer it directly, start research, or refuse.

The same layering as the Customer Support router, cheapest first:

    refusal patterns     deterministic — no model is asked whether to help build a weapon
    research cues        deterministic — the message asks for a research run or names a kind
    encoder              nearest labelled examples; the winning side must clear its own bar
    LLM                  only for what the encoder is unsure of or could not read

The bars are asymmetric: research is the expensive, interview-driven path, so the encoder needs
more confidence to start one (`research_min_confidence`) than to answer (`min_confidence`).
Every layer leaves a step on the route, so the front-door card shows what each one made of it.

Answering directly is the default mode, so there are no cues for it: with no encoder trained, a
message no rule claims is answered directly, which is what a model call to decide would say.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field

from agent_platform.intent.types import IntentCandidate
from agent_platform.logging import get_logger, instrument
from research_agent.config.settings import IntentRouterSettings, ScopingSettings
from research_agent.llm.structured import StructuredCaller
from research_agent.prompts import intent as intent_prompt
from research_agent.scoping.encoder_gate import IntentBackend
from research_agent.scoping.rule_gate import RuleGate

Label = Literal["answer", "research", "refuse"]
Source = Literal["rule", "encoder", "llm", "default"]


class RouteStep(BaseModel):
    """What one layer made of the message: a card line key (`intent.<key>`), its tone, values."""

    key: str
    tone: str
    values: dict[str, str] = Field(default_factory=dict)


class IntentRoute(BaseModel):
    """Where a message goes, which layer decided, and what each layer on the way said."""

    label: Label
    source: Source
    reason: str = ""
    # The model that decided: the local encoder's, or the LLM's; empty when a rule did.
    model: str = ""
    steps: list[RouteStep] = Field(default_factory=list)


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
        steps: list[RouteStep] = []
        decided = self._by_rule(message, steps)
        if decided is not None:
            return self._logged(decided, message, steps)
        if self._backend is None:
            steps.append(RouteStep(key="encoder_off", tone="dim"))
            return self._logged(self._default("no encoder is trained"), message, steps)
        try:
            candidates = await self._backend.classify(message)
        except Exception as exc:
            self._logger.warning("Intent encoder unavailable", error=str(exc))
            steps.append(RouteStep(key="encoder_failed", tone="warn", values={"error": str(exc)}))
            return self._logged(await self._unsure(message, [], steps), message, steps)
        decided = self._by_encoder(candidates, steps)
        return self._logged(
            decided or await self._unsure(message, candidates, steps), message, steps
        )

    # -- layers ---------------------------------------------------------------------------

    def _by_rule(self, message: str, steps: list[RouteStep]) -> IntentRoute | None:
        if _hits(self._refuse, message):
            steps.append(RouteStep(key="refusal_hit", tone="fail"))
            return IntentRoute(label="refuse", source="rule", reason="matches a refusal pattern")
        steps.append(RouteStep(key="refusal_clear", tone="pass"))
        cues = _hits(self._research, message)
        if len(cues) >= self._settings.research_cue_min_hits:
            steps.append(RouteStep(key="cues_hit", tone="warn", values={"cues": ", ".join(cues)}))
            return IntentRoute(label="research", source="rule", reason=", ".join(cues))
        kind = self._rules.verdict(message)
        if kind is not None:
            values = {"kind": str(kind.archetype), "cues": kind.reason}
            steps.append(RouteStep(key="kind_hit", tone="warn", values=values))
            return IntentRoute(label="research", source="rule", reason=kind.reason)
        steps.append(RouteStep(key="cues_clear", tone="dim"))
        return None

    def _by_encoder(
        self, candidates: list[IntentCandidate], steps: list[RouteStep]
    ) -> IntentRoute | None:
        """Answer against research, each side the sum of its labels' scores; decided when sure."""
        answer = sum(
            c.score for c in candidates if c.intent in self._settings.encoder_answer_labels
        )
        research = sum(
            c.score for c in candidates if c.intent not in self._settings.encoder_answer_labels
        )
        total = answer + research
        if not total:
            steps.append(RouteStep(key="encoder_empty", tone="warn"))
            return None
        label: Label = "answer" if answer >= research else "research"
        share = max(answer, research) / total
        bar = self._bar(label)
        values = {
            "answer": f"{answer / total:.0%}",
            "research": f"{research / total:.0%}",
            "label": label,
            "bar": f"{bar:.0%}",
        }
        if share < bar:
            steps.append(RouteStep(key="encoder_unsure", tone="warn", values=values))
            return None
        steps.append(RouteStep(key="encoder_decided", tone="pass", values=values))
        model = str(getattr(self._backend, "model_name", ""))
        return IntentRoute(label=label, source="encoder", reason=f"p={share:.2f}", model=model)

    def _bar(self, label: Label) -> float:
        """The share a side needs to decide: starting research asks for more than answering."""
        if label == "research":
            return self._settings.research_min_confidence
        return self._settings.min_confidence

    async def _unsure(
        self, message: str, candidates: list[IntentCandidate], steps: list[RouteStep]
    ) -> IntentRoute:
        if self._caller is None or not self._settings.llm_fallback:
            steps.append(RouteStep(key="llm_off", tone="dim"))
            return self._default("the encoder was unsure")
        hint = [{"label": item.intent, "score": round(item.score, 3)} for item in candidates]
        try:
            answer = await self._caller.call(
                intent_prompt.build(message, hint), intent_prompt.IntentAnswer
            )
        except Exception as exc:
            self._logger.warning("Intent LLM unavailable; answering directly", error=str(exc))
            steps.append(RouteStep(key="llm_failed", tone="warn", values={"error": str(exc)}))
            return self._default("the LLM was unavailable")
        values = {"label": answer.label, "reason": answer.reason}
        steps.append(RouteStep(key="llm_decided", tone="pass", values=values))
        model = self._caller.model_key(intent_prompt.ROLE)
        return IntentRoute(label=answer.label, source="llm", reason=answer.reason, model=model)

    @staticmethod
    def _default(reason: str) -> IntentRoute:
        return IntentRoute(label="answer", source="default", reason=reason)

    def _logged(self, route: IntentRoute, message: str, steps: list[RouteStep]) -> IntentRoute:
        route = route.model_copy(update={"steps": steps})
        scores = next((step.values for step in steps if step.key.startswith("encoder_")), {})
        self._logger.info(
            "Intent route", decision=route.label, decided_by=route.source, model=route.model,
            answer=scores.get("answer", ""), research=scores.get("research", ""),
            bar=scores.get("bar", ""), reason=route.reason, text=message,
        )  # fmt: skip
        return route


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def _hits(patterns: list[re.Pattern[str]], message: str) -> list[str]:
    return [match.group(0) for pattern in patterns if (match := pattern.search(message))]
