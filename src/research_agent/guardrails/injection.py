"""Fetched-content defence, layer 2: the injection screen over every fetched page.

The platform's prompt-injection detector reads every fetched page; a flagged
page is dropped, logged as a security event, and counted as a rejected
source. It is a guardrail, so it lives with the guardrails; the research
engine and the tools are handed one rather than reaching into each other.
"""

import re
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from agent_platform.logging import instrument
from agent_platform.security.guard_types import GuardVerdict
from agent_platform.security.prompt_injection.detector import InstructionDataSeparator


class InjectionGuardModel(Protocol):
    """A hosted injection/jailbreak model (the platform's `HostedJailbreakGuard`)."""

    @property
    def model(self) -> str: ...

    async def classify(self, text: str) -> GuardVerdict: ...


class InjectionVerdict(BaseModel):
    """What the screen decided about one page, and on which evidence."""

    decision: Literal["clean", "kept", "dropped"] = "clean"
    phrases: list[str] = Field(default_factory=list)
    score: float | None = None  # the highest model score seen
    model: str = ""
    why: str = ""

    @property
    def drop(self) -> bool:
        return self.decision == "dropped"


@instrument("research_agent")
class InjectionScreen:
    """Layer 2: a phrase check over fetched text, with a hosted model as the second opinion.

    A phrase hit alone is a weak signal ("you are now" turns up in ordinary prose), so with a
    model configured the phrase is only a feature: the model reads the text around it and the
    page is dropped only if the model agrees. With `scan_all` the model also reads pages the
    phrase check passed, which is how an attack in no listed wording is caught.

    Fail-safe either way: a phrase hit whose model check is unavailable still drops the page,
    exactly as it did before a model was added.
    """

    def __init__(
        self,
        separator: InstructionDataSeparator,
        guard: InjectionGuardModel | None = None,
        *,
        threshold: float = 0.5,
        scan_all: bool = False,
        window_chars: int = 2000,
        max_windows: int = 4,
    ) -> None:
        self._separator = separator
        self._guard = guard
        self._threshold = threshold
        self._scan_all = scan_all
        self._window = window_chars
        self._max_windows = max_windows

    def flagged(self, text: str) -> list[str]:
        """The injection phrasings found (empty when the page looks clean)."""
        return self._separator.find_phrases(text)

    async def assess(self, text: str) -> InjectionVerdict:
        phrases = self.flagged(text)
        if self._guard is None:
            if not phrases:
                return InjectionVerdict()
            return InjectionVerdict(decision="dropped", phrases=phrases, why="phrase check")
        if not phrases and not self._scan_all:
            return InjectionVerdict(model=self._guard.model)
        best: float | None = None
        unavailable = False
        for window in self._windows(text, phrases):
            verdict = await self._guard.classify(window)
            if not verdict.available:
                unavailable = True
                continue
            if verdict.score is not None:
                best = verdict.score if best is None else max(best, verdict.score)
            if verdict.flagged or (verdict.score is not None and verdict.score >= self._threshold):
                return self._result(
                    "dropped",
                    phrases,
                    verdict.score if verdict.score is not None else best,
                    "model flagged it",
                )
        model = self._guard.model
        if phrases and unavailable:
            return InjectionVerdict(
                decision="dropped", phrases=phrases, model=model,
                why="phrase hit and the injection model was unavailable",
            )  # fmt: skip
        if phrases:
            return self._result("kept", phrases, best, "model scored the context as benign")
        return InjectionVerdict(score=best, model=model)

    def _result(
        self,
        decision: Literal["kept", "dropped"],
        phrases: list[str],
        score: float | None,
        why: str,
    ) -> InjectionVerdict:
        assert self._guard is not None
        shown = f" (score {score:.2f})" if score is not None else ""
        return InjectionVerdict(
            decision=decision,
            phrases=phrases,
            score=score,
            model=self._guard.model,
            why=why + shown,
        )

    def _windows(self, text: str, phrases: list[str]) -> list[str]:
        """Text around each flagged phrase; with none, evenly spaced windows over the page."""
        half = self._window // 2
        found: list[str] = []
        for phrase in phrases:
            hit = re.search(re.escape(phrase), text, re.IGNORECASE)
            if hit is not None:
                found.append(text[max(0, hit.start() - half) : hit.start() + half])
        if phrases:
            return found[: self._max_windows] or [text[: self._window]]
        count = min(self._max_windows, max(1, -(-len(text) // self._window)))
        step = max(len(text) - self._window, 0) / max(count - 1, 1)
        return [text[int(i * step) : int(i * step) + self._window] for i in range(count)]
